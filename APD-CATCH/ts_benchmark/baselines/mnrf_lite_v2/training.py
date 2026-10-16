"""Purged splits, two-stage optimization, frozen feature cache and calibration."""

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import MNRFConfig
from .fusion import AlternatingAffineFlow
from .normalization import FixedFeatureStandardizer, FixedInputStandardizer
from .views import FourViewEncoder


def purged_ranges(length, window):
    gap = window - 1
    available = int(length) - 2 * gap
    if available <= 0:
        raise ValueError(f"training segment length {length} is too short for two {gap}-point purge gaps")
    train_n = int(math.floor(available * 0.8))
    valid_n = int(math.floor(available * 0.1))
    calibration_n = available - train_n - valid_n
    if min(train_n, valid_n, calibration_n) < window:
        raise ValueError(
            f"training segment length {length} cannot produce train/validation/calibration "
            f"windows of length {window} after {gap}-point purge gaps"
        )
    train = (0, train_n)
    validation = (train_n + gap, train_n + gap + valid_n)
    calibration = (validation[1] + gap, validation[1] + gap + calibration_n)
    return {
        "gap": gap,
        "input_length": int(length),
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "window_counts": {
            name: end - start - window + 1
            for name, (start, end) in (
                ("train", train), ("validation", validation), ("calibration", calibration)
            )
        },
    }


class EndWindowDataset(Dataset):
    def __init__(self, values, start, end, window):
        self.values = values
        self.start, self.end, self.window = int(start), int(end), int(window)
        self.endpoints = np.arange(self.start + self.window - 1, self.end, dtype=np.int64)

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        endpoint = int(self.endpoints[index])
        return torch.from_numpy(np.asarray(self.values[endpoint - self.window + 1:endpoint + 1], dtype=np.float32)), endpoint


class GaussianAuxiliaryDecoder(nn.Module):
    def __init__(self, variables, hidden=32, view_dim=8):
        super().__init__()
        self.task_identity = nn.Parameter(torch.randn(3, 8) * 0.02)
        self.decoder = nn.Sequential(nn.Linear(view_dim + 8 + hidden + 1, 64), nn.GELU(), nn.Linear(64, 2))

    def loss(self, z, target, relative_position, variable_identity, task_index):
        batch, variables, width = z.shape
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        if relative_position.ndim == 2:
            positions = relative_position.unsqueeze(-1).unsqueeze(-1)
        elif relative_position.ndim == 3:
            positions = relative_position.unsqueeze(-1)
        else:
            raise ValueError("relative_position must have shape [N,D] or [N,D,K]")
        count = target.size(-1)
        z_rep = z.unsqueeze(2).expand(-1, -1, count, -1)
        task = self.task_identity[task_index].view(1, 1, 1, -1).expand(batch, variables, count, -1)
        identity = variable_identity.view(1, variables, 1, -1).expand(batch, -1, count, -1)
        decoder_input = torch.cat((z_rep, task, identity, positions), dim=-1)
        output = self.decoder(decoder_input)
        mean = output[..., 0]
        sigma = 0.05 + torch.nn.functional.softplus(output[..., 1])
        nll = (target - mean).square() / (2.0 * sigma.square()) + torch.log(sigma) + 0.5 * math.log(2.0 * math.pi)
        return nll.mean()


class MNRFTrainer:
    def __init__(self, config: MNRFConfig, variables: int, device: torch.device):
        self.config = config
        self.variables = variables
        self.device = device
        self.encoder = FourViewEncoder(variables, config.hidden_dim, config.view_dim).to(device)
        self.auxiliary = GaussianAuxiliaryDecoder(variables).to(device)
        self.flow = AlternatingAffineFlow(config.hidden_dim).to(device)
        self.fixed_input = FixedInputStandardizer()
        self.fixed_feature = FixedFeatureStandardizer(config.feature_std_floor)
        self.noise_vectors = None
        self.calibration_center = None
        self.calibration_scale = None
        self.threshold = None
        self.history = {"stage1": [], "stage2": []}
        self.split_info = None

    def _windows(self, values, segment, shuffle, seed=None):
        start, end = segment
        dataset = EndWindowDataset(values, start, end, self.config.seq_len)
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )

    def _pretrain_batch(self, windows, task, endpoints=None, fixed_seed=None):
        values = windows.to(self.device)
        valid = torch.ones_like(values, dtype=torch.bool)
        mask_start = None
        if task == "pattern":
            if fixed_seed is None:
                mask_start = torch.randint(0, 25, (values.size(0), self.variables), device=self.device)
            else:
                ep = torch.as_tensor(endpoints, device=self.device).view(-1, 1)
                variable = torch.arange(self.variables, device=self.device).view(1, -1)
                mask_start = ((ep * 1103515245 + variable * 12345 + int(fixed_seed)) % 25).long()
        z, target, relative = self.encoder.encode_pretrain(values, valid, task, mask_start=mask_start)
        task_index = {"evolution": 0, "pattern": 1, "relation": 2}[task]
        return self.auxiliary.loss(z, target, relative, self.encoder.point.variable_identity, task_index)

    @torch.no_grad()
    def _validation_stage1(self, values, segment):
        self.encoder.eval()
        self.auxiliary.eval()
        loader = self._windows(values, segment, shuffle=False)
        per_task = {name: [0.0, 0] for name in ("evolution", "pattern", "relation")}
        for windows, endpoints in loader:
            endpoints_np = endpoints.numpy()
            for task in per_task:
                loss = self._pretrain_batch(windows, task, endpoints_np, fixed_seed=self.config.seed)
                per_task[task][0] += float(loss.cpu()) * len(endpoints_np)
                per_task[task][1] += len(endpoints_np)
        task_losses = [total / count for total, count in per_task.values()]
        self.encoder.train()
        self.auxiliary.train()
        return float(np.mean(task_losses))

    def stage1(self, values, train_segment, validation_segment, out_dir):
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.auxiliary.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_loader = self._windows(values, train_segment, shuffle=True, seed=self.config.seed)
        best = float("inf")
        best_state = None
        patience = 0
        for epoch in range(self.config.stage1_epochs):
            self.encoder.train()
            self.auxiliary.train()
            task_losses = []
            for windows, _ in train_loader:
                task = random.choice(("evolution", "pattern", "relation"))
                optimizer.zero_grad(set_to_none=True)
                loss = self._pretrain_batch(windows, task)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite stage1 {task} loss")
                loss.backward()
                optimizer.step()
                task_losses.append(float(loss.detach().cpu()))
            validation_loss = self._validation_stage1(values, validation_segment)
            self.history["stage1"].append({
                "epoch": epoch + 1,
                "steps": len(task_losses),
                "train_nll": float(np.mean(task_losses)),
                "validation_macro_nll": validation_loss,
            })
            if validation_loss < best:
                best = validation_loss
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()},
                    "auxiliary": {k: v.detach().cpu().clone() for k, v in self.auxiliary.state_dict().items()},
                }
                patience = 0
            else:
                patience += 1
                if patience >= self.config.patience:
                    break
        if best_state is None:
            raise RuntimeError("stage1 did not produce a finite validation checkpoint")
        self.encoder.load_state_dict(best_state["encoder"])
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.encoder.eval()
        # The temporary auxiliary decoder is deliberately excluded from deployment.
        del self.auxiliary
        torch.save(best_state["encoder"], Path(out_dir) / "best_encoder.pt")
        return best

    @torch.no_grad()
    def extract_cache(self, values, segment, name, cache_dir):
        start, end = segment
        dataset = EndWindowDataset(values, start, end, self.config.seq_len)
        count = len(dataset)
        path = Path(cache_dir) / f"{name}_features.npy"
        indices_path = Path(cache_dir) / f"{name}_window_end_indices.npy"
        np.save(indices_path, dataset.endpoints)
        mmap = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32,
            shape=(count, self.variables, 25),
        )
        self.encoder.eval()
        cursor = 0
        for offset in range(0, count, self.config.cache_chunk_size):
            batch = [dataset[i][0] for i in range(offset, min(count, offset + self.config.cache_chunk_size))]
            tensor = torch.stack(batch).to(self.device)
            feature = self.encoder.encode_views(tensor)["features"]
            size = feature.size(0)
            mmap[cursor:cursor + size] = feature.cpu().numpy()
            cursor += size
        mmap.flush()
        return str(path), count, str(indices_path)

    def _fixed_noises(self):
        if self.noise_vectors is None:
            generator = torch.Generator(device="cpu").manual_seed(self.config.noise_seed)
            self.noise_vectors = torch.randn(4, 25, generator=generator) * self.config.train_noise_std
            self.noise_vectors = self.noise_vectors.to(self.device)
        return self.noise_vectors

    def _flow_nll(self, features, noise=None):
        identity = self.encoder.point.variable_identity
        if noise is not None:
            features = features + noise.view(1, 1, -1)
        return self.flow(features, identity)["nll"]

    @torch.no_grad()
    def _stage2_validation(self, features):
        self.flow.eval()
        noises = self._fixed_noises()
        total_loss = 0.0
        total_count = 0
        for offset in range(0, len(features), self.config.batch_size):
            batch = torch.as_tensor(np.asarray(features[offset:offset + self.config.batch_size]), device=self.device)
            values = [self._flow_nll(batch, noise).mean() for noise in noises]
            size = len(batch)
            total_loss += float(torch.stack(values).mean().cpu()) * size
            total_count += size
        self.flow.train()
        return total_loss / max(total_count, 1)

    def stage2(self, train_features, validation_features, out_dir):
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.encoder.eval()
        optimizer = torch.optim.AdamW(
            self.flow.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best = float("inf")
        best_state = None
        patience = 0
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 17)
        self._fixed_noises()
        for epoch in range(self.config.stage2_epochs):
            self.flow.train()
            indices = torch.randperm(len(train_features), generator=generator).numpy()
            losses = []
            for start in range(0, len(indices), self.config.batch_size):
                selection = indices[start:start + self.config.batch_size]
                feature = torch.as_tensor(np.asarray(train_features[selection]), device=self.device)
                noise = torch.randn(
                    feature.size(0), 1, 25, device=self.device
                ) * self.config.train_noise_std
                optimizer.zero_grad(set_to_none=True)
                result = self.flow(feature + noise, self.encoder.point.variable_identity)
                loss = result["nll"].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite stage2 flow NLL")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_loss = self._stage2_validation(validation_features)
            self.history["stage2"].append({
                "epoch": epoch + 1,
                "steps": len(losses),
                "train_nll": float(np.mean(losses)),
                "validation_fixed_noise_nll": validation_loss,
            })
            if validation_loss < best:
                best = validation_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.flow.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.config.patience:
                    break
        if best_state is None:
            raise RuntimeError("stage2 did not produce a finite validation checkpoint")
        self.flow.load_state_dict(best_state)
        torch.save(best_state, Path(out_dir) / "best_fusion.pt")
        return best

    @torch.no_grad()
    def per_variable_scores(self, features):
        noises = self._fixed_noises()
        results = []
        self.encoder.eval()
        self.flow.eval()
        for offset in range(0, len(features), self.config.batch_size):
            batch = torch.as_tensor(np.asarray(features[offset:offset + self.config.batch_size]), device=self.device)
            scores = torch.stack([self._flow_nll(batch, noise) for noise in noises]).mean(dim=0)
            results.append(scores.cpu().numpy())
        return np.concatenate(results, axis=0)

    def calibrate(self, validation_features, calibration_features):
        validation_scores = self.per_variable_scores(validation_features)
        self.calibration_center = np.median(validation_scores, axis=0).astype(np.float32)
        mad = np.median(np.abs(validation_scores - self.calibration_center), axis=0)
        self.calibration_scale = np.maximum(1.4826 * mad, self.config.score_mad_floor).astype(np.float32)
        calibration_scores = self.per_variable_scores(calibration_features)
        aggregate = np.max((calibration_scores - self.calibration_center) / self.calibration_scale, axis=1)
        self.threshold = float(np.quantile(aggregate, self.config.calibration_quantile))

    def fingerprint(self):
        digest = hashlib.sha256()
        for module in (self.encoder, self.flow):
            for key, value in sorted(module.state_dict().items()):
                digest.update(key.encode("utf-8"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
