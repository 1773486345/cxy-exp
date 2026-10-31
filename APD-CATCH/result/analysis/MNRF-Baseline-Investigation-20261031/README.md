# MNRF baseline investigation

This is a read-only investigation generated on e7532c231f046cf844e43afa4ac65c878c3112c9 with `/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python` (Python 3.8.20). Current catch_env packages: numpy=1.24.4, pandas=1.5.3, scikit-learn=1.3.2, torch=2.4.1, salesforce-merlion=1.3.1, pyod=2.1.0, tods=not-importable. It did not train, encode, fit a head, recalculate a cache, or modify an existing result. Evidence labels are `direct-code`, `direct-record`, `inference`, and `unknown`.

## Scientific conclusion and comparison boundary

The primary numerical baseline reference is the fixed [ICLR 2025 conference-paper Table 2](https://proceedings.iclr.cc/paper_files/paper/2025/file/2b25c39788e5cf11d3541de433ebf4c0-Paper-Conference.pdf) for all eight scoped datasets and all 16 methods. The PDF reports `A-R (AUC-ROC)` and `Aff-F`; its dataset label `Credit` is mapped to the repository task `Creditcard`. Paper values are stored only in the numeric `paper_auc_roc` and `paper_affiliation_f` columns. The server scripts and configurations are treated as the authoritative implementation settings; a method does not need a local checkpoint or successful run to be included in the settings table.

The report keeps three protocol classes separate: `paper reference` (the Table 2 PDF), `original-protocol local archive` (CATCH-master score and label runs), and `common-protocol diagnostic` (the APD-CATCH frozen-source comparison). Local archive values never populate the paper columns. `baseline_settings.csv` has one `detect_score` and one `detect_label` row for every method x dataset pair, so an explicit command override is distinguishable from config/class defaults.

The existing eight-dataset APD-CATCH common contract (MNRF Full, CATCH-common, IsolationForest, and PCA) is reported separately as a diagnostic comparison. Its rows use the same frozen Full source, endpoint coordinates, independent calibration convention, and common metric entry point. This is a protocol comparison, not a claim that the historical implementations are identical.

Historical CATCH is not one jointly evaluated row. `CATCH-master/result/score/CATCH` and `.../label/CATCH` are separate runs; the archive summary records score AUC from the score run and Affiliation-F from the label run. Those values are original-protocol records, not paper values, and cannot be joined with a common-protocol AP or point metric. The CATCH paper table is a third category and is not a rerun.

Most methods have no local eight-dataset result in the scoped APD-CATCH result roots. This is an evidence-availability fact, not a failed experiment. Their server scripts and code paths are still recorded for all eight datasets in `baseline_settings.csv`; the paper table remains the numerical reference. Historical TODS artifacts in CATCH-master are sparse and do not establish an eight-dataset matrix. `tods` is not importable in the current `catch_env`, while current Merlion is 1.3.1 and PyOD is 2.1.0; no historical environment recovery is attempted.

The available common-protocol snapshot is:

| Dataset | MNRF Full AUC/AP | CATCH-common AUC/AP | IF AUC/AP | PCA AUC/AP |
|---|---:|---:|---:|---:|
| PSM | 0.7449263160543438/0.437281131313489 | 0.6348493183304912/0.4240888404343812 | 0.6838996572568468/0.4396883617689315 | 0.7925755416458282/0.5348424872628248 |
| MSL | 0.602445207747461/0.1590062828565859 | 0.6553048696275066/0.16958545733253008 | 0.60400746476379/0.14227718294478026 | 0.5052713757147835/0.14401553140684592 |
| GECCO | 0.4424521821007449/0.22330328601085492 | 0.9713602630034501/0.4671908750658895 | 0.956131446078608/0.43982960492109335 | 0.6795650390685988/0.3618443522146658 |
| Genesis | 0.8995684848484848/0.04680510216819125 | 0.758450101010101/0.01443279437085916 | 0.6501236363636363/0.005485510713137226 | 0.9875620202020201/0.5155251498045019 |
| NYC | 0.6913389836317612/0.04408917284828918 | 0.6849371073801515/0.04995928028494186 | 0.6656617067426569/0.03782264534169968 | 0.5171700034763534/0.026368639864231172 |
| CalIt2 | 0.7130399038765574/0.0539712685708693 | 0.8358123063302345/0.10698591547329232 | 0.8001897173211915/0.10114745115463555 | 0.7327863150572314/0.08011683650834776 |
| Creditcard | 0.920187986500546/0.0510074536414816 | 0.9593156130904322/0.09120467227791135 | 0.9550520212852475/0.09650064918227143 | 0.800732065970704/0.045347014929507774 |
| CICIDS | 0.5388251328959158/0.0009173981247266992 | 0.6937611144750719/0.0013601503766721241 | 0.6746888902550002/0.0012951081143985665 | 0.5738736137464496/0.0010659702564034075 |

These rows support ranking and alarm comparisons only within the common contract. They do not establish a 16-method comparison, and they do not prove a representation or causal mechanism. A threshold is one alarm-metric difference; changing a threshold on the same continuous score cannot change ROC-AUC or AP. Training configuration, score formula, model context, window alignment and threshold source also differ here, and no causal ordering of those differences is claimed without a controlled intervention.

## Time and score alignment

The previous generic “past/current, no future” wording is not used. The native CATCH `mode=thre` path uses non-overlapping full windows. For sequence length `L=4`, the window `[0,1,2,3]` produces flattened scores for positions 0, 1, 2 and 3; the score reported for position 0 can therefore read through input index 3. MNRF's frozen endpoint score for endpoint 3 reads the causal window ending at 3 and emits only that endpoint. The CATCH path is offline full-window scoring, not endpoint-causal scoring or a claim of leakage. `SegLoader` omits a remainder shorter than `L`, and the native score path does not invent scores for omitted positions. Each method's actual context, stride, overlap, cold-start, tail and missing-score behavior is recorded in `method_mechanisms.csv`; adapter-dependent cases remain explicitly unknown.

The project-local TODS adapter is under `ts_benchmark/baselines/tods/third_party`; a failed top-level `import tods` in `catch_env` does not mean the repository entry is unavailable. Merlion and PyOD behavior is described from the currently installed packages (Merlion 1.3.1 and PyOD 2.1.0); package-relative `merlion/...` references in `method_mechanisms.csv` point to that environment's site-packages source. Historical package versions are not inferred. The one missing scoped entry (`CICIDS` `detect_label` `Isolation Forest`) is recorded as missing, not as a failed experiment.

## Dataset-specific qualifications

- **CalIt2 (direct-record):** the old local input is a modulo-10 conversion of UCI counts. The RawCount rerun report records MNRF AUC/AP changing from 0.713040/0.053971 to 0.853538/0.170287, CATCH from 0.835812/0.106986 to 0.881578/0.281496, IF from 0.800190/0.101147 to 0.890166/0.253774, and PCA from 0.732786/0.080117 to 0.822361/0.146648. The RawCount report's sentence saying “PCA AP decreases slightly” contradicts its own numbers; the numbers imply an increase. The PCA comparison must remain three-way: old full-dimensional reconstruction, old data-repair residual PCA, and RawCount residual PCA. The historical file is left unchanged.
- **MSL (direct-record/unknown):** 27 telemetry sequences have 55 input fields each. `col_0` is a candidate telemetry target and `col_1..54` are not proven exogenous conditions. The existing col_0 aggregation is a post-hoc diagnostic on concatenated data, not a segmented retraining result.
- **NYC (direct-record):** three fields are passenger count, time-of-day, and day-of-week. The field-role diagnostic excluded evidenced calendar fields from target aggregation and did not improve results; this does not prove that removing context is generally correct.
- **Creditcard and CICIDS (direct-record):** the frozen inputs do not retain card/customer or flow/session identifiers sufficient to reconstruct entity continuity. Row adjacency must not be presented as a verified per-entity time series.

## What is confirmed versus unknown

The code directly confirms reconstruction versus discrepancy heads, losses, score direction, provider window behavior, and label-mode percentile thresholding for the 16 methods; see `method_mechanisms.csv`. The server entry script, exact command, hyperparameters, special handling, and paper-reference availability for every method x dataset pair are in `baseline_settings.csv`. Existing common-run source/metric bindings and the separate CATCH archive score/label runs are diagnostic evidence only.

Unknowns include the final selection rule when a method emits several anomaly ratios and results for methods without a local run. Historical package archaeology is outside this round; the current `catch_env` version is recorded only to explain current server behavior. No test-label selection was inferred merely from an `anomaly_ratio` list; the code does show train+test score distributions in label-mode thresholding, which is a protocol difference from the independent q=.995 common calibration.

## Minimum next experiment

Use the eight registered Full sources and one fixed common contract. Preserve raw file hashes, original row order and known contamination; use the saved train/validation/calibration/test endpoints and valid masks; prohibit crossing verified MSL sequence boundaries in the evaluation diagnostic, while clearly separating that from a segmented retrain. Fit each method with its native head and loss on training data only, use validation only for early stopping, derive one independent q=.995 threshold from the calibration segment, and predict with strict `score > threshold`. Do not force GMM, PCA, or a common head onto methods whose native scores differ. This is a minimum diagnostic, not a default eight-dataset x 16-method rerun: first freeze an auxiliary decoder/score-head diagnostic and the corrected CalIt2 RawCount input, then compare the resulting score and alarm changes under the same coordinates.

Evaluate ROC-AUC/AP on the exact common scored points; report point precision/recall/F1/FPR and Affiliation precision/recall/F using the same valid contiguous segments, gaps, entity boundaries, and no point adjustment. Publish paper values, original-protocol archive values, and common-protocol values in separate tables. The unresolved choices before execution are (a) whether calendar/context fields are scored or only supplied as conditions for each native method, and (b) what explicit boundary contract is defensible for Creditcard/CICIDS without entity/session IDs.

No real benchmark was started in this investigation. No method is marked failed merely because a local result is absent.
