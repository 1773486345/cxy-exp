import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters

from ts_benchmark.baselines.mnrf_view_probe.config import ViewProbeConfig
from ts_benchmark.baselines.mnrf_view_probe.probe import (
    _gmm_fit_values, _gmm_numeric_config, _gmm_score_values,
    _runtime_free_config, _validate_gmm_numeric_state,
)


def test_fixed_responsibilities_expose_float32_variance_cancellation():
    # One component is a singleton; another feature is almost constant around
    # a large offset. The exact same responsibilities isolate moment dtype
    # from any KMeans or random initialization change.
    x32 = np.column_stack((np.linspace(10000.0, 10000.01, 20),
                           np.linspace(1.0, 1.0001, 20))).astype(np.float32)
    responsibility = np.zeros((len(x32), 2), dtype=np.float64)
    responsibility[0, 0] = 1.0
    responsibility[1:, 1] = 1.0
    _, _, cov32 = _estimate_gaussian_parameters(x32, responsibility, 0.001, "diag")
    _, _, cov64 = _estimate_gaussian_parameters(
        x32.astype(np.float64), responsibility, 0.001, "diag")
    assert np.isfinite(cov32).all() and np.isfinite(cov64).all()
    assert np.array_equal(responsibility[:, 0], np.r_[1.0, np.zeros(19)])
    assert cov32[1, 0] < 0.0
    assert cov64[1, 0] >= 0.001


def test_float64_gmm_fit_and_likelihood_are_finite_for_singleton_and_near_constant_feature():
    values = np.asarray([[10000.0 + i * 0.001, 1.0 + i * 1e-6]
                         for i in range(20)] + [[10100.0, 1.2]], dtype=np.float32).astype(np.float64)
    model = GaussianMixture(n_components=2, covariance_type="diag", reg_covar=0.001,
                            n_init=1, max_iter=100, random_state=2021).fit(values)
    scores = model.score_samples(values)
    assert model.means_.dtype == np.float64
    assert model.covariances_.dtype == np.float64
    assert np.isfinite(model.covariances_).all() and np.all(model.covariances_ >= 0.001 - 1e-12)
    assert np.isfinite(scores).all()
    assert np.min(model.weights_) < 0.1
    _validate_gmm_numeric_state(model, "float64")


def test_gmm_compute_dtype_is_part_of_resume_identity_and_legacy_defaults_to_float32():
    legacy = ViewProbeConfig().to_dict()
    legacy.pop("gmm_compute_dtype")
    old = ViewProbeConfig().to_dict()
    precise = ViewProbeConfig(gmm_compute_dtype="float64").to_dict()
    assert _runtime_free_config(legacy) == _runtime_free_config(old)
    assert _gmm_numeric_config(legacy) == _gmm_numeric_config(old)
    assert _gmm_numeric_config(old) != _gmm_numeric_config(precise)


def test_gmm_boundary_widens_only_after_original_float32_normalization_and_noise():
    values = np.asarray([[10000.0, 1.00001], [10000.0, 1.00002]], dtype=np.float32)
    mean = np.asarray([10000.0, 1.0], dtype=np.float32)
    scale = np.asarray([1.0, 0.00001], dtype=np.float32)
    noise = np.asarray([0.01, -0.02], dtype=np.float32)
    fit32 = _gmm_fit_values(values, mean, scale, "float32")
    fit64 = _gmm_fit_values(values, mean, scale, "float64")
    score32 = _gmm_score_values(values, mean, scale, noise, "float32")
    score64 = _gmm_score_values(values, mean, scale, noise, "float64")
    assert fit64.dtype == np.float64 and score64.dtype == np.float64
    np.testing.assert_array_equal(fit64, fit32.astype(np.float64))
    np.testing.assert_array_equal(score64, score32.astype(np.float64))
