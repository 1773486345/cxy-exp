from types import SimpleNamespace

import numpy as np
from scipy.stats import multivariate_normal

from scripts.run_mnrf_conditional_score_diagnostic import (
    BRANCH_CONTEXT, BRANCH_TARGET, conditional_mixture_log_density,
)


def _model(weights, means, covariances):
    return SimpleNamespace(covariance_type="full", weights_=np.asarray(weights, dtype=np.float64),
                           means_=np.asarray(means, dtype=np.float64),
                           covariances_=np.asarray(covariances, dtype=np.float64))


def test_conditional_density_matches_direct_gaussian_ratio_on_small_case():
    mean = np.asarray([1.0, -2.0, 0.5])
    covariance = np.asarray([[4.0, 1.2, 0.3], [1.2, 3.0, 0.4], [0.3, 0.4, 2.0]])
    samples = np.asarray([[0.5, -1.0, 1.5], [2.0, 0.2, -0.5]])
    model = _model([1.0], [mean], [covariance])
    actual = conditional_mixture_log_density(model, samples, [0, 2], [1])
    expected = []
    for sample in samples:
        expected.append(
            multivariate_normal.logpdf(sample, mean=mean, cov=covariance)
            - multivariate_normal.logpdf(sample[[1]], mean=mean[[1]],
                                         cov=covariance[np.ix_([1], [1])])
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_conditional_density_uses_posterior_component_weights():
    means = np.asarray([[-1.0, 0.0], [2.0, 1.0]])
    covariances = np.asarray([[[1.0, 0.5], [0.5, 2.0]],
                              [[3.0, -0.4], [-0.4, 1.0]]])
    weights = np.asarray([0.8, 0.2])
    samples = np.asarray([[0.2, 1.2], [2.5, -0.5]])
    model = _model(weights, means, covariances)
    actual = conditional_mixture_log_density(model, samples, [0], [1])
    expected = []
    for sample in samples:
        joint_logs = np.asarray([
            np.log(weight) + multivariate_normal.logpdf(sample, mean=mean, cov=cov)
            for weight, mean, cov in zip(weights, means, covariances)
        ])
        condition_logs = np.asarray([
            np.log(weight) + multivariate_normal.logpdf(sample[[1]], mean=mean[[1]],
                                                        cov=cov[np.ix_([1], [1])])
            for weight, mean, cov in zip(weights, means, covariances)
        ])
        expected.append(np.logaddexp.reduce(joint_logs) - np.logaddexp.reduce(condition_logs))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_full_view_partition_and_extreme_finite_inputs():
    assert BRANCH_TARGET.tolist() == [0] + list(range(9, 17))
    assert BRANCH_CONTEXT.tolist() == list(range(1, 9)) + list(range(17, 25))
    mean = np.zeros(25)
    covariance = np.eye(25)
    covariance[0, 1] = covariance[1, 0] = 0.2
    model = _model([1.0], [mean], [covariance])
    values = np.zeros((2, 25), dtype=np.float64)
    values[0] = 1e100
    values[1, 0] = -1e100
    score = conditional_mixture_log_density(model, values, BRANCH_TARGET, BRANCH_CONTEXT)
    assert np.isfinite(score).all()
    try:
        conditional_mixture_log_density(model, np.zeros((1, 24)), BRANCH_TARGET, BRANCH_CONTEXT)
    except ValueError as error:
        assert "wrong width" in str(error)
    else:
        raise AssertionError("wrong feature width must be rejected")
