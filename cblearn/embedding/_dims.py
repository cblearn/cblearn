""" This file contains dimension estimation utilities. """
from dataclasses import dataclass
from sklearn.model_selection import validation_curve
from sklearn.model_selection import RepeatedKFold
from scipy import stats
import numpy as np

from cblearn import utils


@dataclass
class DimensionEstimationResult:
    estimated_dimension: int
    dimensions: np.ndarray
    train_scores: np.ndarray
    test_scores: np.ndarray
    stats_result: dict

    def plot_scores(self, train_kwargs={}, test_kwargs={}):
        import matplotlib.pyplot as plt

        plot_validation_curve(self.dimensions, self.train_scores, self.test_scores,
                              train_kwargs, test_kwargs)
        plt.axvline(self.estimated_dimension, color='k', linestyle='--', label="Estimated Dimension")
        plt.legend(loc="best")


def plot_validation_curve(x, train_scores, test_scores, train_kwargs={}, test_kwargs={}):
    import matplotlib.pyplot as plt

    _train_kwargs = {"lw": 2, "color": "C0"}
    _train_kwargs.update(train_kwargs)
    _test_kwargs = {"lw": 2, "color": "C1"}
    _test_kwargs.update(test_kwargs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(
        x, train_scores_mean, label="Training", **_train_kwargs
    )
    plt.fill_between(
        x,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        **_train_kwargs
    )
    plt.plot(
        x, test_scores_mean, label="Validation", **_test_kwargs
    )
    plt.fill_between(
        x,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        **_test_kwargs
    )


def _holm_correction(pvals, alpha=0.05, is_sorted=False, returnsorted=False):
    """ Holm-Bonferroni method for multiple hypothesis testing correction.
        This code was adapted from the statsmodels library, which is licensed under BSD-3.
    """
    pvals = np.asarray(pvals)
    alphaf = alpha  # Notation ?

    if not is_sorted:
        sortind = np.argsort(pvals)
        pvals = np.take(pvals, sortind)

    ntests = len(pvals)
    alphacSidak = 1 - np.power((1. - alphaf), 1./ntests)
    alphacBonf = alphaf / float(ntests)
    notreject = pvals > alphaf / np.arange(ntests, 0, -1)
    nr_index = np.nonzero(notreject)[0]
    if nr_index.size == 0: # nonreject is empty, all rejected
        notrejectmin = len(pvals)
    else:
        notrejectmin = np.min(nr_index)
    notreject[notrejectmin:] = True
    reject = ~notreject
    pvals_corrected_raw = pvals * np.arange(ntests, 0, -1)
    pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
    del pvals_corrected_raw

    if pvals_corrected is not None:  #not necessary anymore
        pvals_corrected[pvals_corrected > 1] = 1
    if is_sorted or returnsorted:
        return reject, pvals_corrected, alphacSidak, alphacBonf
    else:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[sortind] = reject
        return reject_, pvals_corrected_, alphacSidak, alphacBonf


def _sequential_crossval_ttest(test_scores_cv, n_splits, alpha):
    """
    sample_sequence: Array of shape (n_samples, n_steps)
    """
    differences = np.diff(test_scores_cv)
    n_samples, n_steps = differences.shape

    test_train_ratio = 1 / (n_splits - 1)
    effect = differences.mean(axis=0) / differences.std(axis=0, ddof=1)
    # Nadeau and Bengio correction of dependent Student's t-test due to Cross Validation
    t_stats = effect / np.sqrt(1 / n_samples + test_train_ratio)

    p_values = stats.t.sf(t_stats, n_samples - 1)  # right-tailed t-test
    assert t_stats.shape == (n_steps,)

    # holm-bonferroni correction
    result = _holm_correction(p_values, alpha=alpha)
    return {'reject': result[0], 'pvals': p_values, 'pvals_corrected': result[1],
            'tstats': t_stats, 'effectsize': effect,
            'alpha': alpha, 'alpha_corrected': result[3], 'step': np.arange(n_steps)}


def estimate_dimensionality_cv(estimator, queries, responses=None,
                               test_dimensions: list = [1, 2, 3], n_splits=10, n_repeats=1,
                               refit=True, alpha=0.05, param_name="n_components", n_jobs=-1, random_state=None):
    """ Estimates the dimensionality of the embedding space through cross-validation
        that has the best fit for the provided data [1]_.

        Attributes:
          estimator: The embedding estimator to use.
          queries: The triplet queries to embed.
          responses: Optional responses, if not encoded in triplets.
          test_dimensions: The dimensions to test as a monotonic increasing list.
          n_splits: The number of splits to use for cross-validation.
          n_repeats: The number of repeatitions of each cross-validation split.
                     Use 1 for fast results, but 10 or more for more reliable results.
          refit: if true, then fit the estimator on the entire dataset using the best dimensionality.
          alpha: The significance level for the hypothesis test.
          param_name: The name of the estimator parameter that describes the embedding dimensionality.
          n_jobs: The number of parallel jobs to use for cross-validation.
          random_state: The random state or seed to use for CV splits.

        Returns:
          result: A result object with the estimated dimension and other information.

        Examples:

        >>> from cblearn.embedding import estimate_dimensionality_cv
        >>> from cblearn.embedding import SOE
        >>> from cblearn.datasets import make_random_triplets
        >>> rs = np.random.RandomState(42)
        >>> true_embedding = rs.rand(15, 2)  # 15 points in 2D
        >>> triplets = make_random_triplets(true_embedding, result_format='list-order', size=1000, random_state=rs)
        >>> estimator = SOE(n_components=1)
        >>> dim_result = estimate_dimensionality_cv(estimator, triplets, test_dimensions=[1, 2, 3], n_splits=5, refit=True)
        >>> dim_result.estimated_dimension
        2
        >>> true_embedding.shape == estimator.embedding_.shape
        True
        >>> dim_result.plot_scores()


        References
        ----------
        .. [1]  Künstle, D.-E., von Luxburg, U., & Wichmann, F. A. (2022).
                Estimating the perceived dimension of psychophysical stimuli
                using triplet accuracy and hypothesis testing.
                Journal of Vision, 22(13), 5. https://doi.org/10.1167/jov.22.13.5
    """
    if np.diff(test_dimensions).min() < 1:
        raise ValueError("test_dimensions must be monotonically increasing")

    queries = utils.check_query_response(queries, responses, result_format='list-order')
    cv_folds = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=random_state)
    train_scores, test_scores = validation_curve(
        estimator, queries, y=None, cv=cv_folds, n_jobs=n_jobs,
        param_name=param_name, param_range=test_dimensions)

    stat_results = _sequential_crossval_ttest(test_scores.T, n_splits, alpha=alpha)
    for ix, reject in enumerate(stat_results['reject']):
        if not reject:
            estimated_dimension = test_dimensions[ix]
            break
    else:
        estimated_dimension = test_dimensions[-1]

    if refit:
        estimator.set_params(**{param_name: estimated_dimension})
        estimator.fit(queries, y=None)
    return DimensionEstimationResult(estimated_dimension, test_dimensions, train_scores, test_scores, stat_results)