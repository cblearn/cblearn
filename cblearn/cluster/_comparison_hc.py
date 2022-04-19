from typing import Optional, Union, List, Tuple
from cblearn import utils
import time
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from abc import ABCMeta, abstractmethod
import time

import time
import itertools

# ComparisonHC Version given in the original module by the author
# in https://github.com/mperrot/ComparisonHC


class _ComparisonHC:
    """ComparisonHC

    Parameters
    ----------
    linkage : Linkage object
        The linkage used to determine the merging order of the
        clusters.

    Attributes
    ----------
    linkage : Linkage object
        The linkage used to determine the merging order of the
        clusters.

    clusters : list of (list of examples), len (n_clusters)
        A list containing the initial clusters (list of
        examples). Initialized to the empy list until the fit method is
        called.

    n_clusters : int
        The number of initial clusters. Initialized to 0 until the fit
        method is called.

    n_examples : int
        The number of examples.

    dendrogram : numpy array, shape (n_clusters-1, 3)
        An array corresponding to the learned dendrogram. After
        iteration i, dendrogram[i,0] and dendrogram[i,1] are the
        indices of the merged clusters, and dendrogram[i,2] is the
        size of the new cluster. The dendrogram is initialized to None
        until the fit method is called.

    time_elapsed : float
        The time taken to learn the dendrogram. It includes the time
        taken by the linkage to select the next clusters to merge. It
        only records the time elapsed during the last call to fit.

    Notes
    -----
    The linkage object should exhibit a closest_clusters(clusters)
    method that takes a list of clusters (that is a list of (list of
    examples)) and returns the indices of the two closest clusters
    that should be merged next. This method should be deterministic,
    that is repeated calls to closest_clusters with the same
    parameters should yield the same result.

    """

    def __init__(self, linkage):
        self.linkage = linkage

        self.clusters = []

        self.n_clusters = 0

        self.n_examples = self.linkage.n_examples

        self.dendrogram = None

        self.time_elapsed = 0

    def fit(self, clusters):
        """Computes the dendrogram of a list of clusters.

        Parameters
        ----------
        clusters : list of (list of examples), len (n_clusters)
            A list containing the initial clusters (list of examples).

        Returns
        -------
        self : object

        Raises
        ------
        ValueError
            If the initial partition has less that n_examples.

        """
        n_examples = sum([len(cluster) for cluster in clusters])
        if n_examples != self.n_examples:
            raise ValueError("The initial partition should have exactly n_examples.")

        time_start = time.process_time()

        self.clusters = clusters

        self.n_clusters = len(clusters)

        self.dendrogram = np.zeros((self.n_clusters-1, 4))

        clusters_indices = list(range(self.n_clusters))

        clusters_copy = [[example for example in cluster] for cluster in self.clusters]

        for it in range(self.n_clusters-1):
            i, j = self.linkage.closest_clusters(clusters_copy)

            if i > j:
                i, j = j, i
            clusters_copy[i].extend(clusters_copy[j])
            del clusters_copy[j]

            self.dendrogram[it, 0] = clusters_indices[i]
            self.dendrogram[it, 1] = clusters_indices[j]
            self.dendrogram[it, 2] = len(clusters_copy[i])

            clusters_indices[i] = self.n_clusters+it
            del clusters_indices[j]

        time_end = time.process_time()
        self.time_elapsed = (time_end-time_start)
        return self

    def _get_k_clusters(self, dendrogram, clusters, k):
        """Cuts a dendrogram of the initial clusters to obtain a partition of
        the space in exactly k clusters.

        If k is higher than the number of initial clusters, the
        initial clusters are returned.

        Parameters
        ----------
        dendrogram : numpy array, shape (n_clusters-1, 3)
            The dendrogram that should be used to obtain the
            partition.

        clusters : list of (list of examples)
            The initial clusters of the dendrogram.

        k : int
            The number of clusters in the partition.

        Returns
        -------
        k_clusters : list of (list of examples)
            The k_clusters that are merged last in the dendrogram.

        """
        n_clusters = len(clusters)
        if k >= n_clusters:
            return clusters

        if k < 2:
            return [[example_i for cluster in clusters for example_i in cluster]]

        k_clusters = [[example_i for example_i in cluster] for cluster in clusters]

        clusters_indices = list(range(n_clusters))

        for it in range(n_clusters-k):
            i = clusters_indices.index(dendrogram[it, 0])
            j = clusters_indices.index(dendrogram[it, 1])

            if i > j:
                i, j = j, i

            k_clusters[i].extend(k_clusters[j])
            del k_clusters[j]

            clusters_indices[i] = n_clusters+it
            del clusters_indices[j]

        return k_clusters


class OrdinalLinkage(metaclass=ABCMeta):
    """An abstract ordinal linkage that controls the merging order in
    hierarchical clustering.

    Parameters
    ----------
    oracle : Oracle object
        An oracle used to query the quadruplets.

    Attributes
    ----------
    oracle : Oracle object
        The oracle used to query the quadruplets.

    n_examples : int
        The number of examples handled by the linkage.

    time_elapsed : float
        The total time taken by the linkage to determine the closest
        clusters in a list of clusters. It includes the time taken by
        the oracle to return the quadruplets.


    Raises
    ------
    ValueError
        If the oracle is not compatible with the linkage.

    """

    def __init__(self, oracle):
        self.oracle = oracle

        self.n_examples = self.oracle.n_examples

        self.time_elapsed = 0

    @abstractmethod
    def closest_clusters(self, clusters):
        """Returns the indices of the two clusters that are closest to each
        other in the list.

        Given a list of clusters, this method should be deterministic.

        Parameters
        ----------
        clusters : list of (list of examples)
            A list containing the clusters (list of examples).

        Returns
        -------
        i : int
            The index of the first of the two closest clusters.

        j : int
            The index of the second of the two closest clusters.

        """
        pass


class OrdinalLinkageAverage(OrdinalLinkage):
    """An ordinal linkage that controls the merging order in hierarchical
    clustering assuming that the oracle returns quadruplets.

    This method directly use the quadruplets in an average linkage
    scheme.

    Parameters
    ----------
    oracle : Oracle object
        An oracle used to query the quadruplets.

    Attributes
    ----------
    oracle : Oracle object
        The oracle used to query the quadruplets.

    n_examples : int
        The number of examples handled by the linkage.

    time_elapsed : float
        The total time taken by the linkage to determine the closest
        clusters in a list of clusters. It includes the time taken by
        the oracle to return the quadruplets.

    Raises
    ------
    ValueError
        If the oracle is not compatible with the linkage, that is it
        does not exhibit a method comparisons() and an attribute
        n_examples.

    Notes
    -----
    To be compatible with this linkage the oracle should exhibit a
    method comparisons() that returns a numpy array of shape
    (n_examples, n_examples, n_examples, n_examples) containing values in
    {1,-1,0}. In entry (i,j,k,l), the value 1 indicates that the
    quadruplet (i,j,k,l) is available, the value -1 indicates that the
    quadruplet (k,l,i,j) is available, and the value 0 indicates that
    neither of the quadruplets is available. This method should be
    deterministic. This numpy array is not modified by this class to
    ensure that it can be passed by reference.

    The oracle should also exhibit an attribute n_examples counting the
    number of examples it handles.

    """

    def __init__(self, oracle):
        if not (hasattr(oracle, "n_examples") and hasattr(oracle, "comparisons")
                and callable(getattr(oracle, "comparisons"))):
            raise ValueError("Incompatible oracle, callable 'comparisons' or attribute 'n_examples' missing.")

        super(OrdinalLinkageAverage, self).__init__(oracle)

    def closest_clusters(self, clusters):
        """Returns the indices of the two clusters that are closest to each
        other in the list.

        Given a list of clusters, this method is deterministic.

        Parameters
        ----------
        clusters : list of (list of examples)
            A list containing at least two clusters.

        Returns
        -------
        i : int
            The index of the first of the two closest clusters.

        j : int
            The index of the second of the two closest clusters.

        """
        time_start = time.process_time()

        n_clusters = len(clusters)

        comparisons = self.oracle.comparisons()

        i, j = None, None

        score_best = -1

        # Prepare the normalization array
        # This is the divisor for each entry in the sum.
        # It depends on the cluster of each example.
        normalization = np.zeros((1, 1, self.n_examples, self.n_examples))
        for r in range(n_clusters):
            normalization[0, 0, [np.array(clusters[r]).reshape(-1, 1)],
                          np.isin(np.arange(self.n_examples), clusters[r], invert=True)] = 1/len(clusters[r])

        for s in range(n_clusters):
            normalization[0, 0, :, clusters[s]] /= len(clusters[s])

        for p in range(n_clusters):
            clusters_p = clusters[p]

            comparisons_p = comparisons[clusters_p, :, :, :]

            n_examples_p = len(clusters_p)

            for q in range(p+1, n_clusters):
                score = 0

                clusters_q = clusters[q]

                comparisons_pq = comparisons_p[:, clusters_q, :, :]

                n_examples_pq = n_examples_p*len(clusters_q)

                # Divide each entry in the matrix by the normalization and sum everything
                score = np.sum(comparisons_pq*normalization)/(n_clusters*(n_clusters-1)*n_examples_pq)

                if score > score_best:
                    i, j = p, q
                    score_best = score

        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)

        return i, j


class Oracle(metaclass=ABCMeta):
    """An abstract oracle that returns quadruplets.

    Parameters
    ----------
    n_examples : int
        The number of examples handled by the oracle.

    seed : int or None
        The seed used to initialize the random number generators. If
        None the current time is used, that is
        int(time.time()). (Default: None).

    Attributes
    ----------
    n_examples : int
        The number of examples handled by the oracle.

    seed : int
        The seed used to initialize the random number generators.

    """

    def __init__(self, n_examples, seed=None):
        self.n_examples = n_examples

        if seed is not None:
            self.seed = seed
        else:
            self.seed = int(time.time())

    @abstractmethod
    def comparisons(self):
        """Returns all the quadruplets associated with the examples.

        Returns
        -------
        comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
            A reference to a numpy array of shape (n_examples,
            n_examples, n_examples, n_examples) containing values in
            {1,-1,0}. In entry (i,j,k,l), the value 1 indicates that
            the quadruplet (i,j,k,l) is available, the value -1
            indicates that the quadruplet (k,l,i,j) is available, and
            the value 0 indicates that neither of the quadruplets is
            available. This method should be deterministic.

        """
        pass

    @abstractmethod
    def comparisons_to_ref(self, k, l):
        """Returns all the quadruplets with respect to the reference of
        examples k,l.

        Returns
        -------
        comparisons_array : numpy array, shape (n_examples, n_examples)
            A reference to a numpy array of shape (n_examples,
            n_examples) containing values in {1,-1,0}. In entry (i,j),
            the value 1 indicates that the quadruplet (i,j,k,l) is
            available, the value -1 indicates that the quadruplet
            (k,l,i,j) is available, and the value 0 indicates that
            neither of the quadruplets is available. This method
            should be deterministic.

        """
        pass

    @abstractmethod
    def comparisons_single(self, i, j, k, l):
        """Returns the quadruplet associated with the examples i,j,k,l.

        Returns
        -------
        comparisons_array : int8
            A int8 in {1,-1,0}. The value 1 indicates that the
            quadruplet (i,j,k,l) is available, the value -1 indicates
            that the quadruplet (k,l,i,j) is available, and the value
            0 indicates that neither of the quadruplets is
            available. This method should be deterministic.

        """
        pass


class OracleComparisons(Oracle):
    """An oracle that returns quadruplets from a precomputed numpy array.

    Parameters
    ----------
    comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
        A numpy array of shape (n_examples, n_examples, n_examples,
        n_examples) containing values in {1,-1,0}. In entry (i,j,k,l),
        the value 1 indicates that the quadruplet (i,j,k,l) is
        available, the value -1 indicates that the quadruplet
        (k,l,i,j) is available, and the value 0 indicates that neither
        of the quadruplets is available.

    Attributes
    ----------
    n_examples : int
        The number of examples.

    comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
        A numpy array of shape (n_examples, n_examples, n_examples,
        n_examples) containing values in {1,-1,0}. In entry (i,j,k,l),
        the value 1 indicates that the quadruplet (i,j,k,l) is
        available, the value -1 indicates that the quadruplet
        (k,l,i,j) is available, and the value 0 indicates that neither
        of the quadruplets is available.

    """

    def __init__(self, comparisons_array):
        self.comparisons_array = comparisons_array

        n_examples = comparisons_array.shape[0]
        super(OracleComparisons, self).__init__(n_examples)

    def comparisons(self):
        return self.comparisons_array

    def comparisons_to_ref(self, k, l):
        return self.comparisons_array[:, :, k, l]

    def comparisons_single(self, i, j, k, l):
        return self.comparisons_array[i, j, k, l]


def triplets_to_quadruplets(triplets: np.ndarray, responses: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Transforms an array of triplets (with responses) to an array of quadruplets.

    Assumes triplets, responses to be in list-boolean form, e.g.

    responses[i] is True if triplets[i][0] is closer to triplets[i][1] than to
    triplets[i][2].

    If responses is None, we assume that all responses are true (e.g. it is always triplets[i][1] closer).

    We return a quadruplet matrix that is filled according to the following scheme:
    If the triplet array allows for a statement (a,b,c) in triplet form then we
    set quadruplet[a,b,a,c] = 1.

    Triplets may contain duplicates or conflicting entries.
    In this case, we replace the value with a majority vote.
    """
    # error checking
    if len(triplets.shape) != 2:
        raise ValueError("Triplets must be a 2D array")
    if triplets.shape[1] != 3:
        raise ValueError("Triplets must have 3 columns")
    num_triplets = triplets.shape[0]
    if responses is None:
        responses = np.ones(num_triplets).astype(bool)
    if len(responses.shape) != 1:
        raise ValueError("Responses must be a 1D array or None")
    n = np.max(triplets) + 1
    q = np.zeros((n, n, n, n))

    for i in range(num_triplets):
        t = triplets[i]
        r = responses[i]
        if r:
            a, b, c = t[0], t[1], t[2]
        else:
            a, b, c = t[0], t[2], t[1]

        if q[a, b, a, c] != 0 or q[a, c, a, b] != 0:
            raise ValueError(
                f"Unreduced triplets found (or responses): {t, r, i}")
        q[a, b, a, c] = 1
        q[a, c, a, b] = -1
    return q


def unify_triplet_order(triplets: np.ndarray, responses: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Takes in an array of triplets and responses, and reorders the triplets such that it always
    holds that a triplet has the meaning

    triplet[0] is closer to triplet[1] than to triplet[2]
    """
    if responses is None:
        return triplets
    if triplets.shape[0] != responses.shape[0]:
        raise ValueError(
            f"Triplets {triplets.shape}/responses {responses.shape} have wrong format: Should agree on first dimension.")
    if len(triplets.shape) != 2:
        raise ValueError(
            f"Triplets {triplets.shape} have wrong format: Should be 2-dimensional matrix.")
    if len(responses.shape) != 1:
        raise ValueError(
            f"Responses {responses.shape} have wrong format: Should be 1-dimensional array.")
    triplets = triplets.copy()
    wrong_order = np.logical_not(responses)
    # swap those in wrong order
    triplets[wrong_order, 1], triplets[wrong_order,
                                       2] = triplets[wrong_order, 2], triplets[wrong_order, 1]
    return triplets


def reduce_triplets(triplets: np.ndarray, responses: Optional[np.ndarray] = None) -> np.ndarray:
    triplets = unify_triplet_order(triplets, responses)
    reduced_triplets = []
    for t in triplets:
        a, b, c = t
        closer_to_b = np.all(triplets == np.array([a, b, c]), axis=1)
        closer_to_c = np.all(triplets == np.array([a, c, b]), axis=1)
        if closer_to_b.sum() > closer_to_c.sum():
            reduced_triplets.append([a, b, c])
        else:
            reduced_triplets.append([a, c, b])
    return np.unique(reduced_triplets, axis=0)


def flatten(l: List[list]) -> list:
    # sum can essentially be used as a mapReduce / flatMap ;)
    return sum(l, [])


class ComparisonHC():
    """ ComparisonHC

    ComparisonHC[1] is an hierarchical clustering algorithm that calculates
    clusters on triplet data without computing an intermediate embedding.
    This is done via an adapted linkage algorithm that only uses the triplet
    information.

    As this is algorithm produces its clusterings via a Dendrogram that
    is created on the whole dataset, we do not provide a fit method. Call
    fit_predict directly with the complete dataset you want to do an
    clustering on.

    Keep in mind that this algorithm was optimized and developed for hierarchical
    clustering, and simply adapted to produce a flat clustering with
    the desired number of clusters. Thus, this algorithm might not have
    optimal performance in these settings when compared to other approaches.

    Attributes:
        dendrogram_: numpy array, shape (n_clusters-1, 4)
            An array corresponding to the learned dendrogram. After
            iteration i, dendrogram[i,0] and dendrogram[i,1] are the
            indices of the merged clusters, and dendrogram[i,2] is the
            size of the new cluster. The dendrogram is initialized to None
            until the fit method is called.
            The last column is set to 0 (implemented like this by the original algorithm).


    Examples:
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.metrics import normalized_mutual_info_score
    >>> from cblearn.datasets import make_random_triplets
    >>> from cblearn.cluster import ComparisonHC
    >>> import numpy as np
    >>> means = np.array([[1,0], [-1, 0]])
    >>> stds = 0.2 * np.ones(means.shape)
    >>> xs, ys = make_blobs(n_samples=[10, 10], centers=means, cluster_std=stds, n_features=2, random_state=2)
    >>> estimator = ComparisonHC(2)
    >>> t, r = make_random_triplets(xs, result_format="list-boolean", size=5000, random_state=2)
    >>> y_pred = estimator.fit_predict(t, r)
    >>> print(f"NMI of ComparisonHC: {normalized_mutual_info_score(y_pred, ys)}")

    References
    ----------
    .. [1] Ghoshdastidar, D., Perrot, M., von Luxburg, U. (2019). Foundations of Comparison-Based Hierarchical Clustering.
            Advances in Neural Information Processing Systems 32.
    """

    def __init__(self, num_clusters: int) -> None:
        """
        Initialize the estimator.

        Parameters:
            num_clusters: Number of clusters desired in the final clustering.
        """
        self.num_clusters = num_clusters

    def score(self, X: Union[np.ndarray, Tuple[np.ndarray,  np.ndarray]], y: np.ndarray) -> float:
        """
        Returns the normalized mutual information score between
        the clustering predicted by the clustering algorithm on X,
        when the true labels are given by Y.
        Args:
            X: The input query of any form. Might be a tuple, if triplets and responses are necessary.
            y: The ground truth labels to compre against.
        """
        if isinstance(X, tuple):
            return normalized_mutual_info_score(self.fit_predict(*X), y)
        else:
            return normalized_mutual_info_score(self.fit_predict(X), y)

    def fit_predict(self, X: utils.Query, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes a clustering of the input using ComparisonHC.

        Args:
            X: The input query of any form. For example an ndarray of size(input_samples, 3).
               will be converted internally to the desired type.
            y: Responses to the input query, if necessary.
        Returns:
            ndarray of shape(input_samples), which contains a label for each datapoint.
        """
        # fit
        triplets = utils.check_query_response(X, y, result_format='list-order')
        assert isinstance(triplets, np.ndarray), f"check_query_response returned unexpected type: {type(triplets)}"
        triplets = reduce_triplets(triplets)
        quads = triplets_to_quadruplets(triplets)
        n = quads.shape[0]
        assert quads.shape == (n, n, n, n)
        oracle = OracleComparisons(quads)
        linkage = OrdinalLinkageAverage(oracle)
        chc = _ComparisonHC(linkage)
        chc.fit([[i] for i in range(n)])
        self.dendrogram_ = chc.dendrogram

        # predict
        clusters = chc._get_k_clusters(
            chc.dendrogram, chc.clusters, self.num_clusters)
        labels_in_order = flatten([[i] * len(cluster)
                                   for i, cluster in enumerate(clusters)])
        labels_for_original = [-1] * len(labels_in_order)
        for lab, pos in zip(labels_in_order, flatten(clusters)):
            labels_for_original[pos] = lab
        assert -1 not in labels_for_original
        return np.array(labels_for_original)
