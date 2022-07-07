""" ComparisonHC clustering algorithm (Ghoshdastidar, Perrot and Luxburg, 2019)

This implementation is an adaptation of Michael Perrot's library code
with his permission. https://github.com/mperrot/ComparisonHC


Copyright (c) 2022 Alexander Conzelmann, David-Elias Künstle, Michael Perrot

Licensed under the MIT license: https://opensource.org/licenses/MIT
Permission is granted to use, copy, modify, and redistribute the work.
Full license information available in the project LICENSE file.
"""
from typing import List
import sparse
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from cblearn import utils


def flatten(l: List[list]) -> list:
    # sum can essentially be used as a mapReduce / flatMap ;)
    return sum(l, [])


def clusters_from_dendrogram(dendrogram, clusters, k):
    """Cuts a dendrogram of the initial clusters to obtain a partition of
    the space in exactly k clusters.

    If k is higher than the number of initial clusters, the
    initial clusters are returned.

    Args:
        dendrogram: numpy array, shape (n_clusters-1, 3)
                     The dendrogram that should be used to obtain the partition.
        clusters: list of (list of examples)
                  The initial clusters of the dendrogram.
        k: int, The number of clusters in the partition.

    Returns:
        k_clusters: list of (list of examples)
                    The k_clusters that are merged last in the dendrogram.

    """
    n_clusters = len(clusters)
    if k >= n_clusters:
        return clusters

    if k < 2:
        return [[example_i for cluster in clusters for example_i in cluster]]

    k_clusters = [[example_i for example_i in cluster] for cluster in clusters]

    clusters_indices = list(range(n_clusters))
    dendrogram = dendrogram.astype(int)  # with floats, the index-search below fails
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


def closest_clusters(clusters, comparisons, n_objects: int):
    """Returns the indices of the two clusters that are closest to each
    other in the list.

    Given a list of clusters, this method is deterministic.

    Args: 
        clusters : A list containing at least two clusters.
        comparisons: Quadruplets in sparse format
        n_objects: number of examples in clusters/comparisons

    Returns:
        i: The index of the first of the two closest clusters.  
        j: The index of the second of the two closest clusters.

    """
    n_clusters = len(clusters)

    i, j = None, None

    score_best = -1

    # Prepare the normalization array
    # This is the divisor for each entry in the sum.
    # It depends on the cluster of each example.
    normalization = np.zeros((1, 1, n_objects, n_objects))
    for r in range(n_clusters):
        normalization[0, 0, [np.array(clusters[r]).reshape(-1, 1)],
                      np.isin(np.arange(n_objects), clusters[r], invert=True)] = 1/len(clusters[r])

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

    return i, j


class ComparisonHC(ClusterMixin, BaseEstimator):
    """ComparisonHC.

    ComparisonHC [1]_ is an hierarchical clustering algorithm that calculates
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
        cluster_: list of list
            Initial cluster information used for fitting.

    Examples:

    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.metrics import normalized_mutual_info_score
    >>> from cblearn.datasets import make_random_triplets
    >>> from cblearn.cluster import ComparisonHC
    >>> import numpy as np
    >>> means = np.array([[1,0], [-1, 0]])
    >>> stds = 0.2 * np.ones(means.shape)
    >>> xs, ys = make_blobs(n_samples=[10, 10], centers=means, cluster_std=stds, 
    ...                     n_features=2, random_state=2)
    >>> estimator = ComparisonHC(2)
    >>> t = make_random_triplets(xs, result_format="list-order", size=5000, random_state=2)
    >>> labels = estimator.fit_predict(t)
    >>> normalized_mutual_info_score(labels, ys)
    1.0

    References
    ----------
    .. [1] Ghoshdastidar, D., Perrot, M., von Luxburg, U. (2019). 
           Foundations of Comparison-Based Hierarchical Clustering.
           Advances in Neural Information Processing Systems 32.
    """
    def __init__(self, n_clusters: int) -> None:
        """
        Initialize the estimator.

        Args:
            n_clusters: Number of clusters desired in the final clustering.
        """
        self.n_clusters = n_clusters

    def _triplets_to_quadruplets(self, X, y=None):
        """ ComparisonHC expects the sparse quadruplet representation,
        where to every entry (1) the inverse (-1), too.

        E.g. if quad[0, 5, 4, 6] == 1, then quad[4, 6, 0, 5] == -1
        """
        triplets = utils.check_query_response(X, y, result_format='tensor-count')
        triplets = triplets.clip(-1, 1)  # remove repeated triplets
        quads = sparse.COO(triplets.coords[[0, 1, 0, 2], :], triplets.data, (len(triplets),) * 4)
        quads = quads + sparse.COO(triplets.coords[[0, 2, 0, 1], :], -1 * triplets.data, (len(triplets),) * 4)
        return quads

    def _fit_dendrogram(self, init_clusters, quadruplets, n_objects):
        n_clusters = len(init_clusters)  # != self.n_clusters, which is number of predicted clusters
        clusters_indices = list(range(n_clusters))
        clusters_copy = [[example for example in cluster] for cluster in init_clusters]

        dendrogram = np.zeros((n_clusters - 1, 4))
        for it in range(n_clusters - 1):
            i, j = closest_clusters(clusters_copy, quadruplets, n_objects)

            if i > j:
                i, j = j, i
            clusters_copy[i].extend(clusters_copy[j])
            del clusters_copy[j]

            dendrogram[it, 0] = clusters_indices[i]
            dendrogram[it, 1] = clusters_indices[j]
            dendrogram[it, 2] = len(clusters_copy[i])

            clusters_indices[i] = n_clusters + it
            del clusters_indices[j]
        return dendrogram

    def fit(self, X, y=None, init_clusters=None):
        """Computes the dendrogram of a list of clusters.

        Args:
            X: Triplets, repeated responses will be ignored (majority vote)
            y: optional responses
            init_clusters: list of (list of examples), len(n_clusters)
                An optional list containing the initial clusters (list of examples).

        Returns:
            self: object

        Raises:
            ValueError: If the initial partition has less that n_examples.

        """
        quads = self._triplets_to_quadruplets(X, y)
        n_objects = len(quads)
    
        if init_clusters is None:
            init_clusters = [[i] for i in range(n_objects)]
        self.clusters_ = init_clusters
        self.dendrogram_ = self._fit_dendrogram(init_clusters, quads, n_objects)

        clusters = clusters_from_dendrogram(
            self.dendrogram_, self.clusters_, self.n_clusters)
        labels_in_order = flatten([[i] * len(cluster)
                                   for i, cluster in enumerate(clusters)])
        labels_for_original = [-1] * len(labels_in_order)
        for lab, pos in zip(labels_in_order, flatten(clusters)):
            labels_for_original[pos] = lab
        self.labels_ = np.array(labels_for_original)
        return self
