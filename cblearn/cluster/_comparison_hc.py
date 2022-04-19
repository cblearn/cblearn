from typing import Optional, Union
from cblearn import utils
import time
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from abc import ABCMeta, abstractmethod
import time
import itertools
import random

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

    def cost_dasgupta(self, similarities):
        """Computes the cost of the dendrogram as proposed by Dasgupta in 'A
        cost function for similarity-based hierarchical
        clustering'. The lower the cost the better the dendrogram is.

        This cost is based on the idea that similar examples should be
        merged earlier in the dendrogram.

        Parameters
        ----------
        similarities : numpy array, shape (n_examples, n_examples)
            A numpy array containing the similarities between all the
            examples.

        Returns
        -------
        cost : float
            The cost of the last dendrogram learned by the fit
            method. Lower is better.

        Raises
        ------
        RuntimeError
            If the dendrogram has not been lerned yet.

        """
        if self.dendrogram is None:
            raise RuntimeError("No dendrogram, the fit method should be called first.")

        cost = 0
        for cluster in self.clusters:
            for (example_i, example_j) in itertools.combinations(cluster, 2):
                cost += len(cluster)*similarities[example_i, example_j]

        for ((cluster_index_i, cluster_i), (cluster_index_j, cluster_j)) in itertools.combinations(enumerate(self.clusters), 2):
            cluster_size = self.dendrogram[self._get_iteration(cluster_index_i, cluster_index_j), 2]

            for example_i in cluster_i:
                for example_j in cluster_j:
                    cost += cluster_size*similarities[example_i, example_j]

        return cost

    def _get_iteration(self, cluster_index_i, cluster_index_j):
        """Returns the iteration at which two clusters are merged.

        The indices cluster_index_i and cluster_index_j refer to
        cluster in the list of initial clusters.

        Parameters
        ----------
        cluster_index_i : int
            The index in clusters of the first cluster.

        cluster_index_j : int
            The index in clusters of the second cluster.

        Returns
        -------
        it : int
            The iteration at which the two clusters are merged. None
            if no such iteration is found.

        Raises
        ------
        RuntimeError
            If the dendrogram has not been lerned yet.

        """
        if self.dendrogram is None:
            raise RuntimeError("No dendrogram, the fit method should be called first.")

        for it in range(self.n_clusters-1):
            if self.dendrogram[it, 0] == cluster_index_i or self.dendrogram[it, 1] == cluster_index_i:
                cluster_index_i = self.n_clusters+it
            if self.dendrogram[it, 0] == cluster_index_j or self.dendrogram[it, 1] == cluster_index_j:
                cluster_index_j = self.n_clusters+it
            if cluster_index_i == cluster_index_j:
                return it
        else:
            return None

    def average_ARI(self, max_level, dendrogram_truth, clusters_truth=None):
        """Computes the score of the learned dendrogram in terms of Average
        Adjusted Rand Index as described in the main paper and
        compared to the ground truth dendrogram. The higher the score
        the better the dendrogram is.

        This score assumes that the learned hierarchy have levels
        which correspond to cuts in the dendrograms with given numbers
        of clusters. Here, we consider power of 2 levels, that is
        partiotions of the space in 2 clusters, 4 clusters, 8
        clusters, ... 2**max_level clusters.

        Parameters
        ----------
        max_level : int
            The number of levels to consider.

        dendrogram_truth : numpy array, shape (n_clusters-1, 3)
            The true dendrogram for the data.

        clusters_truth : list of (list of examples)
            The initial clusters used to generate dendrogram_truth. If
            None, the same initial clusters than fit are
            used. (Default: None).

        Returns
        -------
        score : float
            The score of the last dendrogram learned by the fit
            method. Higher is better.

        Raises
        ------
        RuntimeError
            If the dendrogram has not been lerned yet.

        """
        if self.dendrogram is None:
            raise RuntimeError("No dendrogram, the fit method should be called first.")

        if clusters_truth is None:
            clusters_truth = self.clusters

        score = 0
        for level in range(1, max_level+1):
            k_clusters_truth = self._get_k_clusters(dendrogram_truth, clusters_truth, 2**level)
            k_clusters = self._get_k_clusters(self.dendrogram, self.clusters, 2**level)

            k_clusters_truth_labels = np.zeros((self.n_examples,))
            for cluster_index, cluster in enumerate(k_clusters_truth):
                k_clusters_truth_labels[np.array(cluster, dtype=int)] = cluster_index

            k_clusters_labels = np.zeros((self.n_examples,))
            for cluster_index, cluster in enumerate(k_clusters):
                k_clusters_labels[np.array(cluster, dtype=int)] = cluster_index

            score += adjusted_rand_score(k_clusters_truth_labels, k_clusters_labels)

        return score/max_level

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


class OrdinalLinkageKernel(OrdinalLinkage):
    """An ordinal linkage that controls the merging order in hierarchical
    clustering assuming that the oracle returns quadruplets.

    This method first computes kernel similarities between the examples
    before using an average linkage scheme.

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

    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the
        examples. Initialized to None until the first call to
        closest_clusters.

    time_elapsed : float
        The total time taken by the linkage to determine the closest
        clusters in a list of clusters. It includes the time taken by
        the oracle to return the quadruplets.

    Raises
    ------
    ValueError
        If the oracle is not compatible with the linkage, that is it
        does not exhibit a method comparisons_to_ref(k,l) and an attribute
        n_examples.

    Notes
    -----
    To be compatible with this linkage the oracle should exhibit a
    method comparisons_to_ref(k,l) that returns a numpy array of shape
    (n_examples, n_examples) containing values in {1,-1,0}. In entry
    (i,j), the value 1 indicates that the quadruplet (i,j,k,l) is
    available, the value -1 indicates that the quadruplet (k,l,i,j) is
    available, and the value 0 indicates that neither of the
    quadruplets is available. This method should be deterministic.

    The oracle should also exhibit an attribute n_examples counting the
    number of examples it handles.

    For an active oracle, a call to comparisons_to_ref(k,l) for a new pair
    (k,l) should return None when the budget of the oracle is reached.

    """

    def __init__(self, oracle):
        if not (hasattr(oracle, "n_examples") and hasattr(oracle, "comparisons_to_ref")
                and callable(getattr(oracle, "comparisons_to_ref"))):
            raise ValueError("Incompatible oracle, callable 'comparisons_to_ref' or attribute 'n_examples' missing.")

        self.kernel = None

        super(OrdinalLinkageKernel, self).__init__(oracle)

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

        if self.kernel is None:
            self.kernel = self._get_kernel()

        n_clusters = len(clusters)

        i, j = None, None

        score_best = -float("inf")

        for p in range(n_clusters):
            for q in range(p+1, n_clusters):
                kernel_pq = self.kernel[clusters[p], :][:, clusters[q]]

                score = np.mean(kernel_pq)

                if score > score_best:
                    i, j = p, q
                    score_best = score

        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)

        return i, j

    def _get_kernel(self):
        """Returns a kernel matrix representing the similarities between all
        the examples and the number of examples handled by the oracle.

        Returns a numpy array of shape (n_examples, n_examples)
        containing the similarties between all the examples handled by
        the oracle.

        Returns
        -------
        kernel : numpy array, shape (n_examples,n_examples)
            A nummpy array of similarities between the examples.

        Notes
        -----
        This method should only be called once as it is not
        deterministic.

        """
        kernel = np.zeros((self.n_examples, self.n_examples))

        combs = list(itertools.combinations(range(self.n_examples), 2))
        random.shuffle(combs)

        for k, l in combs:
            comparisons = self.oracle.comparisons_to_ref(k, l)

            # Check whether the budget is exhausted for an active oracle
            if comparisons is None:
                break

            for i in range(self.n_examples):
                kernel[i, i+1:] += (comparisons[i+1:, :].astype(int)
                                    @ comparisons[i, :].astype(int))

        kernel += kernel.transpose()

        return kernel


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


class OrdinalLinkageSingle(OrdinalLinkage):
    """An ordinal linkage that controls the merging order in hierarchical
    clustering assuming that the oracle returns quadruplets.

    This method directly use the quadruplets in a single linkage
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
        does not exhibit a method comparisons_single(i,j,k,l) and an
        attribute n_examples.

    Notes
    -----
    To be compatible with this linkage the oracle should exhibit a
    method comparisons_single(i,j,k,l) that returns a value in
    {1,-1,0}. In entry (i,j), the value 1 indicates that the
    quadruplet (i,j,k,l) is available, the value -1 indicates that the
    quadruplet (k,l,i,j) is available, and the value 0 indicates that
    neither of the quadruplets is available. This method should be
    deterministic.

    """

    def __init__(self, oracle):
        if not (hasattr(oracle, "comparisons_single")
                and callable(getattr(oracle, "comparisons_single"))):
            raise ValueError("Incompatible oracle, callable 'comparisons_single' missing.")

        super(OrdinalLinkageSingle, self).__init__(oracle)

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

        i, j = 0, 1

        for p in range(n_clusters):
            for q in range(p+1, n_clusters):
                if self._is_closer(clusters[p], clusters[q], clusters[i], clusters[j]):
                    i, j = p, q

        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)

        return i, j

    def _is_closer(self, cluster_p, cluster_q, cluster_i, cluster_j):
        """Returns a Boolean indicating whether the clusters cluster_p and
        cluster_q are closer to each other than the clusters cluster_i
        and cluster_j.

        Parameters
        ----------
        cluster_p : list of examples
            The first cluster of the first pair.

        cluster_q : list of examples
            The second cluster of the first pair.

        cluster_i : list of examples
            The first cluster of the second pair.

        cluster_j : list of examples
            The second cluster of the second pair.

        Returns
        -------
        : Boolean
            Whether cluster_p and cluster_q are closer to each other
            than cluster_i and cluster_j.

        """
        cluster_p_ref = cluster_p[0]
        cluster_q_ref = cluster_q[0]

        for k in cluster_p:
            for l in cluster_q:
                if self.oracle.comparisons_single(k, l, cluster_p_ref, cluster_q_ref) == 1:
                    cluster_p_ref = k
                    cluster_q_ref = l

        cluster_i_ref = cluster_i[0]
        cluster_j_ref = cluster_j[0]

        for k in cluster_i:
            for l in cluster_j:
                if self.oracle.comparisons_single(k, l, cluster_i_ref, cluster_j_ref) == 1:
                    cluster_i_ref = k
                    cluster_j_ref = l

        return self.oracle.comparisons_single(cluster_p_ref, cluster_q_ref, cluster_i_ref, cluster_j_ref) == 1


class OrdinalLinkageComplete(OrdinalLinkage):
    """An ordinal linkage that controls the merging order in hierarchical
    clustering assuming that the oracle returns quadruplets.

    This method directly use the quadruplets in a complete linkage
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
        does not exhibit a method comparisons_single(i,j,k,l) and an
        attribute n_examples.

    Notes
    -----
    To be compatible with this linkage the oracle should exhibit a
    method comparisons_single(i,j,k,l) that returns a value in
    {1,-1,0}. In entry (i,j), the value 1 indicates that the
    quadruplet (i,j,k,l) is available, the value -1 indicates that the
    quadruplet (k,l,i,j) is available, and the value 0 indicates that
    neither of the quadruplets is available. This method should be
    deterministic.

    """

    def __init__(self, oracle):
        if not (hasattr(oracle, "comparisons_single")
                and callable(getattr(oracle, "comparisons_single"))):
            raise ValueError("Incompatible oracle, callable 'comparisons_single' missing.")

        super(OrdinalLinkageComplete, self).__init__(oracle)

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

        i, j = 0, 1

        for p in range(n_clusters):
            for q in range(p+1, n_clusters):
                if self._is_closer(clusters[p], clusters[q], clusters[i], clusters[j]):
                    i, j = p, q

        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)

        return i, j

    def _is_closer(self, cluster_p, cluster_q, cluster_i, cluster_j):
        """Returns a Boolean indicating whether the clusters cluster_p and
        cluster_q are closer to each other than the clusters cluster_i
        and cluster_j.

        Parameters
        ----------
        cluster_p : list of examples
            The first cluster of the first pair.

        cluster_q : list of examples
            The second cluster of the first pair.

        cluster_i : list of examples
            The first cluster of the second pair.

        cluster_j : list of examples
            The second cluster of the second pair.

        Returns
        -------
        : Boolean
            Whether cluster_p and cluster_q are closer to each other
            than cluster_i and cluster_j.

        """
        cluster_p_ref = cluster_p[0]
        cluster_q_ref = cluster_q[0]

        for k in cluster_p:
            for l in cluster_q:
                if self.oracle.comparisons_single(cluster_p_ref, cluster_q_ref, k, l) == 1:
                    cluster_p_ref = k
                    cluster_q_ref = l

        cluster_i_ref = cluster_i[0]
        cluster_j_ref = cluster_j[0]

        for k in cluster_i:
            for l in cluster_j:
                if self.oracle.comparisons_single(cluster_i_ref, cluster_j_ref, k, l) == 1:
                    cluster_i_ref = k
                    cluster_j_ref = l

        return self.oracle.comparisons_single(cluster_p_ref, cluster_q_ref, cluster_i_ref, cluster_j_ref) == 1


__all__ = ['Oracle', 'OraclePassive', 'OracleComparisons', 'OracleActive', 'OracleActiveBudget']


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


class OraclePassive(Oracle):
    """An oracle that returns passively queried quadruplets from standard
    data.

    Parameters
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between
        examples. It should take two numpy arrays of shapes
        (n_examples_1,n_features) and (n_examples_2,n_features) and
        return a distance matrix of shape (n_examples_1,n_examples_2).

    proportion_quadruplets : float, optional
        The overall proportion of quadruplets that should be
        generated. (Default: 0.1).

    seed : int or None
        The seed used to initialize the random states. (Default:
        None).

    Attributes
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between examples.

    proportion_quadruplets : float
        The overall proportion of quadruplets that should be generated.

    n_examples : int
        The number of examples.

    comparisons_array : numpy array, shape (n_examples, n_examples, n_examples, n_examples)
        A numpy array of shape (n_examples, n_examples, n_examples,
        n_examples) containing values in {1,-1,0}. In entry (i,j,k,l),
        the value 1 indicates that the quadruplet (i,j,k,l) is
        available, the value -1 indicates that the quadruplet
        (k,l,i,j) is available, and the value 0 indicates that neither
        of the quadruplets is available. Initialized to None until one
        of the comparison methods is called.

    seed : int
        The seed used to initialize the random states.

    """

    def __init__(self, x, metric, proportion_quadruplets=0.1, seed=None):
        self.x = x

        self.metric = metric

        self.proportion_quadruplets = proportion_quadruplets

        self.comparisons_array = None

        n_examples = x.shape[0]
        super(OraclePassive, self).__init__(n_examples, seed)

    def comparisons(self):
        if self.comparisons_array is None:
            self.comparisons_array = self._get_comparisons()

        return self.comparisons_array

    def comparisons_to_ref(self, k, l):
        if self.comparisons_array is None:
            self.comparisons_array = self._get_comparisons()

        return self.comparisons_array[:, :, k, l]

    def comparisons_single(self, i, j, k, l):
        if self.comparisons_array is None:
            self.comparisons_array = self._get_comparisons()

        return self.comparisons_array[i, j, k, l]

    def _get_comparisons(self):
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
        random_state = np.random.RandomState(self.seed)

        similarities = self.metric(self.x, self.x)

        comparisons_array = np.zeros((self.n_examples, self.n_examples,
                                      self.n_examples, self.n_examples), dtype='int8')

        # This is to take into account the symmetry effect that makes us query each quadruplet twice
        proportion_effective = (1-np.sqrt(4-4*self.proportion_quadruplets)/2)

        for i in range(self.n_examples):
            for j in range(i+1, self.n_examples):
                selector = np.triu(random_state.rand(self.n_examples, self.n_examples), 1)
                selector = (selector + selector.transpose()) < proportion_effective

                comparisons_array[i, j, :, :] = np.where(np.logical_and(
                    selector, similarities[i, j] > similarities), 1, 0) + np.where(np.logical_and(selector, similarities[i, j] < similarities), -1, 0)
                comparisons_array[j, i, :, :] = comparisons_array[i, j, :, :]

        comparisons_array -= comparisons_array.transpose()
        comparisons_array = np.clip(comparisons_array, -1, 1)

        return comparisons_array


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


class OracleActive(Oracle):
    """An oracle that returns actively queried quadruplets from standard
    data.

    Parameters
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between
        examples. It should take two numpy arrays of shapes
        (n_examples_1,n_features) and (n_examples_2,n_features) and
        return a distance matrix of shape (n_examples_1,n_examples_2).

    seed : int or None
        The seed used to initialize the random states. (Default:
        None).

    Attributes
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between examples.

    n_examples : int
        The number of examples.

    similarities : numpy array, shape (n_examples,n_examples)
        A numpy array containing the similarities between all the
        examples. Initialized to None until the first call to one of
        the comparisons method.

    seed : int
        The seed used to initialize the random states.

    """

    def __init__(self, x, metric, seed=None):
        self.x = x
        n_examples = x.shape[0]

        self.metric = metric

        self.similarities = None

        super(OracleActive, self).__init__(n_examples, seed)

    def comparisons(self):
        raise NotImplemented("Querying all the quadruplets with an active oracle is prohibited.")

    def comparisons_to_ref(self, k, l):
        if self.similarities is None:
            self.similarities = self.metric(self.x, self.x)

        comparisons_array = (self.similarities > self.similarities[k, l])*1 - (self.similarities < self.similarities[k, l])*1

        return comparisons_array

    def comparisons_single(self, i, j, k, l):
        if self.similarities is None:
            self.similarities = self.metric(self.x, self.x)

        comparisons_array = (self.similarities[i, j] > self.similarities[k, l])*1 - \
            (self.similarities[i, j] < self.similarities[k, l])*1

        return comparisons_array


class OracleActiveBudget(OracleActive):
    """An oracle that returns actively queried quadruplets from standard
    data within a given budget.

    This oracle queries quadruplets with respect to reference pairs
    only and can only query a limited number of quadruplets
    (controlled by a proportion of quadruplets that can be queried).

    Parameters
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between
        examples. It should take two numpy arrays of shapes
        (n_examples_1,n_features) and (n_examples_2,n_features) and
        return a distance matrix of shape (n_examples_1,n_examples_2).

    proportion_quadruplets : float, optional
        The overall proportion of quadruplets that should be
        generated. (Default: 0.1).

    seed : int or None
        The seed used to initialize the random states. (Default:
        None).

    Attributes
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the examples.

    metric : function
        The metric to use to compute the similarity between examples.

    proportion_quadruplets : float
        The overall proportion of quadruplets that should be generated.

    budget : int
        The maximum number of pairs that can be queried.

    n_examples : int
        The number of examples.

    references : list of (int,int)
        A list containing the reference pairs that have already been
        queried.

    similarities : numpy array, shape (n_examples,n_examples)
        A numpy array containing the similarities between all the
        examples. Initialized to None until the first call to one of
        the comparisons method.

    seed : int
        The seed used to initialize the random states.

    """

    def __init__(self, x, metric, proportion_quadruplets=0.1, seed=None):
        super(OracleActiveBudget, self).__init__(x, metric, seed)

        self.proportion_quadruplets = proportion_quadruplets

        # Compute the number of quadruplets for n_examples (excluding obvious quadruplets)
        # First substraction is to remove obvious quadruplets of the form i,i,k,l and i,i,l,k and k,l,i,i and l,k,i,i
        # Second substraction is to remove obvious quadruplets of the form i,i,j,j
        # Third substraction is to remove obvious quadruplets of the form i,j,i,j and j,i,i,j and i,j,j,i and j,i,j,i
        # Divided by 8 since each quadruplet has 8 counterparts with the same meaning
        effective_quadruplets = (self.n_examples**4 - 2*self.n_examples*self.n_examples *
                                 (self.n_examples-1) - self.n_examples**2 - 2*self.n_examples*(self.n_examples-1))/8
        # Compute the number of effective quadruplets for a given reference pair, the -1 is to account for the case i,j,i,j
        effective_quadruplets_ref = self.n_examples*(self.n_examples-1)/2 - 1
        # After rep repetitions the effective_quadruplets_ref of the new pair is effective_quadruplets_ref-rep+1 because of the symmetry
        # Hence we have to sovle rep*(effective_quadruplets_ref+effective_quadruplets_ref-rep+1)/2 <= effective_quadrupletss*proportion_quadruplets
        if self.proportion_quadruplets >= 1:
            self.budget = int(self.n_examples*(self.n_examples-1)/2)
        else:
            self.budget = int(effective_quadruplets_ref + 1/2 - np.sqrt((2*effective_quadruplets_ref + 1)
                              ** 2 - 8*effective_quadruplets*self.proportion_quadruplets)/2)

        self.references = []

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
            neither of the quadruplets is available. This method is
            deterministic. This array is None if the budget has been
            reached.

        """
        if self.similarities is None:
            self.similarities = self.metric(self.x, self.x)

        if k > l:
            k, l = l, k

        if (k, l) in self.references:
            comparisons_array = (self.similarities > self.similarities[k, l]) * \
                                 1 - (self.similarities < self.similarities[k, l])*1
        elif self.budget > len(self.references):
            comparisons_array = (self.similarities > self.similarities[k, l]) * \
                                 1 - (self.similarities < self.similarities[k, l])*1
            self.references.append((k, l))
        else:
            comparisons_array = None

        return comparisons_array

    def comparisons_single(self, i, j, k, l):
        raise NotImplemented("Querying a single quadruplet with a budgeted active oracle is prohibited.")


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


def flatten(l: list[list]) -> list:
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
    >>> xs, ys = make_blobs(n_samples=[10, 10], centers=means, cluster_std=stds, n_features=2,
    >>>                   random_state=2)
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

    def score(self, X: Union[np.ndarray, tuple[np.ndarray,  np.ndarray]], y: np.ndarray) -> float:
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
