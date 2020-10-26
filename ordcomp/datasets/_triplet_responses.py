""" Function in this file judge triplets, based on ground-truth embedding and possible noise patterns. """

from sklearn.utils import check_random_state
import numpy as np

from ordcomp import utils


def triplet_responses(triplets, responses=None, embedding=None, noise=None, noise_options={}, random_state=None):
    input_dim = embedding.shape[1]
    triplets = utils.check_triplets(triplets, responses, format='array', response_type='implicit')

    y_triplets = embedding[triplets.ravel()].reshape(-1, 3 * input_dim)
    if isinstance(noise, str):
        random_state = check_random_state(random_state)
        noise = getattr(random_state, noise)
    if noise:
        y_triplets += noise(size=y_triplets.shape, **noise_options)

    pivot, left, right = (y_triplets[:, 0:input_dim],
                          y_triplets[:, input_dim:(2 * input_dim)],
                          y_triplets[:, (2 * input_dim):])
    return np.linalg.norm(pivot - left, axis=1) < np.linalg.norm(pivot - right, axis=1)


def noisy_distances(y, noise=None, options={}, clip=False, symmetrize=True, random_state=None, **kwargs):
    random_state = check_random_state(random_state)
    if clip is True:
        clip = (y.min(), y.max())
    if noise and False:
        # TODO: fix this or remove funciton
        pass
        # y1 = _add_noise(y, noise, options, clip, random_state)
        # y2 = _add_noise(y, noise, options, clip, random_state)
    else:
        y1, y2 = y, y
    distances = sklearn.metric.pairwise.pairwise_distances(y1, y2, **kwargs)
    if symmetrize:
        distances = np.triu(distances, k=0) + np.triu(distances, k=1).T
    return distances
