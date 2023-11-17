import csv
import io
from pathlib import Path
import logging
import joblib
import os
from os.path import join
from typing import Optional, Union
from urllib.request import urlretrieve
import zipfile
import scipy
import warnings

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch

ARCHIVE = _base.RemoteFileMetadata(
    filename='osfstorage-archive.zip',
    url='https://files.osf.io/v1/resources/z2784/providers/osfstorage/?zip=',
    checksum=('cannot check - zip involves randomness'))

logger = logging.getLogger(__name__)


def fetch_things_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                            shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                            return_data: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the things similarity dataset (odd-one-out).

    ===================   =====================
    Trials                              146,012
    Objects (Things)                      1,854
    Query                 3 images, odd one out
    ===================   =====================

    See :ref:`things_similarity_dataset` for a detailed description.

    >>> dataset = fetch_things_similarity(shuffle=True)  # doctest: +REMOTE_DATA
    >>> dataset.word[[0, -1]].tolist()  # doctest: +REMOTE_DATA
    ['aardvark', 'zucchini']
    >>> dataset.data.shape  # doctest: +REMOTE_DATA
    (146012, 3)
    >>> dataset.embedding.shape  # doctest: +REMOTE_DATA
    (1854, 49)
    >>> dataset.images.shape  # doctest: +REMOTE_DATA
    (1854, 150, 150, 3)

    Args:
        data_home : optional, default: None
            Specify another download and cache folder for the datasets. By default
            all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
        download_if_missing : optional, default=True
        shuffle: default = True
            Shuffle the order of triplet constraints.
        random_state: optional, default = None
            Initialization for shuffle random generator
        return_triplets : boolean, default=False.
            If True, returns numpy array instead of a Bunch object.

    Returns:
        dataset : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            data : ndarray, shape (n_query, 3)
                Each row corresponding a odd-one-out query, entries are object indices.
                The first column is the selected odd-one.
            word : (n_objects,)
                Single word associated with the thing objects.
            uid : (n_objects,)
                Unique word that identifies the images.
            images: (n_objects, 150, 150, 3)
                Preview images of the stimuli.
            embedding : (n_objects, 49)
                Sparse 49d embedding of the thing object based on the odd-one-out queries.
            synset : (n_objects,)
                Wordnet Synset associated with the thing objects.
            wordnet_id : (n_objects,)
                Wordnet Id associated with the thing objects.
            thing_id : (n_objects,)
                Unique Id string associated with the thing objects.
            DESCR : string
                Description of the dataset.
        data : numpy arrays (n_query, 3)
            Only present when `return_data=True`.

    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """
    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    filepath = Path(_base._pkl_filepath(data_home, 'things_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading imagenet similarity data from {} to {}'.format(ARCHIVE.url, data_home))

        archive_path = (ARCHIVE.filename if data_home is None
                        else join(data_home, ARCHIVE.filename))
        urlretrieve(ARCHIVE.url, archive_path)

        with zipfile.ZipFile(archive_path) as zf:
            with zf.open('variables/unique_id.txt', 'r') as f:
                uid = np.loadtxt(f, delimiter=' ', dtype=str)

            images, _ = _load_images(zf, uid)

            with zf.open('data/spose_embedding_49d_sorted.txt', 'r') as f:
                # explicitly state cols to avoid error due to trailing whitespace
                embedding = np.loadtxt(f, delimiter=' ', usecols=range(0, 49))

            data = _load_test_data(zf)

            with zf.open('items1854names.tsv', 'r') as f:
                objects = np.array(list(csv.reader(io.TextIOWrapper(f), dialect='excel-tab'))[1:]).T

        joblib.dump((data, objects, embedding, uid, images), filepath, compress=6)
        os.remove(archive_path)
    else:
        cache = joblib.load(filepath)
        if len(cache) == 5:
            (data, objects, embedding, uid, images) = cache
        else:  # backwards compatibility
            (data, objects) = cache
            embedding, uid, images = None, None, None
            warnings.warn("Could not load embedding, uid and images from old cache file.\n"
                          f"Consider cleaning cached file {filepath}.")

    if shuffle:
        random_state = check_random_state(random_state)
        data = random_state.permutation(data)

    if return_data:
        return data

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'things_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=data,
                 embedding=embedding,
                 images=images,
                 uid=uid,
                 word=objects[0],
                 synset=objects[1],
                 wordnet_id=objects[2],
                 thing_id=objects[5],
                 DESCR=fdescr)


def _load_images(zf, uid):
    with zf.open('variables/im.mat', 'r') as f:
        mat = scipy.io.loadmat(f)
        images = np.stack([im[0] for im in mat['im']])
        imwords = np.array([w[0][0] for w in mat['imwords']])

    # the order of images is different from
    # the order of words and ids. Let's fix this here.
    curr_order = np.argsort(imwords)
    new_order = np.argsort(uid)
    images[new_order] = images[curr_order]
    imwords[new_order] = imwords[curr_order]
    np.testing.assert_array_equal(uid, imwords)
    return images, imwords


def _load_test_data(zf):
    with zf.open('variables/sortind.mat', 'r') as f:
        sortind = scipy.io.loadmat(f)['sortind'].squeeze() - 1

    with zf.open('data/data1854_batch5_test10.txt', 'r') as f:
        data = np.loadtxt(f, delimiter=' ')

    # the order in the test data is different from the order in the training data,
    # so we need to sort the objects in the test data.
    # Sorting adopted from matlab code in the dataset repo, make_figures_behavsim.m.
    for i_obj in range(1854):
        data[data == sortind[i_obj]] = 10000 + i_obj
    data = data - 10000
    assert data.min() == 0 and data.max() == 1853, "Something went wrong with sorting the test data."
    return data
