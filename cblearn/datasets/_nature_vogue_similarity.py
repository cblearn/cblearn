from pathlib import Path
import logging
import joblib
import os
from typing import Optional, Union
import zipfile

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch
from cblearn.preprocessing import query_from_columns

ARCHIVE = _base.RemoteFileMetadata(
    filename='nature_and_vogue_triplets.zip',
    url='http://anttiukkonen.com/nature_and_vogue_triplets.zip',
    checksum=('db4718bb742a2492dee87e94daead3070b2afa3a9174aa717be336940e47e3cc'))

logger = logging.getLogger(__name__)


def fetch_nature_scene_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                                  shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                                  return_triplets: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the nature scene similarity dataset (odd-one-out).

    ===================   =====================
    Triplets                               3355
    Objects (Scenes)                        120
    ===================   =====================

    See :ref:`nature_vogue_dataset` for a detailed description.

    >>> dataset = fetch_nature_scene_similarity(shuffle=True)  # doctest: +REMOTE_DATA
    >>> dataset.image_label[[0, -1]].tolist() # doctest: +REMOTE_DATA
    ['art114.jpg', 'n344019.jpg']
    >>> dataset.triplet.shape # doctest: +REMOTE_DATA
    (3355, 3)

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

            triplet : ndarray, shape (n_triplets, 3)
                Each row corresponding odd-one-out query.
                The columns represent the odd image and two others.
            class_label : ndarray, shape (120, )
                Names of the scene images.
            DESCR : string
                Description of the dataset.
        triplets : numpy arrays (n_triplets, 3)
            Only present when `return_triplets=True`.

    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """
    return _fetch_nature_vogue('nature', data_home, download_if_missing, shuffle, random_state, return_triplets)


def fetch_vogue_cover_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                                 shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                                 return_triplets: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the vogue cover similarity dataset (odd-one-out).

    ===================   =====================
    Triplets                               1107
    Objects (Covers)                         60
    ===================   =====================

    See :ref:`nature_vogue_dataset` for a detailed description.

    >>> dataset = fetch_vogue_cover_similarity(shuffle=True)  # doctest: +REMOTE_DATA
    >>> dataset.image_label[[0, -1]].tolist()  # doctest: +REMOTE_DATA
    ['Cover_uk_VOgue_MAY10_V_29mar10_bt_268x353.jpg', 'voguecoverapr11_bt_268x353.jpg']
    >>> dataset.triplet.shape  # doctest: +REMOTE_DATA
    (1107, 3)

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

            triplet : ndarray, shape (n_triplets, 3)
                Each row corresponding odd-one-out query.
                The columns represent the odd image and two others.
            class_label : ndarray, shape (120, )
                Names of the scene images.
            DESCR : string
                Description of the dataset.
        triplets : numpy arrays (n_triplets, 3)
            Only present when `return_triplets=True`.

    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """
    return _fetch_nature_vogue('vogue', data_home, download_if_missing, shuffle, random_state, return_triplets)


def _fetch_nature_vogue(dataset: str, data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                        shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                        return_triplets: bool = False) -> Union[Bunch, np.ndarray]:

    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    filepath = Path(_base._pkl_filepath(data_home, 'nature_vogue_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading material similarity data from {} to {}'.format(ARCHIVE.url, data_home))

        archive_path = _base._fetch_remote(ARCHIVE, dirname=data_home)
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open('nature/nature_triplets.txt', 'r') as f:
                nature_data = np.loadtxt(f, dtype=str)

            with zf.open('vogue/vogue_triplets.txt', 'r') as f:
                vogue_data = np.loadtxt(f, dtype=str)

        joblib.dump((nature_data, vogue_data), filepath, compress=6)
        os.remove(archive_path)
    else:
        (nature_data, vogue_data) = joblib.load(filepath)

    if dataset == 'nature':
        data = nature_data[:, [2, 0, 1]]
    elif dataset == 'vogue':
        data = vogue_data[:, [2, 0, 1]]

    triplets, encoder = query_from_columns(data, [0, 1, 2], return_transformer=True)
    classes = encoder.encoder_.classes_

    if shuffle:
        random_state = check_random_state(random_state)
        triplets = random_state.permutation(triplets)

    if return_triplets:
        return triplets

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'nature_vogue_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(triplet=triplets,
                 image_label=classes,
                 DESCR=fdescr)
