from pathlib import Path
import logging
import joblib
import os
from os.path import join
from typing import Optional, Union
from urllib.request import urlretrieve
import zipfile

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch

ARCHIVE = _base.RemoteFileMetadata(
    filename='osfstorage-archive.zip',
    url='https://files.osf.io/v1/resources/7f96y/providers/osfstorage/?zip=',
    checksum=('cannot check - zip involves randomness'))

logger = logging.getLogger(__name__)
__doctest_requires__ = {'fetch_imagenet_similarity': ['h5py']}


def fetch_imagenet_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                              shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                              version: str = '0.1', return_data: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the imagenet similarity dataset (rank 2 from 8).

    ===================   =====================
    Trials    v0.1/v0.2        25,273 / 384,277
    Objects (Images)             1,000 / 50,000
    Classes                               1,000
    Query                         rank 2 from 8
    ===================   =====================

    See :ref:`imagenet_similarity_dataset` for a detailed description.

    .. Note :
        Loading dataset requires the package `h5py`_, which was not installed as an dependency of cblearn.

    .. _`h5py`: https://docs.h5py.org/en/stable/build.html

    >>> dataset = fetch_imagenet_similarity(shuffle=True, version='0.1')  # doctest: +REMOTE_DATA
    >>> dataset.class_label[[0, -1]].tolist()  # doctest: +REMOTE_DATA
    ['n01440764', 'n15075141']
    >>> dataset.n_select, dataset.is_ranked  # doctest: +REMOTE_DATA
    (2, True)
    >>> dataset.data.shape  # doctest: +REMOTE_DATA
    (25273, 9)

    Args:
        data_home : optional, default: None
            Specify another download and cache folder for the datasets. By default
            all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
        download_if_missing : optional, default=True
        shuffle: default = True
            Shuffle the order of triplet constraints.
        random_state: optional, default = None
            Initialization for shuffle random generator
        version: Version of the dataset.
            '0.1' contains one object per class,
            '0.2' 50 objects per class.
        return_triplets : boolean, default=False.
            If True, returns numpy array instead of a Bunch object.

    Returns:
        dataset : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            data : ndarray, shape (n_query, 9)
                Each row corresponding a rank-2-of-8 query, entries are object indices.
                The first column is the reference, the second column is the most similar, and the
                third column is the second most similar object.
            rt_ms : ndarray, shape (n_query, )
                Reaction time in milliseconds.
            n_select : int
                Number of selected objects per trial.
            is_ranked : bool
                Whether the selection is ranked in similarity to the reference.
            session_id : (n_query,)
                Ids of the survey session for query recording.
            stimulus_id : (50.000,)
                Ids of the images.
            stimulus_filepath : (50.000,)
                Filepaths of images.
            class_id : (50.000,)
                ImageNet class assigned to each image.
            class_label : (1.000,)
                WordNet labels of the classes.
            DESCR : string
                Description of the dataset.
        data : numpy arrays (n_query, 9)
            Only present when `return_data=True`.

    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """
    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    filepath = Path(_base._pkl_filepath(data_home, 'imagenet_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading imagenet similarity data from {} to {}'.format(ARCHIVE.url, data_home))

        archive_path = (ARCHIVE.filename if data_home is None
                        else join(data_home, ARCHIVE.filename))
        urlretrieve(ARCHIVE.url, archive_path)

        with zipfile.ZipFile(archive_path) as zf:
            try:
                import h5py
            except ImportError:
                raise ImportError(
                    "This function needs the extra package 'h5py' but could not find it.\n"
                    "The package can be installed with pip install h5py.\n"
                    "On some platforms you might have to install hdf5 libraries separately.")

            with zf.open('data/deprecated/psiz0.4.1/obs-118.hdf5', 'r') as f:
                data_v1 = {k: np.asarray(v[()]) for k, v in h5py.File(f, mode='r').items()}

            with zf.open('data/deprecated/psiz0.4.1/obs-195.hdf5', 'r') as f:
                data_v2 = {k: np.asarray(v[()]) for k, v in h5py.File(f, mode='r').items()}

            with zf.open('data/deprecated/psiz0.4.1/catalog.hdf5', 'r') as f:
                catalog = {k: np.asarray(v[()]) for k, v in h5py.File(f, mode='r').items()}

        joblib.dump((data_v1, data_v2, catalog), filepath, compress=6)
        os.remove(archive_path)
    else:
        (data_v1, data_v2, catalog) = joblib.load(filepath)

    if str(version) == '0.1':
        data = data_v1
    elif str(version) == '0.2':
        data = data_v2
    else:
        raise ValueError(f"Expects version '0.1' or '0.2', got '{version}'.")

    data.pop('trial_type')
    catalog['class_map_label'] = catalog['class_map_label'].astype(str)
    catalog['stimulus_filepath'] = catalog['stimulus_filepath'].astype(str)

    if shuffle:
        random_state = check_random_state(random_state)
        ix = random_state.permutation(len(data['stimulus_set']))
        data = {k: v[ix] for k, v in data.items()}

    if return_data:
        return data['stimulus_set']

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'imagenet_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=data['stimulus_set'],
                 rt_ms=data['rt_ms'],
                 n_select=int(np.unique(data['n_select'])[0]),
                 is_ranked=bool(np.unique(data['is_ranked'])[0]),
                 session_id=data['session_id'],
                 stimulus_id=catalog['stimulus_id'],
                 stimulus_filepath=catalog['stimulus_filepath'],
                 class_id=catalog['class_id'],
                 class_label=catalog['class_map_label'][1:],
                 DESCR=fdescr)
