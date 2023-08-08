from pathlib import Path
import logging
import joblib
import os
from typing import Optional, Union
import zipfile
import ssl

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch


ARCHIVE = _base.RemoteFileMetadata(
    filename='food100-dataset.zip',
    url='http://vision.cornell.edu/se3/wp-content/uploads/2014/09/food100-dataset.zip',
    checksum=('18f5e210174dfdbf6a7b4ed7538cf8ba53fd65e0cbe193519231b8ab4ea8fc62'))

logger = logging.getLogger(__name__)


def fetch_food_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                          shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                          return_triplets: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the Food-100 food similarity dataset (triplets).

    .. warning::
        This function downloads the file without verifying the ssl signature to circumvent an outdated certificate of the dataset hosts.
        However, after downloading the function verifies the file checksum before loading the file to minimize the risk of man-in-the-middle attacks.
    
    ===================   =====================
    Triplets                             190376
    Objects                                 100
    Dimensionality                      unknown
    ===================   =====================

    See :ref:`food_similarity_dataset` for a detailed description.

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

            data: ndarray, shape (n_triplets, 3)
                Each row corresponding a triplet constraint.
                The columns represent the target, more similar and more distant food index.
            image_names : ndarray, shape (n_objects,)
                The food image names corresponding to the indices.
            DESCR : string
                Description of the dataset.
        triplets : numpy array (n_triplets, 3)
            Only present when `return_triplets=True`.

    Raises:
        IOError: If the data is not locally available, but downlaod_if_missing=False
    """

    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    filepath = Path(_base._pkl_filepath(data_home, 'food_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading food similarity from {} to {}'.format(ARCHIVE.url, data_home))

        try:
            ssl_default = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            archive_path = _base._fetch_remote(ARCHIVE, dirname=data_home)
        finally:
            ssl._create_default_https_context = ssl_default
            
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open('food100-dataset/all-triplets.csv', 'r') as f:
                triplets = np.loadtxt(f, dtype=str, delimiter=';')

            image_names = np.asarray([name[len('food100-dataset/'):] for name in zf.namelist()
                                      if name.startswith('food100-dataset/images/')
                                      and name.endswith('.jpg')])

        joblib.dump((triplets, image_names), filepath, compress=6)
        os.remove(archive_path)
    else:
        triplets, image_names = joblib.load(filepath)

    image_names = np.sort(image_names)
    triplets = np.searchsorted(image_names, triplets)

    if shuffle:
        random_state = check_random_state(random_state)
        triplets = random_state.permutation(triplets)

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'food_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    if return_triplets:
        return triplets

    return Bunch(data=triplets,
                 image_names=image_names,
                 DESCR=fdescr)
