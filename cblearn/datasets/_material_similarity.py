from pathlib import Path
import logging
import joblib
import json
import os
from typing import Optional, Union
import zipfile

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch
from cblearn.utils import check_query_response


ARCHIVE = _base.RemoteFileMetadata(
    filename='material-appearance-similarity-master.zip',
    url='https://github.com/mlagunas/material-appearance-similarity/archive/refs/heads/master.zip',
    checksum=('f0be4d573829fd5e5a7e7b332989545cbf6584eaf25e2555371703a9264f5937'))

logger = logging.getLogger(__name__)


def fetch_material_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                              shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                              return_triplets: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the material similarity dataset (triplets).

    ===================   =====================
    Triplets Train/Test            22801 / 3000
    Responses                     92892 / 11800
    Objects (Materials)                     100
    ===================   =====================

    See :ref:`material_similarity_dataset` for a detailed description.

    >>> dataset = fetch_material_similarity(shuffle=True)  # doctest: +REMOTE_DATA
    >>> dataset.material_name[[0, -1]].tolist()  # doctest: +REMOTE_DATA
    ['alum-bronze', 'yellow-plastic']
    >>> dataset.triplet.shape, dataset.response.shape  # doctest: +REMOTE_DATA
    ((92892, 3), (92892,))

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
                Each row corresponding a triplet constraint.
                The columns represent the reference and two other material indices.
            response : ndarray, shape (n_triplets, )
                The count of subject responses that chose the first other (positive) or second other (negative)
                material to be more similar to the reference material.
            test_triplet : ndarray, shape (n_test_triplets, 3)
                handoff test set.
            test_response : ndarray, shape (n_test_triplets, )
                handoff test set.
            material_name : ndarray, shape (100, )
                Names of the materials.
            DESCR : string
                Description of the dataset.
        triplets, response : numpy arrays (n_triplets, 3) and (n_triplets, )
            Only present when `return_triplets=True`.
    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """

    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    filepath = Path(_base._pkl_filepath(data_home, 'material_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading material similarity data from {} to {}'.format(ARCHIVE.url, data_home))

        archive_path = _base._fetch_remote(ARCHIVE, dirname=data_home)
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open('material-appearance-similarity-master/data/answers_processed_test.json', 'r') as f:
                test_data = json.load(f)
            with zf.open('material-appearance-similarity-master/data/answers_processed_train.json', 'r') as f:
                train_data = json.load(f)

            image_path = 'material-appearance-similarity-master/data/havran1_ennis_298x298_LDR/'
            material_names = np.asarray([name[len(image_path):-len('.jpg')] for name in zf.namelist()
                                        if name.startswith(image_path) and name.endswith('.jpg')])
            material_names.sort()
        joblib.dump((train_data, test_data, material_names), filepath, compress=6)
        os.remove(archive_path)
    else:
        (train_data, test_data, material_names) = joblib.load(filepath)

    train_triplets = np.array(train_data['answers'])
    train_agreement = np.array(train_data['agreement'])
    train_triplets_1, train_response_1 = check_query_response(train_triplets[train_agreement[:, 0] > 0],
                                                              train_agreement[train_agreement[:, 0] > 0][:, 0],
                                                              result_format='list-count')
    train_triplets_2, train_response_2 = check_query_response(train_triplets[train_agreement[:, 1] > 0],
                                                              train_agreement[train_agreement[:, 1] > 0][:, 1],
                                                              result_format='list-count')
    train_triplets, train_response = np.r_[train_triplets_1, train_triplets_2], np.r_[train_response_1, train_response_2]

    test_triplets = np.array(test_data['answers'])
    test_agreement = np.array(test_data['agreement'])
    test_triplets_1, test_response_1 = check_query_response(test_triplets[test_agreement[:, 0] > 0],
                                                            test_agreement[test_agreement[:, 0] > 0][:, 0],
                                                            result_format='list-count')
    test_triplets_2, test_response_2 = check_query_response(test_triplets[test_agreement[:, 1] > 0],
                                                            test_agreement[test_agreement[:, 1] > 0][:, 1],
                                                            result_format='list-count')
    test_triplets, test_response = np.r_[test_triplets_1, test_triplets_2], np.r_[test_response_1, test_response_2]

    if shuffle:
        random_state = check_random_state(random_state)
        train_ix = random_state.permutation(len(train_triplets))
        train_triplets, train_response = train_triplets[train_ix], train_response[train_ix]
        test_ix = random_state.permutation(len(test_triplets))
        test_triplets, test_response = test_triplets[test_ix], test_response[test_ix]

    if return_triplets:
        return train_triplets, train_response

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'material_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(triplet=train_triplets,
                 response=train_response,
                 test_triplet=test_triplets,
                 test_response=test_response,
                 material_name=material_names,
                 DESCR=fdescr)
