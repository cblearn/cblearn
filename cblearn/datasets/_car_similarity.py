from pathlib import Path
import logging
import joblib
import os
from typing import Optional, Union
import zipfile

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch


ARCHIVE = _base.RemoteFileMetadata(
    filename='60_cars_data.zip',
    url='http://www.tml.cs.uni-tuebingen.de/team/luxburg/code_and_data/60_cars_data.zip',
    checksum=('5fa2ad932d48adf5cfe36bd16a08b25fd88d1519d974908f6ccbba769f629640'))

logger = logging.getLogger(__name__)


def fetch_car_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                         shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                         return_triplets: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the 60-car dataset (most-central triplets).

    ===================   =====================
    Triplets                               7097
    Objects (Cars)                           60
    Query                  3 cars, most-central
    Sessions                                146
    Queries per Session                   30-50
    ===================   =====================

    See :ref:`central_car_dataset` for a detailed description.

    >>> dataset = fetch_car_similarity(shuffle=False)  # doctest: +REMOTE_DATA
    >>> dataset.class_name.tolist()  # doctest: +REMOTE_DATA
    ['OFF-ROAD / SPORT UTILITY VEHICLES', 'ORDINARY CARS', 'OUTLIERS', 'SPORTS CARS']
    >>> dataset.triplet.shape  # doctest: +REMOTE_DATA
    (7097, 3)
    >>> rounds, round_count = np.unique(dataset.survey_round, return_counts=True)  # doctest: +REMOTE_DATA
    >>> len(rounds), round_count.min(), round_count.max()  # doctest: +REMOTE_DATA
    (146, 30, 50)

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
                The columns represent the three indices shown per most-central question.
            response : ndarray, shape (n_triplets, )
                The car per question (0, 1, or 2) that was selected as "most-central".
            survey_round : ndarray of int, shape (n_triplets, )
                Survey rounds, grouping responses from a participant during a session.
                Some participants responded in multiple rounds at different times.
            rt : ndarray of float, shape (n_triplets, )
                Reaction time of the response in seconds.
            class_id : np.ndarray (60, )
                The class assigned to each object.
            class_name : list (4)
                Names of the classes.
            DESCR : string
                Description of the dataset.
        triplets : numpy array (n_triplets, 3)
            Only present when `return_triplets=True`.

    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """

    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    filepath = Path(_base._pkl_filepath(data_home, 'car_centrality.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading 60-car dataset from {} to {}'.format(ARCHIVE.url, data_home))

        archive_path = _base._fetch_remote(ARCHIVE, dirname=data_home)
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open('60_cars_data/survey_data.csv', 'r') as f:
                survey_data = np.loadtxt(f, dtype=str, delimiter=',', skiprows=1)

        joblib.dump(survey_data, filepath, compress=6)
        os.remove(archive_path)
    else:
        survey_data = joblib.load(filepath)

    class_map = {
        'ORDINARY CARS': [2, 6, 7, 8, 9, 10, 11, 12, 16, 17, 25, 32, 35, 36, 37, 38,
                          39, 41, 44, 45, 46, 55, 58, 60],
        'SPORTS CARS': [15, 19, 20, 28, 40, 42, 47, 48, 49, 50, 51, 52, 54, 56, 59],
        'OFF-ROAD / SPORT UTILITY VEHICLES': [1, 3, 4, 5, 13, 14, 18, 22, 24, 26, 27,
                                              29, 31, 33, 34, 43, 57],
        'OUTLIERS': [21, 23, 30, 53],
    }
    class_names = np.asarray(sorted(class_map.keys()))
    classes = np.empty(60, dtype=int)
    for cls_ix, cls_name in enumerate(class_names):
        classes[np.array(class_map[cls_name]) - 1] = cls_ix

    if shuffle:
        random_state = check_random_state(random_state)
        shuffle_ix = random_state.permutation(len(survey_data))
        survey_data = survey_data[shuffle_ix]

    raw_triplets = survey_data[:, [2, 3, 4]].astype(int)
    triplets = raw_triplets - 1
    response = (survey_data[:, [1]].astype(int) == raw_triplets).nonzero()[1]
    survey_round = survey_data[:, [0]].astype(int)
    rt = survey_data[:, [5]].astype(float)
    if return_triplets:
        return triplets

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'car_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(triplet=triplets,
                 response=response,
                 survey_round=survey_round,
                 rt=rt,
                 class_id=classes,
                 class_name=class_names,
                 DESCR=fdescr)
