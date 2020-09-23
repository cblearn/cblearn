from pathlib import Path
import logging
import joblib
import os
from typing import Optional, Union

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch


ARCHIVE = _base.RemoteFileMetadata(
    filename='musicseer-results-2002-10-15.txt',
    url='https://labrosa.ee.columbia.edu/projects/musicsim/musicseer.org/results/musicseer-results-2002-10-15.txt',
    checksum=('149bc77d872d35cbd139833d29b788545873c4d1160a57a59f5ad8b9507bbad0'))


logger = logging.getLogger(__name__)

# todo: own datahome
def fetch_musician_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                              shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                              return_triplets: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the MusicSeer musician similarity dataset (triplets).

    ===================   =====================
    Triplets                             224793
    Objects (Musicians)                     413
    Dimensionality                      unknown
    ===================   =====================

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
            data : ndarray, shape (n_triplets, 3)
                Each row corresponding a triplet constraint.
                The columns represent the target, choosen and other musician index.
            judgement_id : np.ndarray, shape (n_triplets, 1)
                Id of survey questions.
            survey_or_game : np.ndarray, shape (n_triplets, 1)
                Letter 'S' or 'G' indicating if comparison origins from survey or game.
            user : np.ndarray, shape (n_triplets, 1)
                Array of the user ids, answering the triplet question
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

    filepath = Path(_base._pkl_filepath(data_home, 'musician_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading musician similarity from {} to {}'.format(ARCHIVE.url, data_home))

        archive_path = _base._fetch_remote(ARCHIVE, dirname=data_home)
        data_dtype = {'names': ('judgement', 'survey', 'user', 'target', 'chosen', 'other'),
                      'formats': ('<u4', 'S1', 'S5', '<u4', '<u4', '<u4', '<u4')}
        musicians_data = np.genfromtxt(archive_path, dtype=data_dtype, delimiter=' ', invalid_raise=False)

        joblib.dump(musicians_data, filepath, compress=6)
        os.remove(archive_path)
    else:
        musicians_data = joblib.load(filepath)

    if shuffle:
        random_state = check_random_state(random_state)
        musicians_data = random_state.permutation(musicians_data)

    triplets = np.empty((len(musicians_data), 3), dtype=np.int32)
    triplets[:, 0] = musicians_data['target']
    triplets[:, 1] = musicians_data['chosen']
    triplets[:, 2] = musicians_data['other']

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'musician_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    if return_triplets:
        return triplets

    return Bunch(data=triplets,
                 judgement_id=musicians_data['judgement'],
                 survey_or_game=musicians_data['survey'],
                 user=musicians_data['user'],
                 DESCR=fdescr)


