import csv
from pathlib import Path
import logging
import joblib
import os
from typing import Optional, Union

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch


ARCHIVE = _base.RemoteFileMetadata(
    filename='musicseer-results-2002-04-29.txt',
    url='https://www.labrosa.org/projects/musicsim/musicseer.org/results/musicseer-results-2002-04-29.txt',
    checksum=('4506fc16a8f9a30a92207720bc599602b514de20e94494f67eb7a1e348401e2d'))

logger = logging.getLogger(__name__)


def fetch_musician_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                              shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                              return_triplets: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the MusicSeer musician similarity dataset (triplets).

    ===================   =====================
    Triplets                            131.970
    Objects (Artists)                       448
    Dimensionality                      unknown
    ===================   =====================

    See :ref:`musician_similarity_dataset` for a detailed description.

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
            judgement_id : np.ndarray, shape (n_triplets, )
                Id of survey Query.
            survey_or_game : np.ndarray, shape (n_triplets,)
                Letter 'S' or 'G' indicating if comparison origins from survey or game.
            user : np.ndarray, shape (n_triplets, )
                Array of the user ids, answering the triplet question
            artist_name : np.ndarray, shape (413,)
                Names of artists, corresponding to the triplet indices.
            artist_id : np.ndarray, shape (413,)
                Ids of artists, corresponding to the triplet indices.
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

    filepath = Path(_base._pkl_filepath(data_home, 'musician_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading musician similarity from {} to {}'.format(ARCHIVE.url, data_home))

        archive_path = _base._fetch_remote(ARCHIVE, dirname=data_home)
        with open(archive_path, 'r') as f:
            cols = ('judgement', 'survey', 'user', 'target', 'chosen', 'other')
            musicians_data = np.array(list(csv.reader(f, delimiter=' '))[1:-2]).T
            musicians_data = {k: v for k, v in zip(cols, musicians_data)}

        joblib.dump(musicians_data, filepath, compress=6)
        os.remove(archive_path)
    else:
        musicians_data = joblib.load(filepath)

    if shuffle:
        random_state = check_random_state(random_state)
        ix = random_state.permutation(len(musicians_data['target']))
        musicians_data = {k: v[ix] for k, v in musicians_data.items()}

    module_path = Path(__file__).parent
    artists = np.genfromtxt(module_path.joinpath('data', 'musician_names.txt'), delimiter=' ',
                            dtype={'names': ('name', 'id'), 'formats': ('U29', '<i8')})

    triplet_filter = musicians_data['other'] != ''   # remove bi-tuples.
    triplet_ids = np.c_[musicians_data['target'], musicians_data['chosen'], musicians_data['other']]
    triplet_ids = triplet_ids[triplet_filter].astype(int)

    all_ids, triplets = np.unique(triplet_ids, return_inverse=True)
    triplets = triplets.reshape(triplet_ids.shape)

    if return_triplets:
        return triplets

    ids_to_names = {id: n for id, n in zip(artists['id'], artists['name'])}
    for ix, id in enumerate(sorted(np.setdiff1d(all_ids, artists['id']))):
        ids_to_names[int(id)] = f"unknown_{ix}"

    with module_path.joinpath('descr', 'musician_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=triplets,
                 judgement_id=musicians_data['judgement'][triplet_filter],
                 survey_or_game=musicians_data['survey'][triplet_filter],
                 user=musicians_data['user'][triplet_filter],
                 artist_name=np.asarray([ids_to_names[id] for id in all_ids]),
                 artist_id=all_ids,
                 DESCR=fdescr)
