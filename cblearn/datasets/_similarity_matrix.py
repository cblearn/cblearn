import logging
import os
import zipfile
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import scipy.io
from sklearn.datasets import _base
from sklearn.utils import Bunch

ARCHIVE = _base.RemoteFileMetadata(
    filename='all.zip',
    url='https://files.osf.io/v1/resources/ey9vp/providers/osfstorage/'
        '5e7a7065d2927f006fdd1cf9?action=download&amp;direct&amp;version=1',
    checksum=('8c799cdebb00192ecb63f3e28c6eeee0e2f64fcb8dad3bc68982e551f2ae5b1c'))

logger = logging.getLogger(__name__)

AVAILABLE_SIMILARITIES = [
    'fruit2_romney', 'nonsense_romney', 'furniture_romney', 'kinship_kimrosenberg', 'rectangle_kruschke',
    'vegetables2_romney', 'animalpictures5', 'auditory', 'druguse', 'faces11', 'fruits', 'dotpatterns',
    'furniture2_romney', 'bodies_viken', 'textures', 'sport_romney', 'bankwiring', 'morsenumbers',
    'faces_busey', 'letters', 'vehicles_romney', 'vehicles2_romney', 'birds_romney', 'fruit_romney', 'risks',
    'morseall', 'texturemit_heaps', 'cartoonfaces', 'country_robinsonhefner', 'congress', 'phonemes',
    'toys_romney', 'colour', 'countriessim', 'faces5', 'tools_romney', 'lines_cohen', 'abstractnumbers',
    'countriesdis', 'animalnames11', 'faces_steyvers', 'weapons2_romney', 'texturebrodatz_heaps',
    'fish_romney', 'flowerpots', 'sizeangle_treat', 'clothing2_romney', 'weapons_romney', 'clothing_romney',
    'animalnames5', 'vegetables_romney', 'animalpictures11']


def fetch_similarity_matrix(name: str, data_home: Optional[os.PathLike] = None, download_if_missing: bool = True
                            ) -> Union[Bunch, np.ndarray]:
    """ Load human similarity judgements, aggregated to a similarity matrix.

    This function provides access to the following similarity matrices:
    `fruit2_romney, nonsense_romney, furniture_romney, kinship_kimrosenberg, rectangle_kruschke, vegetables2_romney,
    animalpictures5, auditory, druguse, faces11, fruits, dotpatterns, furniture2_romney, bodies_viken,
    textures, sport_romney, bankwiring, morsenumbers, faces_busey, letters, vehicles_romney, vehicles2_romney,
    birds_romney, fruit_romney, risks, morseall, texturemit_heaps, cartoonfaces, country_robinsonhefner, congress,
    phonemes, toys_romney, colour, countriessim, faces5, tools_romney, lines_cohen, abstractnumbers, countriesdis,
    animalnames11, faces_steyvers, weapons2_romney, texturebrodatz_heaps, fish_romney, flowerpots, sizeangle_treat,
    clothing2_romney, weapons_romney, clothing_romney, animalnames5, vegetables_romney, animalpictures11`.

    See :ref:`similarity_matrix_dataset` for a detailed description.

    >>> dataset = fetch_similarity_matrix('colour')  # doctest: +REMOTE_DATA
    >>> dataset.labels[:2].tolist()  # doctest: +REMOTE_DATA
    ['434', '445']
    >>> dataset.similarity.shape  # doctest: +REMOTE_DATA
    (14, 14)

    Args:
        name: Name of the similarity dataset
        data_home : optional, default: None
            Specify another download and cache folder for the datasets. By default
            all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
        download_if_missing : optional, default=True

    Returns:
        dataset : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            similarity : ndarray, shape (n_objects, n_objects)
                Symmetric matrix of normalized object similarities.
                None for some datasets.
            proximity : ndarray, shape (n_objects, n_objects)
                Symmetric matrix of normalized pairwise proximities.
                None for some datasets.
            n_objects: int
                Number of objects
            labels : (n_objects,)
                Single word describing each object
            sigma: float
                Uncertainty of the similarity values.
                Not available for all datasets.
            DESCR : string
                Description of the dataset.

    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """
    if name not in AVAILABLE_SIMILARITIES:
        raise ValueError(f"Unexpected similarity name = {name}. Use one of {AVAILABLE_SIMILARITIES}.")

    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    basepath = Path(_base._pkl_filepath(data_home, 'similarity_collection/'))
    filepath = basepath.joinpath(f'{name}.pkz')
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading imagenet similarity data from {} to {}'.format(ARCHIVE.url, data_home))

        archive_path = _base._fetch_remote(ARCHIVE, dirname=data_home)
        if not basepath.exists():
            basepath.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as zf:
            for _this_name in AVAILABLE_SIMILARITIES:
                with zf.open(f'{_this_name}.mat', 'r') as f:
                    _raw = scipy.io.loadmat(f)
                    _this_dict = {
                        'similarity': np.array(_raw.get('s', None)),
                        'proximity': np.array(_raw.get('d', None)),
                        'n_objects': int(_raw['n'][0, 0]),
                        'labels': np.array(_raw['labs'], dtype=str),
                        'sigma': float(_raw.get('sigma_emp', np.array([[np.nan]]))[0, 0]),
                    }
                    _this_filepath = basepath.joinpath(f'{_this_name}.pkz')
                    joblib.dump(_this_dict, _this_filepath, compress=6)
                if name == _this_name:
                    data_dict = _this_dict
        os.remove(archive_path)
    else:
        data_dict = joblib.load(filepath)

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'similarity_matrix.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(**data_dict,
                 DESCR=fdescr)
