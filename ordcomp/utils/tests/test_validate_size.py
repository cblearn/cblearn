import pytest

import ordcomp.utils


def test_check_size():
    assert 6 == ordcomp.utils._validate_size.check_size(None, 6)
    assert 3 == ordcomp.utils._validate_size.check_size(3, 6)
    assert 3 == ordcomp.utils._validate_size.check_size(3., 6)
    assert 3 == ordcomp.utils._validate_size.check_size(.5, 6)
    assert 15 == ordcomp.utils._validate_size.check_size(15, 6)

    with pytest.raises(ValueError):
        ordcomp.utils._validate_size.check_size(-1, 6)
