from typing import Union


def check_size(size: Union[None, int, float], max_objects: int) -> int:
    """ Convert size argument to the number of objects.

    Args:
        size: The ommited, relative, or absolute number of objects.
        max_objects: The maximum number of objects for relative size.

    Returns:
        The absolute size, corresponding to
            max_objects, if size is None
            size, if size is int
            size * max_objects, if size is float

    Raises
       ValueError:
           If size is int and < 0 or > max_objects
           If size is float and < 0 or > 1.
    """
    if size is None:
        return max_objects
    elif size < 0:
        raise ValueError(f'Expects size above 0, got {size}.')
    elif isinstance(size, int) or size > 1:
        return int(size)
    elif isinstance(size, float):
        return int(size * max_objects)
