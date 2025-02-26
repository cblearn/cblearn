import enum


class NoiseTarget(enum.Enum):
    POINTS = "points"
    DIFFERENCES = "differences"


class Distance(enum.Enum):
    EUCLIDEAN = "euclidean"
    PRECOMPUTED = "precomputed"
