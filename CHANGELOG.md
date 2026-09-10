# Changelog

## Upcoming

## 0.4

- Feature: `embedding.LORE`, a low-rank ordinal embedding that estimates the intrinsic dimensionality
- Feature: Support for Python 3.12 and 3.13
- Improvement: Compatibility with scikit-learn 1.6 and newer, which replaced the estimator tag dictionary
  by the `__sklearn_tags__` API. scikit-learn 1.6 is now the minimum supported version.
- Improvement: Readable errors for invalid embedding dimensions, raised in `fit` instead of `__init__`,
  as required by scikit-learn
- Improvement: Validation of `test_dimensions` in `embedding.estimate_dimensionality_cv`
- Improvement: The food, nature and vogue datasets are downloaded from OSF mirrors,
  since the original hosts no longer serve the archives
- Improvement: Externally hosted archives are pinned to fixed files or commits,
  so that their checksums no longer change on every upstream push
- Fix: Gradient of the STE embedding, which weighted each triplet by P * (1 - P) instead of (1 - P)
- Fix: `estimate_dimensionality_cv` returns the estimated dimension as a Python int
- Fix: `utils.torch_device` returns and validates an explicit device instead of None
- Fix: Sparse input is detected with `scipy.sparse.issparse`, so that the newer sparray classes are recognized
- Fix: The numpy converter for rpy2 is registered directly, since `numpy2ri.activate` raises in rpy2 3.5.12 and newer
- Others: Extended unit tests and coverage, seeded triplet sampling in the test suite

## 0.3

- Feature: JOSS paper
- Feature: Quickstart guide in documentation
- Feature: Data point sampling from manifolds.
- Improvement: Extended documentation
- Improvement: cblearn logo and new style in documentation
- Improvement: Filter invalid responses in datasets
- Improvement: Full compatibility to sklearn estimator tests
  
## 0.2

- Improvement: Extended documentation
- Feature: `embedding.estimate_dimensionality_cv` function (Künstle et al., 2022)
- Fix: Avoid numpy deprecation warning for scalar variables in `fetch_similarity_matrix`
- Fix: Various errors in the examples
- Fix: Minor errors in the unit tests
- Others: Updated dependencies

## 0.1
### 0.1.2

- support python 3.11
- update core dependencies
  
### 0.1.1

- Minor fixes in the documentation.
- Adapt loading of food and imagenet dataset to solve problems caused by changes in externally hosted files
  
### 0.1.0

- Support python 3.9 and 3.10.
- Introduce semantic versioning
- Publish to PyPI
