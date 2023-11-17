# Changelog

## In development

- Refactor core functions for validating comparison data types #35. 
  These functions are now available from the base module `cblearn`.
- SOE, GNMDS, and MLDS estimators support quadruplet comparison data.
- API of embedding estimators is compatible to sklearn's binary classifiers.
- Refactoring of embedding estimators to share more functionality.
- Fix order of object ids in the things odd-one-out testset.
- Add precomputed embedding and preview images to the things odd-out-out dataset.
- Add images to the cars most-central dataset.
- Optionally return the original query index in `triplets_from_oddoneout` and `triplets_from_mostcentral`

- TODO: Core functions for generating comparison data w/ flexible sampling and noise models #27 #24
- TODO: Add kNN sampling # 25
- TODO: Core functions for triplet and quadruplet decision making with flexible distance metrics #27 #34
- TODO: Flexible distance metrics for embedding estimators #34
- TODO: Deprecate validation functions in `utils` module
- TODO: Update documentation of comparison data

## 0.1.0

- Support python 3.9 and 3.10.
- Introduce semantic versioning
- Publish to PyPI