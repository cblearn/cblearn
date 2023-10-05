# Changelog

## In development

- Refactor core functions for validating comparison data types #35. 
  These functions are now available from the base module `cblearn`.
- SOE, GNMDS, and MLDS estimators support quadruplet comparison data.
- API of embedding estimators is compatible to sklearn's binary classifiers.
- Refactoring of embedding estimators to share more functionality.

- TODO: Core functions for generating comparison data w/ flexible sampling and noise models #27 #24
- TODO: Add kNN sampling # 25
- TODO: Core functions for triplet and quadruplet decision making with flexible distance metrics #27 #34
- TODO: Flexible distance metrics for embedding estimators #34
- TODO: Deprecate validation functions in `utils` module
- TODO: Deprecate sampling and decision functions in `datasets` module
- TODO: Update documentation of comparison data
## 0.1.0

1- Support python 3.9 and 3.10.
- Introduce semantic versioning
- Publish to PyPI