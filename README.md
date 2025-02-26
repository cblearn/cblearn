<h1 align="center">
<img src="https://raw.githubusercontent.com/cblearn/cblearn/main/docs/logo-light.png" width="300">
</h1><br>

## Comparison-based Machine Learning in Python
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06139/status.svg)](https://doi.org/10.21105/joss.06139)
[![PyPI version](https://img.shields.io/pypi/v/cblearn.svg)](https://pypi.python.org/pypi/cblearn)
[![Documentation](https://readthedocs.org/projects/cblearn/badge/?version=stable)](https://cblearn.readthedocs.io/en/stable/?badge=stable)
[![Test status](https://github.com/cblearn/cblearn/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/cblearn/cblearn/actions/workflows/test.yml)
[![Test Coverage](https://codecov.io/gh/cblearn/cblearn/branch/master/graph/badge.svg?token=P9JRT6OK6O)](https://codecov.io/gh/cblearn/cblearn)

Comparison-based learning methods are machine learning algorithms using similarity comparisons ("A and B are more similar than C and D") instead of featurized data. 


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from cblearn.datasets import make_random_triplets
from cblearn.embedding import SOE

X = load_iris().data
triplets = make_random_triplets(X, result_format="list-order", size=2000)

estimator = SOE(n_components=2)
# Measure the fit with scikit-learn's cross-validation
scores = cross_val_score(estimator, triplets, cv=5)
print(f"The 5-fold CV triplet error is {sum(scores) / len(scores)}.")

# Estimate the scale on all triplets
embedding = estimator.fit_transform(triplets)
print(f"The embedding has shape {embedding.shape}.")
```

## Getting Started

* [Installation & Quickstart](https://cblearn.readthedocs.io/en/stable/getting_started/index.html)
* [Examples](https://cblearn.readthedocs.io/en/stable/generated_examples/index.html).
* [User Guide](https://cblearn.readthedocs.io/en/stable/user_guide/index.html).


## Contribute

We are happy about your bug reports, questions or suggestions as Github Issues and code or documentation contributions as Github Pull Requests. 
Please see our [Contributor Guide](https://cblearn.readthedocs.io/en/stable/contributor_guide/index.html). 

## Related packages

There are more Python packages for comparison-based learning:

- [metric-learn](http://contrib.scikit-learn.org/metric-learn) is a collection of algorithms for metric learning. The *weakly supervised* algorithms learn from triplets and quadruplets.
- [salmon](https://docs.stsievert.com/salmon/) is a package for efficiently collecting triplets in crowd-sourced experiments. The package implements ordinal embedding algorithms and sampling strategies to query the most informative comparisons actively.

## Authors and Acknowledgement
*cblearn* was initiated by current and former members of the [Theory of Machine Learning group](http://www.tml.cs.uni-tuebingen.de/index.php) of Prof. Dr. Ulrike von Luxburg at the University of Tübingen.
The leading developer is [David-Elias Künstle](http://www.tml.cs.uni-tuebingen.de/team/kuenstle/index.php).

We want to thank all the contributors here on GitHub.
This work has been supported by the Machine Learning Cluster of Excellence, funded by EXC number 2064/1 – Project number 390727645. The authors would like to thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting David-Elias Künstle. 

## License

This library is free to use, share, and adapt under the [MIT License](https://github.com/cblearn/cblearn/blob/master/LICENSE) conditions.

## Citation

Please cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.06139#) if you publish work using `cblearn`:

**Künstle et al., (2024). cblearn: Comparison-based Machine Learning in Python. Journal of Open Source Software, 9(98), 6139, https://doi.org/10.21105/joss.06139**

```
@article{Künstle2024, 
    doi = {10.21105/joss.06139}, 
    url = {https://doi.org/10.21105/joss.06139}, 
    year = {2024}, 
    publisher = {The Open Journal}, 
    volume = {9}, number = {98}, pages = {6139}, 
    author = {David-Elias Künstle and Ulrike von Luxburg}, 
    title = {cblearn: Comparison-based Machine Learning in Python}, 
    journal = {Journal of Open Source Software} 
} 
```
