# cblearn
## Comparison-based Machine Learning in Python
[![PyPI version](https://img.shields.io/pypi/v/cblearn.svg)](https://pypi.python.org/pypi/cblearn)
[![Documentation](https://readthedocs.org/projects/cblearn/badge/?version=stable)](https://cblearn.readthedocs.io/en/stable/?badge=stable)
[![Test status](https://github.com/cblearn/cblearn/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/cblearn/cblearn/actions/workflows/test.yml)
[![Test Coverage](https://codecov.io/gh/cblearn/cblearn/branch/master/graph/badge.svg?token=P9JRT6OK6O)](https://codecov.io/gh/cblearn/cblearn)

Comparison-based Learning algorithms are the Machine Learning algorithms to use when training data contains similarity comparisons ("A and B are more similar than C and D") instead of data points. 

Triplet comparisons from human observers help model the perceived similarity of objects.
These human triplets are collected in studies, asking questions like 
"Which of the following bands is most similar to Queen?" or 
"Which color appears most similar to the reference?".

This library provides an easy-to-use interface for comparison-based learning algorithms.
It plays hand-in-hand with scikit-learn:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from cblearn.datasets import make_random_triplets
from cblearn.embedding import SOE
from cblearn.metrics import QueryScorer

X = load_iris().data
triplets = make_random_triplets(X, result_format="list-order", size=1000)

estimator = SOE(n_components=2)
# Measure the fit with scikit-learn's cross-validation
scores = cross_val_score(estimator, triplets, cv=5)
print(f"The 5-fold CV triplet error is {sum(scores) / len(scores)}.")

# Estimate the scale on all triplets
embedding = estimator.fit_transform(triplets)
print(f"The embedding has shape {embedding.shape}.")
```

Please try the [Examples](https://cblearn.readthedocs.io/en/stable/generated_examples/index.html).

## Getting Started

Install cblearn as described [here](https://cblearn.readthedocs.io/en/stable/install.html) and try the [examples](https://cblearn.readthedocs.io/en/stable/generated_examples/index.html).

Find a theoretical introduction to comparison-based learning, the datatypes, 
algorithms, and datasets in the [User Guide](https://cblearn.readthedocs.io/en/stable/user_guide/index.html).

## Features

### Datasets

*cblearn* provides utility methods to simplify the loading and conversion
of your comparison datasets. In addition, some functions download and load multiple real-world comparisons.

| Dataset  | Query | #Object | #Response | #Triplet |
| --- | --- | ---:| ---:| ---:|
| Vogue Cover | Odd-out Triplet | 60 | 1,107 | 2,214 | 
| Nature Scene | Odd-out Triplet | 120 | 3,355 | 6,710 | 
| Car | Most-Central Triplet | 60 | 7,097 | 14,194 | 
| Material | Standard Triplet | 100 | 104,692 |104,692 | 
| Food | Standard Triplet | 100 | 190,376 |190,376 | 
| Musician | Standard Triplet | 413 | 224,792 |224,792 | 
| Things Image Testset | Odd-out Triplet | 1,854 | 146,012 | 292,024 | 
| ImageNet Images v0.1 | Rank 2 from 8 | 1,000 | 25,273 | 328,549 | 
| ImageNet Images v0.2 | Rank 2 from 8 | 50,000 | 384,277 | 5M | 


### Embedding Algorithms

| Algorithm                   | Default | Pytorch (GPU) | Reference Wrapper |
| --------------------------- |  :---:  | :-----------: | :---------------: |
| Crowd Kernel Learning (CKL) | X       | X             |                   |
| FORTE                       |         | X             |                   |
| GNMDS                       | X       | X             |                   |
| Maximum-Likelihood Difference Scaling (MLDS) | X |              | [MLDS (R)](https://cran.r-project.org/web/packages/MLDS/index.html)|
| Soft Ordinal Embedding (SOE) | X      | X             | [loe (R)](https://cran.r-project.org/web/packages/loe/index.html) |
| Stochastic Triplet Embedding (STE/t-STE) | X       | X  |   |

## Contribute

We are happy about your bug reports, questions or suggestions as Github Issues and code or documentation contributions as Github Pull Requests. 
Please see our [Contributor Guide](https://cblearn.readthedocs.io/en/stable/contributor_guide/index.html). 

## Authors and Acknowledgement
*cblearn* was initiated by current and former members of the [Theory of Machine Learning group](http://www.tml.cs.uni-tuebingen.de/index.php) of Prof. Dr. Ulrike von Luxburg at the University of Tübingen.
The leading developer is [David-Elias Künstle](http://www.tml.cs.uni-tuebingen.de/team/kuenstle/index.php).

We want to thank all the contributors here on GitHub.
This work has been supported by the Machine Learning Cluster of Excellence, funded by EXC number 2064/1 – Project number 390727645. The authors would like to thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting David-Elias Künstle. 

## License

This library is free under the [MIT License](https://github.com/cblearn/cblearn/blob/master/LICENSE) conditions.
Please cite this library appropriately if it contributes to your scientific publication. We would also appreciate a short email (optionally) to see how our library is being used. 
