# ordcomp
Machine Learning Toolkit for Comparison-based Learning in Python

[![Unit tests](https://github.com/dekuenstle/ordcomp/workflows/Python%20package/badge.svg)](https://github.com/dekuenstle/ordcomp/actions)
[![Test Coverage](https://codecov.io/gh/dekuenstle/ordcomp/branch/master/graph/badge.svg?token=P9JRT6OK6O)](https://codecov.io/gh/dekuenstle/ordcomp)
[![Documentation](https://readthedocs.org/projects/ordcomp/badge/?version=latest)](https://ordcomp.readthedocs.io/en/latest/?badge=latest)

Comparison-based Learning are the Machine Learning algorithms to use, when training data
are ordinal comparisons instead of Euclidean points. 
Triplet comparisons can be gathered e.g. from human studies with questions like 
"Which of the following bands is most similar to Queen?".

This library provides an easy to use interface to comparison-based learning algorithms.
It plays hand-in-hand with scikit-learn:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from ordcomp.datasets import make_random_triplets
from ordcomp.embedding.wrapper import SOE
from ordcomp.metrics import TripletScorer


X = load_iris().data
triplets, responses = make_random_triplets(X, size=1000)

estimator = SOE(n_components=2)
scores = cross_val_score(estimator, triplets, responses, scoring=TripletScorer, cv=5)
print(f"The 5-fold CV triplet error is {sum(scores)/len(scores)}.")

embedding = estimator.fit_transform(triplets, responses)
print(f"The embedding has shape {embedding.shape}.")
```

Please try the [Examples](https://ordcomp.readthedocs.io/en/latest/generated_examples/index.html).

## Getting Started

OrdComp required Python 3.7 or newer. The easiest way to install is using `pip`:

```
pip install https://github.com/dekuenstle/ordcomp.git
```
Find more details in the [installation instructions](https://ordcomp.readthedocs.io/en/latest/install.html).


In the [User Guide](https://ordcomp.readthedocs.io/en/latest/user_guide/index.html) you find a detailed introduction.

## Contribute

We are happy about your contributions.
Please see our [Contributor Guide](https://ordcomp.readthedocs.io/en/latest/contributor_guide/index.html). 

## License

This library is free to use under the [LGPL-2.1 License](https://github.com/dekuenstle/ordcomp/blob/master/LICENSE) conditions.
