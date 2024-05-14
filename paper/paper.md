---
title: 'cblearn: Comparison-based Machine Learning in Python'
tags:
  - Python
  - Machine Learning
  - Comparison-based Learning
  - Ordinal Embedding
  - Triplets
  - Behaviour
  - Clustering
  - Psychology
  - Psychophysics
  - Scaling
authors:
  - name: David-Elias Künstle
    orcid: 0000-0001-5507-3731
    corresponding: true
    affiliation: "1, 2"
  - given-names: Ulrike
    dropping-particle: von
    surname: Luxburg
    affiliation: "1, 2"
affiliations:
 - name: University of Tübingen, Germany
   index: 1
 - name: Tübingen AI Center, Germany
   index: 2
date: 22 September 2023
bibliography: references.bib
---

# Summary

The `cblearn` package implements comparison-based machine learning algorithms
and routines to process comparison-based data in Python.
Comparison-based learning algorithms are used when only comparisons of similarity between data points are available, but no explicit similarity scores or features.
For example, humans struggle to assign *numeric* similarities to apples, pears, and bananas.
Still, they can easily *compare* the similarity of pears and apples with the similarity of apples and bananas---
pears and apples usually appear more similar.
There exist comparison-based algorithms for most machine learning tasks,
like clustering, regression, or classification [e.g., @balcan2016learning; @heikinheimo2013crowd; @perrot_near-optimal_2020];
The most frequently applied algorithms, however, are the so-called ordinal embedding algorithms
[e.g., @agarwal_generalized_2007; @tamuz_adaptively_2011; @van_der_maaten_stochastic_2012; @terada_local_2014;
@amid2015; @anderton2019scaling; @ghosh2019landmark].
Ordinal embedding algorithms estimate a metric representation, such that the distances between embedded objects reflect the similarity comparisons.
These embedding algorithms have recently come into fashion in psychology and cognitive science to objectively quantify the perceived similarity of various stimuli
[e.g., @haghiri_estimation_2020; @wills_toward_2009; @roads_obtaining_2019].



# Statement of need

This work presents `cblearn`, an open-source Python package for comparison-based learning. In contrast to related packages,
`cblearn` provides not just a specific algorithm but an ecosystem for comparison-based data with access to multiple real-world datasets and a collection of algorithm implementations.
`cblearn` is fast and user-friendly for applications but flexible for research on new algorithms and methods.
The package integrates well into the scientific Python ecosystem; for example, third-party functions for cross-validation or hyperparameter tuning of `scikit-learn` estimators can typically be used with `cblearn` estimators.
Although our package is relatively new, it has already been used for algorithm development [@mandal2023revenue] and data analysis in several studies [@schonmann_using_2022; @kunstle_estimating_2022; @van_assen_identifying_2022; @zhao2023perceiving; @sauer2023seeing; @huber2024tracing].

We designed `cblearn` as a modular package with functions
for processing and converting the comparison data in all its varieties (`cblearn.preprocessing`, `cblearn.utils`, `cblearn.metrics`), routines to generate artificial or load real-world datasets (`cblearn.datasets`), and algorithms for ordinal embedding and clustering (`cblearn.embedding`, `cblearn.cluster`).

## Various data formats supported

The atomic datum in comparison-based learning is the quadruplet,
a comparison of the similarity $\delta$ between two pairs $(i, j)$ and $(k, l)$,
for example, asserting that $\delta(i, j) < \delta(k, l)$.
Alternative comparisons like the triplet, where $i == l$, can be reduced to one or more quadruplets.
Comparison-based learning algorithms estimate classes, clusters, or metrics to fulfill as many quadruplets as possible.
In ordinal embedding, for example, the problem is to find  $x_i, x_j, x_k, x_l \in \mathbb{R}^d$
s.t. $\left\lVert x_i - x_j \right\rVert_2 < \left\lVert x_k - x_l \right\rVert_2  \Leftrightarrow \delta(i, j) < \delta(k, l)$.

Besides triplets and quadruplets, there are many ways to ask for comparisons.
Some tasks ask for the "odd-one-out", the "most-central" object, or the two most similar objects to a reference. `cblearn` can load these different queries and convert them to triplets, ready for subsequent embedding or clustering tasks.

Different data types can store triplets, and  `cblearn` converts them internally.
A 2D array with three columns for the object indices $(i, j, k)$ stores a triplet per row. In some applications, it is comfortable to separate the comparison "question" and "response", which leads to an additional list of labels $1$, if $\delta(i, j) \le \delta(i, k)$, and $-1$, if $\delta(i, j) > \delta(i, k)$.
An alternative format stores triplets as a 3-dimensional sparse array.
These sparse arrays convert fast back and forth to dense 2D arrays while providing an intuitive comparison representation via multidimensional indexing. For example, the identical triplet can be represented as `[[i, j, k]]`, `([[i, k, j]], [-1])` or `sparse_arr[i, j, k] == 1`.


## Interfaces to diverse datasets

There is no Iris, CIFAR, or ImageNet in comparison-based learning---the community lacks accessible real-world datasets to evaluate new algorithms.
`cblearn` provides access to various real-world datasets, summarized in \autoref{fig:datasets}, with functions to download and load the comparisons.
These datasets---typically comparisons between images or words---consist of human responses.
Additionally, our package provides preprocessing functions to convert different comparisons to triplets or quadruplets, which many algorithms expect.

![Real-world datasets that can be accessed with `cblearn` cover many object and triplet numbers. Please find a detailed description and references to the dataset authors in our package documentation. \label{fig:datasets}](images/datasets.pdf){ width=35% }

## Algorithms implemented for CPU and GPU

In the current version `0.2.0`, `cblearn` implements an extensive palette of ordinal embedding algorithms and a clustering algorithm (\autoref{tablealgorithms}); additional algorithms can be contributed easily within the modular design.
Most algorithm implementations are built with the scientific ecosystem around `scipy` [@virtanenSciPyFundamentalAlgorithms2020;@harris_array_2020] to be fast and lightweight. Inspired by the work of @vankadara_insights_2020, we added GPU implementations with `torch` [@paszke2019pytorch;@anselPyTorchFasterMachine2024] that use automatic differentiation and stochastic optimization routines known from deep learning methods.
These GPU implementations can be used with large datasets and rapidly adapted thanks to automated differentiation.

: Algorithm implementations in `cblearn`. Most of these come in multiple variants: Different backends for small datasets on CPU or large datasets on GPU, or varied objective functions. \label{tablealgorithms}

|Algorithm | Reference |
|:---------|:---------------|
|Crowd Kernel Learning | @tamuz_adaptively_2011 |
|Fast Ordinal Triplet Embedding | @jain_finite_2016 |
|Generalized Non-metric MDS | @agarwal_generalized_2007 |
|Maximum-likelihood Difference Scaling | @maloney_maximum_2003 |
|Soft Ordinal Embedding  | @terada_local_2014 |
|Ordinal Embedding Neural Network | @vankadara_insights_2020 |
|Stochastic Triplet Embedding | @van_der_maaten_stochastic_2012 |
|ComparisonHC (clustering) | @perrot_near-optimal_2020 |


## User-friendly and compatible API
One of Python's greatest strengths is the scientific ecosystem, into which `cblearn` integrates. Our package does not only make use of this ecosystem internally but adopts their API conventions–––every user of `scikit-learn` [@pedregosa_scikit-learn_2011;@buitinck_api_2013] is already familiar with the API  of `cblearn`:
Estimator objects use the well-known `scikit-learn` methods `.fit(X, y)`,`.transform(X)`, and `.predict(X)`. This convention allows the use of many routines from the `scikit-learn` ecosystem with `cblearn`'s estimators while representing comparisons as `numpy` arrays [@harris_array_2020].
Interested readers can find a code example in the [Supplementary Material](https://github.com/cblearn/cblearn/blob/main/paper/supplementary.pdf), which shows in just four lines how to fetch a real-world dataset, preprocess the data, estimate an embedding, and cross-validate the fit. More examples are available in the package's documentation.

# Related work and empirical comparison

Most comparison-based learning algorithms were implemented independently as part of a research paper [e.g., @ghoshdastidar_foundations_2019; @hebartRevealingMultidimensionalMental2020; @van_der_maaten_stochastic_2012; @roads_obtaining_2019];
Just a few of these implementations, for example `loe` [@terada_local_2014] or `psiz` [@roads_obtaining_2019], come in the form of software packages.

Related packages with collections of comparison-based learning algorithms  have a focus on metric learning [`metric-learn` with a high compatibility to scikit-learn, @metric-learn] and crowd-sourced data collection, using active ordinal embedding algorithms [e.g., `NEXT`, @NIPS2015_89ae0fe2 and `salmon`, @Sievert2023]. Our package `cblearn`, on the other hand, focuses on providing comparison data and interoperable estimator implementations of the remaining areas of comparison-based learning. 

A small empirical comparison to third-party packages reveals that `cblearn`'s algorithm implementations
typically are accurate and fast. Details are described in [Supplementary Material](https://github.com/cblearn/cblearn/blob/main/paper/supplementary.pdf).
A more comprehensive evaluation of various ordinal embedding algorithms per se, focusing on large data sets, can be found in @vankadara_insights_2020.

# Acknowledgements
We want to thank Debarghya Ghoshdastidar, Leena Vankadara, Siavash Haghiri, Michael Lohaus, and especially Michaël Perrot for the inspiring discussions about comparison-based learning in general and the `cblearn` package in particular.
Additionally, we thank Thomas Klein for the helpful feedback on this manuscript and Alexander Conzelmann for the contributions to the `cblearn.cluster` module.
The paper, code and documentation profited considerably from the feedback of the JOSS reviewers and the editor.

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC number 2064/1 – Project number 390727645 and supported by the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039A.
The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting David-Elias Künstle.

# References
