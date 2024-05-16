---
title: |
       | Supplementary Material
       | `cblearn`: Comparison-based Machine Learning in Python*
author:
  - |
    | David-Elias Künstle and Ulrike von Luxburg
    | University of Tübingen and Tübingen AI Center, Germany
date: 22 September 2023
---

# Empirical evaluation

We generated embeddings of comparison-based datasets to measure runtime and triplet error as a small empirical evaluation of our ordinal embedding implementations.
We compared various CPU and GPU implementations in `cblearn` with third-party implementations in *R* [`loe` @terada_local_2014], and *MATLAB* [@van_der_maaten_stochastic_2012].
In contrast to synthetic benchmarks [e.g., @vankadara_insights_2020], we used the real-world datasets
that can be accessed through `cblearn`, converted to triplets. The embeddings were arbitrarily chosen to be 2D.
Every algorithm runs once per dataset on a compute node (8 CPU cores; 96GB RAM; NVIDIA RTX 2080ti) with a run-time limit of 24 hours. Some runs did fail by exceeding those constraints: our FORTE implementation failed by an out of memory error on the `imagenet-v2` dataset. The *MATLAB* implementation of tSTE timed out on `things` and `imagenet-v2` datasets. The *R* implementation of SOE on the `imagenet-v2` dataset by an "unsupported long vector" error, caused by the large size of the requested embedding.

The benchmarking scripts and results are publicly available in a separate repository[^1].

[^1]: [https://github.com/cblearn/cblearn-benchmark](https://github.com/cblearn/cblearn-benchmark)

## Is there a "best" estimator?


Comparing the ordinal embedding estimators in `cblearn`, SOE, CKL, GNMDS, and tSTE were performing about equally well in both runtime and accuracy (\autoref{fig:performance-per-algorithm_cblearn}).
The GPU implementations are slower on the tested datasets and for SOE and GNMDS noticeably less accurate.

![\label{fig:deltaerror-per-algorithm_cblearn-all}](./images/deltaerror-per-algorithm_cblearn-all.pdf){width=45%}
![\label{fig:deltatime-per-algorithm_cblearn-all}](images/deltatime-per-algorithm_cblearn-all.pdf){width=45%}
\begin{figure}
\caption{The triplet error and runtime per estimator and dataset, relative to the mean error or the fastest run. Thin lines show runs on the different datasets; the thick lines indicate the respective median. With the exception of STE, all CPU algorithms are able to embed the triplets similarly well. There are just minor differences in the runtime of the CPU implementations. The GPU implementations are usually significantly slower on the data sets used.  
} 
    \label{fig:performance-per-algorithm_cblearn}
\end{figure}

## When should GPU implementations be preferred?

In terms of accuracy and runtime, our GPU implementations using the `torch` backend could not outperform the CPU pendants using the `scipy` backend on the tested datasets. However, \autoref{fig:performance-per-algorithm_cblearn} shows the GPU runtime grows slower with the number of triplets, such that they potentially outperform CPU implementations with large datasets of $10^7$ triplets and more. In some cases, the `torch` implementations show the overall best accuracy.

![The runtime increases almost linearly with the number of triplets. However, GPU implementations have a flatter slope and thus can compensate for the initial time overhead on large datasets.
    \label{fig:time-per-triplets_gpu}](images/time-per-triplets_gpu.pdf){width=50%}

There are various explanations for the speed disadvantage of our pytorch implementations. On the one hand, it may be due to the overhead of converting between numpy and pytorch and calculating the gradient (AutoGrad). On the other hand, it can also be due to the optimizer or the selected hyperparameters. 
To get a first impression of these factors, we have built minimal examples of the CKL algorithm [@tamuz_adaptively_2011] and estimated 2D embeddings of the Vogue Cover dataset [@heikinheimo2013crowd]. \autoref{fig:torch-speedtest} shows the runtimes and triplet accuracies on a standard laptop. The small markers show runs with different initialization and the bold markers the respective median performance. The CKL implementation of `cblearn` is approximately three times slower than the minimal version, probably due to data validation and conversion overheads. If the gradient is not provided directly but calculated automatically with PyTorch's AutoGrad functions, the minimal example runs ~9 times slower. The most severe impact has changing the optimization algorithm to stochastic optimization (*Adam*, lr=10).  However, it can be assumed in accordance with the results in previous sections, that this overhead is compensated with increasing dataset size. 

![The runtime and error for different optimization methods in minimal CKL implementations. `cblearn`'s CKL implementation is shown for reference.
\label{fig:torch-speedtest}](images/torch_speedtest_triplets.pdf){width=50%}

An additional challenge of stochastic optimizers like *Adam* [@kingma2014adam] is their sensitivity to hyperparameter choices. This sensitivity is demonstrated in \autoref{fig:adam_lr}, where the learning rate of Adam is varied for the toy example. Especially runtime largely depends on the learning rate, while the error is less sensitive to it. Likewise, the performance of `torch` ordinal embedding implementations could be improved by using more sophisticated tuning of optimizer parameters.


![The runtime and error for different learning rates of the Adam optimizer in a minimal example with CKL estimating a 2D embedding of 60 objects.
\label{fig:adam_lr}](images/adam_lr_triplet.pdf){width=50%}

Besides all discussions about runtime and accuracy, the `torch` backend provides benefits for maintainance and exension of the library. It uses PyTorch's automatic differentiation [@paszke2019pytorch], so that the loss gradient does not have to be explicitly defined and new algorithms can be implemented very quickly. 

## How does `cblearn` compare to other implementations?

In a small comparison, our implementations run multiple times faster with approximately the same accuracy as reference implementations (\autoref{fig:performance-per-algorithm_library}).
We compared our CPU implementations of SOE the corresponding reference implementations in R , `loe` [@terada_local_2014], and our implementation of CKL, GNMDS, STE, tSTE with the *MATLAB* of @van_der_maaten_stochastic_2012.
This comparison is not exhaustive, but it shows that our implementations are competitive with the reference implementations in terms of accuracy and runtime. Of course, we cannot separate the factors of algorithm implementation and runtime environment. 

![\label{fig:deltaerror-per-algorithm_library}](./images/deltaerror-per-algorithm_library.pdf){width=45%}
![\label{fig:deltatime-per-algorithm_library}](images/deltatime-per-algorithm_library.pdf){width=45%}
\begin{figure}[!ht]
    \caption{The triplet error and runtime per estimator and dataset, relative to the mean error and the fastest run. Thin lines show runs on the different datasets; the thick lines indicate the respective median. The triplet error is approximately similar for all implementations but STE. For all algorithms, `cblearn` provides the fastest implementation.}
    \label{fig:performance-per-algorithm_library}
\end{figure}


# Code example\label{sec:code-example}

```Python
from cblearn import datasets, preprocessing, embedding
from sklearn.model_selection import cross_val_score
import seaborn as sns; sns.set_theme("poster", "whitegrid")

cars = datasets.fetch_car_similarity()
triplets = preprocessing.triplets_from_mostcentral(cars.triplet, cars.response)
accuracy = cross_val_score(embedding.SOE(n_components=2), triplets, cv=5).mean()
embedding = embedding.SOE(n_components=2).fit_transform(triplets)
fg = sns.relplot(x=embedding[:, 0], y=embedding[:, 1],
        hue=cars.class_name[cars.class_id])
fg.set(title=f"accuracy={accuracy:.2f}", xticklabels=[], yticklabels=[])
fg.tight_layout(); fg.savefig("images/car_example.pdf")
```
![](images/car_example.pdf){width=75%}

# References
