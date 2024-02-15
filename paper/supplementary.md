---
title: |
       | Supplementary Material
       | *cblearn: Comparison-based Machine Learning in Python*
author:
  - |
    | David-Elias Künstle and Ulrike von Luxburg
    | University of Tübingen and Tübingen AI Center, Germany
date: 22 September 2023
---

# Empirical evaluation

We generated embeddings of comparison-based datasets to measure runtime and triplet accuracy as a small empirical evaluation of our ordinal embedding implementations.
We compared various CPU and GPU implementations in `cblearn` with third-party implementations in R [`loe` @terada_local_2014], and *MATLAB* [@van_der_maaten_stochastic_2012].
In contrast to synthetic benchmarks [e.g., @vankadara_insights_2020], we used the real-world datasets
that can be accessed through *cblearn*, converted to triplets.
Every algorithm runs once per dataset on a compute node (8-core of Intel\textregistered Xeon \textregistered Gold 6240; 96GB RAM; NVIDIA 2080ti); just a few runs did not yield results due to resource demanding implementations or bugs: our *FORTE* implementation exceeded the memory limit on the *imagenet-v2* dataset, the third-party implementation of *tSTE* timed out on *things* and *imagenet-v2* datasets. The third-party *SOE* implementation reached a limit on dataset size on *imagenet-v2*. Probably due to numerical issues, our *CKL-GPU* implementation did not crash but returned non-numerical values on the *musician* dataset.

The benchmarking scripts and results are publicly available in a separate repository[^1].

[^1]: [https://github.com/cblearn/cblearn-benchmark](https://github.com/cblearn/cblearn-benchmark)

## Is there a "best" estimator?


Comparing all ordinal embedding estimators in `cblearn`, *SOE* shows accurate and fast results; *CKL*, *GNMDS*, and *tSTE* were performing about equally well (\autoref{fig:performance-per-algorithm_cblearn}).
The GPU implementations are slower on the tested datasets, presumably because they had some initial computational overhead.

![\label{fig:deltaerror-per-algorithm_cblearn-all}](./images/deltaerror-per-algorithm_cblearn-all.pdf){width=45%}
![\label{fig:deltatime-per-algorithm_cblearn-all}](images/deltatime-per-algorithm_cblearn-all.pdf){width=45%}
\begin{figure}
\caption{The triplet error and runtime improvement per estimator. The improvement is relative to the average accuracy or runtime (at 0), calculated over all shown estimators. Compared to this average, a positive accuracy improvement means the algorithm can represent more triplets, while a positive runtime improvement means the algorithm is faster. Likewise, negative improvement indicates fewer triplets and lower speed, respectively.}
    \label{fig:performance-per-algorithm_cblearn}
\end{figure}

## When should GPU implementations be preferred?

In terms of accuracy and runtime, our GPU (or `pytorch`) implementations could not outperform the CPU (or `scipy`) pendants on the tested datasets. However, \autoref{fig:performance-per-algorithm_cblearn} shows the GPU runtime grows slower with the number of triplets, such that they potentially outperform CPU implementations with large datasets of $10^7$ triplets and more. In some cases, the GPU implementations show the overall best accuracy.
An additional advantage of GPU implementations is that they require no explicit gradient definition, which simplifies the implementation of new algorithms.

![The runtime increases almost linearly with the number of triplets. However, GPU implementations have a flatter slope and thus can compensate for the initial time overhead on large datasets.
    \label{fig:time-per-triplets_gpu}](images/time-per-triplets_gpu.pdf){width=50%}

There are various explanations for the speed disadvantage of our pytorch implementations. On the one hand, it may be due to the overhead of converting between numpy and pytorch and calculating the gradient (AutoGrad). On the other hand, it can also be due to the optimizer or the selected hyperparameters. 
To get a first impression of these factors, we have built a toy example, linear regression with 200 observations and 100 dimensions. The \autoref{fig:torch-speedtest} shows that the overhead of autograd and the stochastic optimization (Adam, lr=0.05) both slow down the optimization multiplicatively by factor ~8 in this example. However, it can be assumed, and in accordance with the above tendency, that this disadvantage decreases with increasing data set size. 

![The runtime and error for different optimization methods in a toy example.\label{fig:torch-speedtest}](
    images/torch_speedtest.pdf
){width=50%}

An additional disadvantage of stochastic optimizers like Adam is, that they are more sensitive to hyperparameter choices and thus require more tuning. This sensitivity is demonstrated in \autoref{fig:adam_lr}, where the learning rate of Adam is varied for the toy example. Especially runtime largely depends on the learning rate, while the error is less sensitive to it. Likewise, the performance of `pytorch` ordinal embedding implementations could be improved by using more sophisticated tuning of optimizer parameters.


![The runtime and error for different learning rates of the Adam optimizer in a toy example.\label{fig:adam_lr}](
    images/adam_lr.pdf
){width=50%}


## How does `cblearn` compare to other implementations?

In a small comparison, our implementations were $50-100\%$ faster with approximately the same accuracy as reference implementations (\autoref{fig:performance-per-algorithm_library}).
We compared our CPU implementations of *SOE*, *STE*
and *tSTE* with the corresponding reference implementations in R , `loe` [@terada_local_2014], and *MATLAB* [@van_der_maaten_stochastic_2012]. Additionally, the latter offers an *CKL* implementation to compare with.
Please note that despite our care, runtime comparisons of different interpreters offer room for some confounding effects.
The results shown should nevertheless indicate a trend.

![\label{fig:deltaerror-per-algorithm_library}](./images/deltaerror-per-algorithm_library.pdf){width=45%}
![\label{fig:deltatime-per-algorithm_library}](images/deltatime-per-algorithm_library.pdf){width=45%}
\begin{figure}[!ht]
    \caption{The triplet error and runtime improvement per estimator. Improvement
    is calculated against the average performance per dataset of all estimators in the plot.}
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