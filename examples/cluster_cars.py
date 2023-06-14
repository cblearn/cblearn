# %% [markdown]
# Hierarchical clustering
# =======================
#
# Clustering of data points (e.g., stimuli) according to triplet responses can be done in two ways: 1. Estimate and ordinal embedding and apply (metric) clustering methods implemented in scikit-learn. 2. Apply clustering directly to the triplet responses. The second approach is implemented here using hierarchical clustering. Directly clustering from the triplet responses has the advantage that the clustering is done based on the responses without compressing (and potentially losing) some information with an embedding of a fixed dimensionality.
#
# In this example, we cluster 60 images of cars based on human responses, judging the "most central" of three.
# First, we load the responses and convert them to triplets. Then we hierarchically cluster the triplets with the ComparisonHC algorithm and visualize the results.
# Afterwards, we estimate a 2d embedding and visualize both "ground truth" labels and the cluster labels.
#
# This example is inspired by an example in the `ComparisonHC documentation <https://github.com/mperrot/ComparisonHC/blob/master/examples/car.ipynb>`_ and thus shows the API differences in the implementations.
#

# %%
from cblearn.cluster import ComparisonHC
from cblearn.embedding import TSTE
from cblearn.datasets import fetch_car_similarity
from cblearn.preprocessing import triplets_from_mostcentral
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# %%
data = fetch_car_similarity()
triplets = triplets_from_mostcentral(data.triplet, data.response)

# %%
chc = ComparisonHC(4).fit(triplets)

# %%
dendrogram(chc.dendrogram_, labels=[f'{cls} ({i})' for i, cls in enumerate(data.class_name[data.class_id])], color_threshold=57)
plt.show()

# %%
ste = TSTE(2).fit(triplets)

# %%
for i in range(4):
    coords = ste.embedding_[data.class_id == i]
    plt.scatter(coords[:, 0], coords[:, 1], c=f'C{i}', label=data.class_name[i])
plt.legend()
plt.show()

for i in range(4):
    coords = ste.embedding_[chc.labels_ == i]
    plt.scatter(coords[:, 0], coords[:, 1], c=f'C{i}', label=f'Cluster {i+1}')
for i in range(len(ste.embedding_)):
    plt.text(ste.embedding_[i, 0], ste.embedding_[i, 1], i)
plt.legend()
plt.show()
