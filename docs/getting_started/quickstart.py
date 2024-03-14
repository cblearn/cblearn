import numpy as np
from cblearn.datasets import make_random_triplets
from cblearn.embedding import SOE
from cblearn.metrics import procrustes_distance

points = np.random.rand(20, 2)
estimator = SOE(n_components=2)

print(f"Triplets | Error (SSE)\n{22 * '-'}")
for n in (25, 100, 400, 1600):
    triplets = make_random_triplets(points, size=n, result_format="list-order")
    embedding = estimator.fit_transform(triplets)
    error = procrustes_distance(points, embedding)
    print(f"    {len(triplets):4d} |       {error:.3f}")
