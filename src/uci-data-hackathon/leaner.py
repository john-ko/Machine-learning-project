import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

just_points = np.genfromtxt('just_points.csv', delimiter=",")

random_state = 170
# Incorrect number of clusters
y_pred = KMeans(n_clusters=5, random_state=random_state).fit_predict(just_points)

plt.subplot(221)
plt.scatter(just_points[:, 0], just_points[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")
plt.show()



print("finished")