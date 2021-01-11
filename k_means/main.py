# K Means

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 2:-1].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans as model

wcss = []

for i in range(1, 11):
    kmeans = model(n_clusters=i,random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11),wcss)
plt.show()
