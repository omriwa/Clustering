# Hierarchical Clustering

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

plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.show()

# using the optimal number of clusters
kmeans = model(n_clusters=5, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualizing the prediction
import random

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c="red", label="C1")
plt.scatter(X[y_pred == 1,0], X[y_pred == 1,1],s=100, c="blue" , label="C2")
plt.scatter(X[y_pred == 2,0], X[y_pred == 2,1],s=100, c="green" , label="C3")
plt.scatter(X[y_pred == 3,0], X[y_pred == 3,1],s=100, c="yellow" , label="C4")
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s=100, c="purple", label="C5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="black", label="Centroids")
plt.legend()
plt.show()


