# K-Means Clustering

## Overview

This repository contains a Jupyter Notebook that demonstrates **K-Means clustering**, an **unsupervised learning algorithm** used to group data points into clusters based on similarity. The notebook focuses on understanding how K-Means works, how clusters are formed, and how to choose an appropriate number of clusters.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. What is K-Means Clustering  
4. K-Means Implementation  
5. Choosing the Number of Clusters (Elbow Method)  
6. Visualization of Clusters    

---

## Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

- `kMeans.ipynb` – Notebook demonstrating K-Means clustering and cluster visualization

---

## What is K-Means Clustering

K-Means is an unsupervised learning algorithm that:
- Groups data into *K* clusters
- Minimizes the distance between data points and their cluster centroids
- Works best with numeric, scaled features

Each data point is assigned to the nearest centroid based on distance.

---

## K-Means Implementation

The notebook applies K-Means using `scikit-learn`.

Basic commands used:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
```

---

## Choosing the Number of Clusters (Elbow Method)

The **Elbow Method** is used to determine the optimal number of clusters.

Key points:
- Plots number of clusters vs inertia
- Looks for a point where improvement slows down
- Helps avoid over- or under-clustering

Common commands:
```python
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
```

---

## Visualization of Clusters

The notebook includes visualization to understand clustering results.

Example:
```python
plt.scatter(X[:, 0], X[:, 1], c=labels)
```

Visualization helps interpret how data points are grouped in feature space.

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  

