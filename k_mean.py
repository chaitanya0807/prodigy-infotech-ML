import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Inspect the dataset
print(df.head())

# Step 1: Select the features for clustering
# Use Annual Income and Spending Score for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 2: Create the K-means model and fit it
kmeans = KMeans(n_clusters=5, random_state=42)  # Choose the number of clusters
kmeans.fit(X)

# Step 3: Add cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Step 4: Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-means Clustering')
plt.show()
