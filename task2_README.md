
# Customer Segmentation using K-means Clustering

## Project Overview
This project implements the K-means clustering algorithm to segment customers based on their annual income and spending score. By clustering customers, we can better understand different purchasing behaviors and create targeted marketing strategies.

## Dataset
The dataset used is `Mall_Customers.csv`. It contains the following key features:
- **Annual Income (k$)**: Annual income of the customer.
- **Spending Score (1-100)**: Score assigned based on customer spending behavior.

## Installation
Install the required libraries using:

```bash
pip install pandas matplotlib scikit-learn
```

## Usage
Run the script using:

```bash
python customer_segmentation.py
```

## Methodology
1. Load and preprocess the dataset.
2. Select `Annual Income` and `Spending Score` for clustering.
3. Apply the **K-means** algorithm with 5 clusters.
4. Visualize clusters using a scatter plot.

## Results
The final plot shows distinct customer segments based on their spending patterns.
