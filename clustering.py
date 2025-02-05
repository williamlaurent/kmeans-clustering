import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Memuat data dari file CSV dan mengembalikan DataFrame.
    
    :param file_path: Path ke file CSV yang berisi data
    :return: DataFrame yang berisi data
    """
    data = pd.read_csv(file_path)
    return data

def perform_kmeans_clustering(data, n_clusters):
    """
    Melakukan K-Means Clustering pada data dan mengembalikan model K-Means.
    
    :param data: DataFrame yang berisi data
    :param n_clusters: Jumlah cluster yang diinginkan
    :return: Model K-Means yang telah dilatih
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    kmeans.fit(data)
    
    return kmeans

def plot_clusters(data, kmeans):
    """
    Menampilkan hasil clustering menggunakan scatter plot.
    
    :param data: DataFrame yang berisi data
    :param kmeans: Model K-Means yang telah dilatih
    """
    labels = kmeans.labels_
    
    centroids = kmeans.cluster_centers_
    
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', marker='o', label='Data Points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def main():
    file_path = 'data.csv'
    
    data = load_data(file_path)
    
    n_clusters = 3
    
    kmeans = perform_kmeans_clustering(data, n_clusters)
    
    plot_clusters(data, kmeans)

if __name__ == "__main__":
    main()
