# In this project we'll try to use a k-means clustering model to divide data into several clusters

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def plot_data(dataset):
    # This function just plots the dataset, unclustered
    
    # Slice up the dataset
    x_axis = dataset.data[:, 0]  # Length
    y_axis = dataset.data[:, 2]  # Width

    # Plot the data
    plt.scatter(x_axis, y_axis)
    plt.show()


def plot_clustered_data(dataset):
    # We'll now separate/filter each cluster into it's own label and plot it using different colors
    filtered_label0 = dataset.data[label == 0]
    filtered_label1 = dataset.data[label == 1]
    filtered_label2 = dataset.data[label == 2]

    plt.scatter(filtered_label0[:, 0], filtered_label0[:, 2], color='blue')
    plt.scatter(filtered_label1[:, 0], filtered_label1[:, 2], color='orange')
    plt.scatter(filtered_label2[:, 0], filtered_label2[:, 2], color='green')
    plt.show()


def plot_clustered_data_with_center(dataset, labels, model):
    # This function plots all the clusters, and also shows their respective centers
    km_centers = model.cluster_centers_
    for i in labels:
        plt.scatter(dataset.data[label == i, 0], dataset.data[label == i, 2], label=i)
    plt.scatter(km_centers[:, 0], km_centers[:, 2], s=80, color='black')
    plt.legend()
    plt.show()


# Main
if __name__ == '__main__':
    # Let's load a default dataset with some simple data related to the iris (flower)
    data = datasets.load_iris()

    # Show the initial state of the dataset
    plot_data(data)

    # Let's now apply a K-Means clustering model to the dataset, we know the iris dataset has 3 classes (clusters)
    km_model = KMeans(n_clusters=3)

    # Predict the labels of the cluster and print them
    label = km_model.fit_predict(data.data)

    print("{}".format(label))

    # Plot the clustered data
    plot_clustered_data(data)

    # Get/show the center of each cluster
    u_labels = np.unique(label)
    plot_clustered_data_with_center(data, u_labels, km_model)

