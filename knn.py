"""
knn.py
~~~~~~~~~~

Module is for building a classic K-Nearest Neighbors Algorithm

The KNN algorithm assumes that similar things exist in close proximity.
The algorithm is simple does not contain much mathematical tasks.
"""

import math
from collections import Counter


class KNN:
    """K-Nearest Neighbors class"""

    def __init__(self, k=None, data=None, query=None):
        """
        Constructor for the KNN. Takes the k of the network
        :param k: int, the no of neighbors to be selected
        :param data: list, the total data to be processed
        :param query: int, the value(age) to be classified
        """
        self.k = k
        self.data = data
        self.query = query

    def nearest_neighbors(self):
        """
           Main function, calculates the Euclidean distance between the query and a sample of the data, sorts it\
            and picks the k no of data points and finally returns the mode in the selected data points
           :return: list of selected data,mode, mode of the selected labels
           """
        neighbor_distances_and_indices = []
        for idx, data_point in enumerate(self.data):
            distance = self.euclidean_dis(data_point[:-1], self.query)  # Calculate the distance between the query
            # example and the current example from the data.

            neighbor_distances_and_indices.append((distance, idx))  # Add the distance and the index of the example
            # to an ordered collection

        sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices, key=lambda x: x[0])  #
        # Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by
        # the distances

        k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:self.k]  # Pick the first K
        # entries from the sorted collection

        k_nearest_labels = [self.data[i][1] for distance, i in k_nearest_distances_and_indices]  # Get the labels of
        # the selected K entries

        return k_nearest_labels, self.mode(k_nearest_labels)

    @staticmethod
    def mode(labels):
        return Counter(labels).most_common(1)[0][0]

    @staticmethod
    def euclidean_dis(point1, point2):
        sum_squared_distance = 0
        for i in range(len(point1)):
            sum_squared_distance += math.pow(point1[i] - point2[i], 2)
        return math.sqrt(sum_squared_distance)


if __name__ == '__main__':
    '''
    # Classification Data
    # 
    # Column 0: age
    # Column 1: likes pineapple
    '''
    clf_data = [
        [22, 1],
        [23, 1],
        [21, 1],
        [18, 1],
        [19, 1],
        [25, 0],
        [27, 0],
        [29, 0],
        [31, 0],
        [45, 0],
    ]
    # Question:
    # Given the data we have, does a 33 year old like pineapples on their pizza?
    clf_query = [33]
    knn = KNN(k=3, data=clf_data, query=clf_query)
    clf_k_nearest_neighbors, clf_prediction = knn.nearest_neighbors()
