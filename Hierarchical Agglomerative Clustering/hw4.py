import csv
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np

def load_data(filepath):
    # Initialize an empty list that store the rows from csv file
    data = []
    # read in the file in the filepath
    with open(filepath, 'r') as file:
        # initialize DictReader object that read the file by each row asa dictionary
        reader = csv.DictReader(file)
        for row in reader:
            # convert to the OrderedDict row and add to the data list
            data.append(dict(row))
    # return a list of dictionaries
    return data

def calc_features(row):

    # create a list 'features' that contains 6 features for one country
    features = [
        # x1 = 'Population' of the country
        float(row['Population']),
        # x2 = ‘Net migration’ of the country
        float(row['Net migration']),
        # x3 = ‘GDP ($ per capita)’ of the country
        float(row['GDP ($ per capita)']),
        # x4 = ‘Literacy (%)’ of the country
        float(row['Literacy (%)']),
        # x5 = ‘Phones (per 1000)’ of the country
        float(row['Phones (per 1000)']),
        # x6 = ‘Infant mortality (per 1000 births)’ of the country
        float(row['Infant mortality (per 1000 births)']),
    ]

    # convert the 'features' list to a Numpy array with dtype of float64
    returnVal =  np.array(features, dtype=('float64'))
    returnVal = np.reshape(returnVal, (6,))

    return returnVal


def hac(features):
    # determine the total number of countries (data points)
    n = len(features)
    # initialize an (n-1) x 4 array with zero matrix
    arrayZ = np.zeros((n-1,4))
    # create a distance matrix 
    distance_matrix = np.zeros((n, n))


    # store the data of calculated Euclidean distance between the countries at first
    # data and second data in distance_matrix
    for i in range(n): # the country at first index
        for j in range(i+1, n): # the next country to calculate with the distance with first country
            distance_matrix[i][j] = np.linalg.norm(features[i]-features[j])
            # the distance between the first data and the second data
            # is same as the distance between the two data vice versa
            distance_matrix[j][i] = distance_matrix[i][j]

    # initialize its own cluster to each data (each country)
    clusters = {i: [i] for i in range(n)}

    # loop through the clusters 
    for iterateNum in range(n-1):
        # initialize a pair of minimum distances for clustering
        min_pair = (None, None)
        # intialize minimum distance to infinite 
        minDistance = np.inf

        # initialize the indices of clusters
        cluster_indices = list(clusters.keys())

        # loop through the clusters of the first data until we reach the last element
        for i in range(len(cluster_indices)-1):
            # loop through elements in the first cluster data
            for j in range(i+1, len(cluster_indices)):
                firstData = cluster_indices[i]
                secondData = cluster_indices[j]
                # loop through elements in the second cluster data
                distance = [distance_matrix[x][y] for x in clusters[firstData] for y in clusters[secondData]]
                # calculate distance between first and second cluster data
                maxDistance = max(distance)
                # update minimum distance when the calculated distance is smaller than previous minimum distance
                if maxDistance < minDistance:
                    minDistance = maxDistance
                    # cluster two elements with minimum distance
                    min_pair = (firstData, secondData)
                # tie-breaking rules
                # when the minimum distance is equal to the distance of two cluster data in the loop
                elif maxDistance == minDistance:
                    # and the first cluster data is smaller than the first cluster data in the pair of current minimum distance
                    # or the index of first cluster data equals first element in the minimum distance pair
                    if firstData < min_pair[0] or (firstData == min_pair[0] and secondData < min_pair[1]):
                        # update the minimum distance pair with this first and second cluster data
                        min_pair = (firstData, secondData)          

        # merge the two cluster in which its distance is minimum
        cluster_merged = clusters[min_pair[0]] + clusters[min_pair[1]]

        # add the new merged cluster data to the last + 1 index of the cluster indices list
        clusters[max(cluster_indices) + 1] = cluster_merged

        # assign the smaller cluster index to array Z[i,0] and the other cluster index to array Z[i,1] 
        # arrayZ[iterate, 0], arrayZ[iterate, 1] = min_pair[0], min_pair[1]
        arrayZ[iterateNum, 0], arrayZ[iterateNum, 1] = min_pair
        arrayZ[iterateNum,2] = minDistance
        arrayZ[iterateNum,3] = len(cluster_merged)

        # remove the pair of two cluster from the clusters (that stores all the country as a cluster)
        del clusters[min_pair[0]]
        del clusters[min_pair[1]]
    # return the matrix with that is clustered using HAC
    return arrayZ


def fig_hac(Z, names):
    # initialize the figure for the plot
    fig = plt.figure()
    # use dendrogram to set from the Z matrix, set the leaf_rotation to 90 degrees
    # for vertical labels
    dendrogram(Z, labels = names, leaf_rotation=90)
    # call tight_layout() of the figure so the labels and titles fit inside the figure
    fig.tight_layout()
    # display the plot in the figure
    plt.show()
    # return the figure
    return fig

def normalize_features(features):
    # Convert the list of feature vectors into NumPy array
    array_f = np.array(features)

    # Calculate the column's mean and column's standard deviation
    mean = np.mean(array_f, axis=0)
    standard_dev = np.std(array_f, axis=0)
    # calculate normalized feature value
    normalized_f = (array_f - mean)/standard_dev
    # converts the array of narmalized feature into list of narmalized feature vectors
    normalized_l = [row for row in normalized_f]
    # return the list of narmalized feature vectors
    return normalized_l








