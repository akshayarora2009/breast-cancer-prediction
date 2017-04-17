#!/usr/bin/env python

"""
Plot distribution and correlation of feature data from breast cancer data set.
"""

from __future__ import print_function
import numpy
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import wdbc

def plot_classification_frequency(classification_data):
    """
    Plot the frequency of the benign/malignant classifications
    using bar charts.
    """
    classification_frequency = scipy.stats.itemfreq(classification_data)

    plt.figure(1)

    for row_n, row in enumerate(classification_frequency):
        if row[0] == 'B':
            label = 'Benign'
            color = 'b'
        elif row[0] == 'M':
            label = 'Malignant'
            color = 'r'
        else:
            raise Exception("Unkown classification:", row[0])
        frequency = int(row[1])
        plt.bar(left=row_n, height=frequency, color=color, label=label)

    plt.gca().axes.xaxis.set_ticklabels([])
    plt.legend()
    plt.xlabel("Diagnosis")
    plt.ylabel("Frequency")
    plt.title("Distribution of Classifications")
    print(
        "In order to have our classifier be adept at spotting all classes,\n"
        "we must ensure our data has a reasonably equal distribution.\n"
    )
    plt.show()

def plot_feature_correlation(feature_data):
    """
    Plot correlation between features.
    """
    # rowvar=0: specify that the first dimension of the matrix (the rows)
    # represent the different cases, and look for correlation between features
    correlation_matrix = numpy.corrcoef(feature_data, rowvar=0)
    print("Correlation matrix: ")
    print(correlation_matrix)
    plt.figure(2)
    plt.title("Feature Correlation")
    colormap = plt.cm.Reds
    plt.gca().pcolor(correlation_matrix, cmap=colormap)
    plt.xlabel("Feature 1 index")
    plt.ylabel("Feature 2 index")

    print(
        "Ideally we want to pick features for input to a classifier that have\n"
        "a minimum amount of correlation. The idea is that the data should be\n"
        "as distinct as possible to enable maximum separation.\n"
        "In this plot, the darker the square, the stronger the correlation.\n"
    )

    plt.show()

def plot_feature_distribution(feature_data, classification_data):
    """
    Plot distribution of feature data to show clustering for different
    classifications.
    """

    # features are measurements of each cell nucleus: (from wdbc.names)
    #  1) radius
    #  2) texture
    #  3) perimeter
    #  4) area
    #  5) smoothness
    #  6) compactness
    #  7) concavity
    #  8) concave points
    #  9) symmetry
    # 10) fractal dimension

    # compare distributions for first 5 features
    n_features = 5
    feature_range = range(n_features)

    # sizes are in inches
    fig = plt.figure(figsize=(4*n_features, 2.5*n_features))
    fig.suptitle("Distribution of breast cancer cell data", fontsize=20)

    n_graph_rows = n_graph_cols = n_features
    graph_n = 1

    for feature_idx_1 in feature_range:
        for feature_idx_2 in feature_range:
            plt.subplot(n_graph_rows, n_graph_cols, graph_n)
            graph_n += 1

            feature_1_data = feature_data[:, feature_idx_1]
            feature_2_data = feature_data[:, feature_idx_2]

            print("feature_1_data: ")
            print(type(feature_1_data))
            if feature_idx_1 == feature_idx_2:
                # uses numpy ndarray-specific boolean indexing
                benign_data = feature_1_data[classification_data == 'B']
                malignant_data = feature_1_data[classification_data == 'M']

                bins = numpy.linspace(
                    min(feature_1_data), max(feature_2_data), 30)

                # draw distribution histograms of the two classifications
                plt.hist(benign_data, bins=bins, alpha=0.4, color='b')
                plt.hist(malignant_data, bins=bins, alpha=0.4, color='r')
                plt.xlabel("Feature %d" % (feature_idx_1+1))
                plt.ylabel("Frequency")
            else:
                # blue for benign, red for malignant
                colour_map = {'B': 'b', 'M': 'r'}
                colours = [0]*len(classification_data)
                for i, classification in enumerate(classification_data):
                    colours[i] = colour_map[classification]
                plt.scatter(
                    x=feature_1_data, y=feature_2_data,
                    c=colours, alpha=0.4
                )
                plt.xlabel("Feature %d" % (feature_idx_1+1))
                plt.ylabel("Feature %d" % (feature_idx_2+1))

    # fix subplot sizes so that everything fits
    plt.tight_layout()
    # tight_layout() doesn't take account of figure title,
    # move the top of the subplots down to make room
    plt.subplots_adjust(top=0.90)
    plt.legend(
        ['Benign', 'Malignant'],
        # use the whole of the figure for the location coordinates
        bbox_transform=plt.gcf().transFigure,
        # place the legend at the top right
        bbox_to_anchor=(0.98, 0.98)
    )
    plt.show()

def main():
    """
    Main function of the script.
    """
    feature_data, classification_data = wdbc.load_data_set()
    plot_classification_frequency(classification_data)
    plot_feature_correlation(feature_data)
    plot_feature_distribution(feature_data, classification_data)

main()
