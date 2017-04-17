"""
Module for loading breast cancer data set.
"""

from __future__ import print_function
import csv
import numpy


def load_data_set():
    """
    Load the UCI breast cancer data set.
    (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
    """
    data_set_file_name = "wdbc.csv"
    with open(data_set_file_name, "r") as data_set_file:
        # csv.reader returns an iterator which must be
        # first converted into a list
        data_set_list = list(csv.reader(data_set_file))

    print(type(data_set_list))

    data_set = numpy.asarray(data_set_list)
    # always examine data before starting!
    # (data may have silently loaded in the wrong format, or truncated)
    # use (tuple,) to form single tuple from original tuple
    print("Shape of data set is:\n%s" % (data_set.shape,))
    print("First row of data set is:\n%s" % data_set[0, :])
    # second column is benign/malignant classification
    classification_data = data_set[:, 1]
    # ignore first two columns (sample ID and classification)
    # to get feature data
    feature_data = data_set[:, 2:].astype(float)
    print("Shape of feature data is:\n%s" %
        (feature_data.shape,))
    print("Shape of classification data is:\n%s\n" %
        (classification_data.shape,))

    return feature_data, classification_data
