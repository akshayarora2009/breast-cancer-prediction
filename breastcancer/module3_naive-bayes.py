"""
Demonstrate Naive Bayes classifier.
"""

from __future__ import print_function
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import wdbc
from sklearn.naive_bayes import GaussianNB


def test_naive_bayes(feature_data, classification_data_numerical):
    """
    Demonstrate usage and accuracy of Naive Bayes classifier.
    """
    feature_data_train, feature_data_test, \
    classification_data_train, classification_data_test = \
        train_test_split(feature_data, classification_data_numerical)

    nbmodel = GaussianNB().fit(feature_data_train, classification_data_train)

    predicted_classification = nbmodel.predict(feature_data_test)
    print(
        "Validation metrics for Naive Bayes classifier are:\n",
        metrics.classification_report(
            classification_data_test, predicted_classification
        )
    )
    print(
        "Accuracy is: %.1f%%" %
        (metrics.accuracy_score(
            classification_data_test, predicted_classification
        ) * 100)
    )


def main():
    """
    Main function of the script.
    """

    feature_data, classification_data = wdbc.load_data_set()

    # scikit-learn functions require classification in terms of numerical
    # values (i.e. 0, 1, 2) instead of strings (e.g. 'benign', 'malignant')
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(classification_data)
    classification_data_numerical = label_encoder.transform(classification_data)

    test_naive_bayes(feature_data, classification_data_numerical)


main()
