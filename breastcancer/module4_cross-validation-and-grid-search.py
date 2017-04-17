#!/usr/bin/env python

"""
Evaluate models using k-fold cross-validation, and find optimum
set of parameters using grid search.
"""

from __future__ import print_function
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import wdbc


def cross_validate(model, features_data, classification_data, n_folds):
    """
    Cross-validate the given model using n_folds folds.
    """
    scores = cross_validation.cross_val_score(
        model, features_data, classification_data,
        cv=n_folds
    )
    return {
        'mean': scores.mean(),
        'sd': scores.std()
    }

def plot_accuracy_vs_folds(feature_data, classification_data_numerical):
    """
    Plot a graph showing how the apparent accuracy of a model changes depending
    on how many folds are used for cross-validation.
    """

    # for reasons that weren't thoroughly explained, we deliberately split
    # the data before further splitting with cross-validation
    # (it might have something to do with the need to reserve some data
    #  for a final testing set?)
    feature_data_train, _, classification_data_train, _ = \
        train_test_split(feature_data, classification_data_numerical)

    nbayes = GaussianNB().fit(feature_data_train, classification_data_train)
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn15 = neighbors.KNeighborsClassifier(n_neighbors=15)
    models = [nbayes, knn3, knn15]
    model_names = ['Naive Bayes', '3 nearest neighbour', '15 nearest neighbour']

    mean_accuracies = {}
    fold_range = range(2, 21)

    for model, model_name in zip(models, model_names):
        for n_folds in fold_range:
            scores = cross_validate(
                model,
                feature_data_train, classification_data_train,
                n_folds
            )
            if not model_name in mean_accuracies.keys():
                mean_accuracies[model_name] = []
            mean_accuracies[model_name].append(scores['mean'])

    for model_name in model_names:
        plt.plot(fold_range, mean_accuracies[model_name], label=model_name)
    plt.legend(loc='best')
    plt.ylim(0.5, 1)
    plt.title("Mean Accuracy of Model for Different Numbers of Folds")
    print("For this data set and this set of models, the accuracy changes\n"
          "very little with differing numbers of folds.\n"
          "This indicates good generalisation of the models.")
    plt.show()

def optimise_knn_parameters(feature_data, classification_data_numerical):
    """
    Find the set of parameters for a k-nearest neighbour classifier that yields
    the best accuracy.
    """

    feature_data_train, _, classification_data_train, _ = \
        train_test_split(feature_data, classification_data_numerical)

    parameters = [{
        'n_neighbors': [1, 3, 5, 10, 50, 100],
        'weights': ['uniform', 'distance']
    }]
    n_folds = 10

    clf = GridSearchCV(
        neighbors.KNeighborsClassifier(), parameters, cv=n_folds,
        scoring="f1" # f1 = standard measure of model accuracy
    )
    clf.fit(feature_data_train, classification_data_train)

    print("\nThe grid search scores for a k-nearest neighbour classifier were:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%.1f (+-%0.03f s.d.) for %r" % (100*mean_score, scores.std()/2, params))

    print("The best parameter set found was:\n", clf.best_estimator_)


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

    plot_accuracy_vs_folds(feature_data, classification_data_numerical)
    #optimise_knn_parameters(feature_data, classification_data_numerical)

main()
