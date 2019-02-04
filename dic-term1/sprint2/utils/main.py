from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
from pandas import DataFrame



def main(X, y, model):
    if isinstance(X, DataFrame):
        (X_train, X_test, y_train, y_test) = train_test_split(X.values, y.values, test_size=0.25)
    else:
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if model == 'LogisticRegression':
        diparameter = {
        "C": [10**i for i in range(-2,4)],
        "random_state": [123],
        "class_weight": ['balanced', None],
    }

        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid=diparameter,
            cv=10,
            )
        grid_search.fit(X_train, y_train)
        y_predict_gridsearched = grid_search.best_estimator_.predict(X_test)

    elif model == 'DecisionTreeClassifier':
        diparameter = {
        'max_depth': list(range(1, 20)),
        'criterion': ['gini', 'entropy'],
        }

        grid_search = GridSearchCV(
            DecisionTreeClassifier(),
            param_grid=diparameter,
            cv=10
            )
        grid_search.fit(X_train, y_train)
        y_predict_gridsearched = grid_search.best_estimator_.predict(X_test)

    elif model == 'SVC':
        diparameter = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
        ]

        grid_search = GridSearchCV(
            SVC(),
            param_grid=diparameter,
            cv=10
            )
        grid_search.fit(X_train, y_train)
        y_predict_gridsearched = grid_search.best_estimator_.predict(X_test)


    return y_predict_gridsearched