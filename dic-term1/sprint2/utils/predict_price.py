import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

def predict_the_price(X, y, model="linearregression"):
    # トレーニングデータとテストデータに分割
    (X_train, X_test,
        y_train, y_test) = train_test_split(X, y, test_size=0.25)

    #標準化
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    parameter = {"fit_intercept": [True, False]}

    grid_search = GridSearchCV(
        LinearRegression(),
        param_grid=parameter,
        cv=10
    )
    grid_search.fit(X_train, y_train)
    y_predict_gridsearched = grid_search.best_estimator_.predict(X_test)

    return y_predict_gridsearched