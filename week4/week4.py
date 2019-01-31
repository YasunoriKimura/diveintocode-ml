import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


df_train = pd.read_csv('application_train.csv')
df_X = df_train.drop("TARGET", axis=1)
df_Y = df_train[["TARGET"]]

#カテゴリーを数値変換
categorical_feats = [
    f for f in df_X.columns if df_X[f].dtype == 'object'
]
categorical_feats_ = categorical_feats.copy()

for f in categorical_feats_:
    df_X[f], _ = pd.factorize(df_X[f])
    df_X[f] = df_X[f].astype('int')

#残りの欠損値を平均で埋める
df_X = df_X.fillna(df_X.mean())

#データ分割
(X_train, X_test,
     y_train, y_test) = train_test_split(df_X.values, df_Y.values.flatten(), test_size=0.3)

#標準化
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#KNeighbors
from sklearn.neighbors import KNeighborsClassifier

#パラメータn_neighbors=3の場合で学習
neigh_5nn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)

neigh_5nn.fit(X_train, y_train)

y_predict_neigh_5nn = neigh_5nn.predict(X_test)

estimation_neigh_5nn_dict = {
    "accuracy": accuracy_score(y_test, y_predict_neigh_5nn),
    "precision": precision_score(y_test, y_predict_neigh_5nn),
    "recall": recall_score(y_test, y_predict_neigh_5nn) ,
    "f1":f1_score(y_test, y_predict_neigh_5nn)
}
estimation_neigh_5nn = pd.DataFrame(estimation_neigh_5nn_dict, index=['KNeighborsClassifier'])

estimation_neigh_5nn