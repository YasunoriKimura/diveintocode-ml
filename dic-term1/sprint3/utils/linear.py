import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter=300, lr=0.01, bias=False, verbose=False, coef=False):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.coef = coef
        self.verbose = verbose
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.train_loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)


    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          線形の仮定関数による推定結果

        """
        return np.dot(X, self.coef)


    def _compute_cost(self, X, y):
        """
        平均二乗誤差を計算する。MSEは共通の関数を作っておき呼び出す

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値

        Returns
        -------
          次の形のndarray, shape (1,)
          平均二乗誤差
        """
        y_pred = self._linear_hypothesis(X)
        return self._MSE(y_pred, y)

    def _MSE(self, y_pred, y):
        """
        平均二乗誤差の計算

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
          推定した値
        y : 次の形のndarray, shape (n_samples,)
          正解値

        Returns
        ----------
        mse : numpy.float
          平均二乗誤差
        """
        m = len(y)
        error = y_pred - y
        total_error = np.sum(error**2)
        J = total_error / (2*m)
        return J


    def _gradient_descent(self, X, y, X_val, y_val):
        """
        最急降下法でパラーメータを更新

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値
        loss :  　損失値

        Returns
        -------
        次の形のndarray, shape (1,)
        パラメータ

        """
        m = len(y)
        
        for i in range(self.iter):
            h  = self._linear_hypothesis(X)
            error = h - np.reshape(y, (len(y),1))
            self.coef  = self.coef - (self.lr/m)*np.dot(X.T, error)
            self.train_loss[i] = self._compute_cost(X, y)
            self.val_loss[i] = self._compute_cost(X_val, y_val)

    def fit(self, X, y, X_val, y_val):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        X = np.insert(X, 0, 1, axis=1)
        X_val = np.insert(X_val, 0, 1, axis=1)
        self.coef = np.reshape(np.random.randn(X.shape[1]), (X.shape[1],1))

        #訓練データを使ってパラメータを算出
        self._gradient_descent(X, y, X_val, y_val)

        if self.verbose:
            #verboseをTrueにした際は学習過程を出力
            print()



    def plot(self):
        """
        算出された損失を可視化する関数
        """
        plt.xlabel('iter', fontsize = 16)
        plt.ylabel('loss', fontsize = 16)
        plt.plot(range(self.iter), self.train_loss, label='train_loss')
        plt.plot(range(self.iter), self.val_loss, label='val_loss')
        plt.legend()


    def predict(self, X):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        pass
        return