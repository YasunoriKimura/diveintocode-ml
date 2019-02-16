import numpy as np
import matplotlib.pyplot as plt


class ScratchLogisticRegression:
    def __init__(self, num_iter=300, lam=0.001, lr=0.01,bias=False, coef=False, verbose=False):
        self.iter = num_iter
        self.lam = lam
        self.bias = bias
        self.coef = coef
        self.lr = lr
        self.verbose = verbose
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.train_loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def _sig(self, z):
        return 1 / (1+np.exp(-z))
    
    def _logstic_hypothesis(self, X):
        return self._sig(np.dot(X, self.coef))
    
    def _J_func(self, X, y):
        m = len(y)
        error_content1 = -y*np.log(self._logstic_hypothesis(X))
        error_content2 = - (1-y)*np.log(1-self._logstic_hypothesis(X))
        errors =  error_content1 + error_content2
        J = (1/m)*np.sum(errors) + (self.lam/2*m)*sum(self.coef**2)
        return J
    
    def _gradient_descent(self, X, y, X_val, y_val):
        m = len(y)
        for i in range(self.iter):
            error = self._logstic_hypothesis(X) - y
            common_part = (1/m) * np.dot(X.T, error)
            self.coef[0] -= self.lr*(common_part[0])
            self.coef[1:] -= self.lr*(common_part[1:] + (self.lam/m) * self.coef[1:])
            self.train_loss[i] = self._J_func(X, y)
            self.val_loss[i] = self._J_func(X_val, y_val)
    
    def fit(self, X, y, X_val=None, y_val=None):
        X_train = np.insert(X, 0, 1, axis=1)
        y_train = y
        X_val = np.insert(X_val, 0, 1, axis=1)
        
        self.coef = np.reshape(np.random.randn(X_train.shape[1]), (X_train.shape[1],1))
        self._gradient_descent(X_train, y_train, X_val, y_val)
        
    def predict(self, X_val):
        X_val = np.insert(X_val, 0, 1, axis=1)
        y_pred = np.where(self._sig(np.dot(X_val, self.coef))>=0.5, 1,0).flatten()
        return y_pred
    
    def predict_proba(self, X_val, ):
        X_val = np.insert(X_val, 0, 1, axis=1)
        predicted = self._sig(np.dot(X_val, self.coef))
        return np.hstack((1-predicted,predicted))

    def plot(self):
        """
        算出された損失を可視化する関数
        """
        #プロット
        plt.xlabel('iter', fontsize = 16)
        plt.ylabel('loss', fontsize = 16)
        plt.plot(range(self.iter), self.train_loss, label='train_loss')
        plt.plot(range(self.iter), self.val_loss, label='val_loss')
        plt.legend()
    
    def scatter(self, X, y):
        """
        任意の特徴量をscatterと決定曲線を引く関数
        """
        plt.scatter(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], label='1', c='red')
        plt.scatter(X[np.where(y==0)[0], 0], X[np.where(y==0)[0], 1], label='0', c='blue')
        x = range(-3, 3, 1)
        plt.plot(x, -(self.coef[0] + self.coef[1] * x) / self.coef[2])
        plt.legend()