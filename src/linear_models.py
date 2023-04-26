import numpy as np
import pandas as pd
from numpy import linalg as LA
from .classifier_all_utils import *
from .constants import ORDER
import random

class LinearRegression:
    def fit(self, _X, _y) -> None:
        X =  np.array(_X)
        y =  np.array(_y)
        xTx = np.dot(X.transpose(), X)
        inverse = np.linalg.inv(xTx)
        self.w = np.dot(np.dot( inverse, X.transpose()), y)
    
    def predict(self, _x)  -> list:
        return [np.sign(np.dot(self.w, xn)) for xn in _x]
    
    def get_w(self) -> list:
        return self.w
    
    def set_w(self, w) -> None:
        self.w = w

class LogisticRegression:
    def __init__(self, eta=0.1, tmax=1000, bs=1000000, lam = 0) -> None:
        self.eta = eta
        self.tmax = tmax
        self.batch_size = bs
        self.lam = lam

    def fit(self, _X, _y) -> None:
        X = np.array(_X)
        y = np.array(_y).reshape(-1, 1)

        N = X.shape[0]
        d = X.shape[1]

        w = np.zeros(d)
        eps = self.eta

        for _ in range(self.tmax):
            if self.batch_size < N:
                ids = random.sample(range(N),self.batch_size)
                batchX = [X[index] for index in ids]
                batchY = [y[index] for index in ids]
            else:
                batchX = X
                batchY = y

            gt = -(1/N) * sum([xi*yi / (1+np.exp(yi*w.dot(xi))) for xi, yi in zip(batchX, batchY)])
            
            if self.lam != 0:
                gt += 2*self.lam * w
                
            if LA.norm(gt) < eps:
                break
                
            w -= self.eta*gt

        self.w = w
    
    def predict_prob(self, X) -> list:
        return [(1/(1+ np.exp(-(self.w.dot(x))))) for x in X]

    def predict(self, X) -> list:
        pred = self.predict_prob(X)
        
        return [1 if i >= 0.5 else -1 for i in pred]

    def get_w(self) -> np.array:
        return self.w
    
    def set_w(self, w) -> None:
        self.w = w

class PocketPLA():
    def __init__(self, tmax = 1000) -> None:
        self.tmax = tmax

    def constroiListaPCI(self, X: np.array, y: np.array, w: np.array) -> tuple:
        yPred = np.sign(X.dot(w))
        mask = (y != yPred)

        return X[mask], y[mask]

    def fit(self, _X, y) -> None:
        X = np.array(_X)
        y = np.array(y)
        self.w = np.zeros(_X.shape[1])

        XPCI = X.copy()
        yPCI = y.copy()

        best_w = self.w
        best_error = len(y)

        for _ in range(self.tmax):  
            if len(XPCI) == 0:
                break

            i = np.random.randint(len(XPCI))

            self.w = self.w + XPCI[i]*yPCI[i]

            XPCI, yPCI = self.constroiListaPCI(X, y, self.w)

            error = len(XPCI)
            if error < best_error:
                best_error = error
                best_w = self.w

        self.w = best_w
    
    def predict(self, X) -> np.array:
        return np.sign(X.dot(self.w))
    
    def get_w(self) -> np.array:
        return self.w
    
    def set_w(self, w) -> None:
        self.w = w

class OneVsAll:
    def __init__(self, model: str, order=ORDER, **kwargs) -> None:
        self.order = order
        
        if model == "lin":
            self.model = LinearRegression()
        elif model == "log":
            self.model = LogisticRegression(eta=kwargs["eta"], 
                                               bs=kwargs["bs"], 
                                               tmax=kwargs["tmax"],
                                               lam=kwargs["lam"]
                                              )                                               
        else:
            self.model = PocketPLA(tmax=kwargs["tmax"])      
    
    def fit(self, train: pd.DataFrame) -> None:
        weigths = []
        for i in self.order[:-1]:
            train["tmp_label"] = train["label"].map(lambda x: 1 if x == i else -1)

            X_train = train[["intensidade", "simetria"]].values
            y_train = train["tmp_label"].values
            X_train = np.c_[np.ones(X_train.shape[0]), X_train]

            self.model.fit(X_train, y_train)

            weigths.append(self.model.get_w())

            train = remove_class(train, i)
        
        self.weigths = weigths
    
    def predict_one(self, img: np.array) -> np.array:
        for i, m in enumerate(self.order[:-1]):
            self.model.set_w(self.weigths[i]) 
            p_ = np.array(self.model.predict(np.array([img])))
            
            if p_ == 1:
                return m
            else:
                if m == self.order[-2]:
                    return self.order[-1]
                
    def get_w(self) -> list:
        return self.weigths
    