import numpy as np
from numpy import linalg as LA
import random

class LinearRegression:
    def fit(self, _X, _y):
        X =  np.array(_X)
        y =  np.array(_y)
        xTx = np.dot(X.transpose(), X)
        inverse = np.linalg.inv(xTx)
        self.w = np.dot(np.dot( inverse, X.transpose()), y)
    
    def predict(self, _x):
        return [np.sign(np.dot(self.w, xn)) for xn in _x]
    
    def get_w(self):
        return self.w

    

class LogisticRegression:
    def __init__(self, eta=0.1, tmax=1000, bs=1000000, lam = 0):
        self.eta = eta
        self.tmax = tmax
        self.batch_size = bs
        self.lam = lam

    # Infere o vetor w da funçao hipotese
    #Executa a minimizao do erro de entropia cruzada pelo algoritmo gradiente de descida
    def fit(self, _X, _y):
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
    
        
    #funcao hipotese inferida pela regressa logistica  
    def predict_prob(self, X):
        return [(1/(1+ np.exp(-(self.w.dot(x))))) for x in X]

    #Predicao por classificação linear
    def predict(self, X):
        pred = self.predict_prob(X)
        
        return [1 if i >= 0.5 else -1 for i in pred]

    def getW(self):
        return self.w

    def getRegressionY(self, regressionX, shift=0):
        return (-self.w[0]+shift - self.w[1]*regressionX) / self.w[2]
    
# class LogisticRegression:
#     def __init__(self, eta=0.1, tmax=30, batch_size=32, lam=0):
#         self.eta = eta
#         self.tmax = tmax
#         self.batch_size = batch_size
#         self.lam = lam

#     # Infere o vetor w da funçao hipotese
#     #Executa a minimizao do erro de entropia cruzada pelo algoritmo gradiente de descida
#     def fit(self, _X, _y):
#         self.w = []    
#         X = np.array(_X, dtype=np.float128)
# #         X = np.concatenate((np.ones((len(_X),1), dtype=np.float128), np.array(_X, dtype=np.float128)), axis=1)
#         y = np.array(_y, dtype=np.float128)
        
#         d = X.shape[1]
#         N = X.shape[0]
#         w = np.zeros(d, dtype=np.float128)
    
        
#         for i in range(self.tmax):
#             vsoma = np.zeros(d, dtype=np.float128)

#             #Escolhendo o lote de entradas
#             if self.batch_size < N:
#                 indices = random.sample(range(N),self.batch_size)
#                 batchX = [X[index] for index in indices]
#                 batchY = [y[index] for index in indices]
#             else:
#                 batchX = X
#                 batchY = y

#             #computando o gradiente no ponto atual
#             for xn, yn in zip(batchX, batchY):
#                 vsoma += (yn * xn) / (1 + np.exp((yn * w).T @ xn))
        
#             gt = vsoma/self.batch_size 
#             if self.lam != 0:
#                 gt += 2*self.lam * w
            
#             #Condicao de parada: se ||deltaF|| < epsilon (0.0001)
#             if LA.norm(gt) < 0.0001 :
#                 break
#             w = w + (self.eta*gt)

#         self.w = w

#     #funcao hipotese inferida pela regressa logistica  
#     def predict_prob(self, X):
#         return [(1 / (1 + np.exp(-(self.w[0] + self.w[1:].T @ x)))) for x in X]

#     #Predicao por classificação linear
#     def predict(self, X):
#         return [1 if (1 / (1 + np.exp(-(self.w[0] + self.w[1:].T @ x)))) >= 0.5 
#                 else -1 for x in X]

#     def getW(self):
#         return self.w

#     def getRegressionY(self, regressionX, shift=0):
#         return (-self.w[0]+shift - self.w[1]*regressionX) / self.w[2]
    
class PocketPLA():
    def __init__(self, tmax = 1000) -> None:
        self.tmax = tmax
        
    def get_w(self):
        return self.w
    
    def set_w(self, w):
        self.w = w

    def constroiListaPCI(self, X: np.array, y: np.array, w: np.array) -> tuple:
        yPred = np.sign(X.dot(w))

        mask = (y != yPred)

        return X[mask], y[mask]

    def fit(self, _X, y):
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

                    
    def getOriginalY(self, originalX):
        return (-self.w[0] - self.w[1]*originalX) / self.w[2]
    
    def predict(self, X):
        return np.sign(X.dot(self.w))
    
    def errorIN(self, X, y):
        error = 0
        for i in range(len(y)):
            if(np.sign(np.dot(self.w, X[i])) != y[i]):
                error += 1
                
        return error
   