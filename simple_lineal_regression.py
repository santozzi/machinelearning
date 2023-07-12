# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:43:20 2023

@author: SERGIO
"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
class SimpleLinearRegression:
   #crea modelo de regresion linear
    def __init__(self,X_train,Y_train,title,label_X,label_Y):
       self.regression = LinearRegression()
       self.regression.fit(X_train,Y_train)
       self.X_train = X_train
       self.Y_train = Y_train
       self.label_X = label_X
       self.label_Y = label_Y
       self.title=title
       
    def y_pred(self,X_test):
       self.ypred = self.regression.predict(X_test)
       
    #visualizar lo resultados de entrenamiento
    def graph(self):
        plt.scatter(self.X_train, self.Y_train, color ="red")
        plt.plot(self.X_train,self.regression.predict(self.X_train),color ="blue")
        plt.title(self.title)
        plt.xlabel(self.label_X)
        plt.ylabel(self.label_Y)
        plt.show()
    
