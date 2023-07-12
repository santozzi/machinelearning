# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:19:15 2023

@author: SERGIO
"""
import numpy as np

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    
    def __init__(self,url):
        self.saludo = "Hello world"
        self.url = url
    
               
        
    #Importar dataset    
    def import_dataset(self):
        self.dataset = pd.read_csv(self.url)
        return self.dataset

    # filas desde : hasta, columnas desde : hasta -1 es la ultima
    def rango_variable_independiente_all_rows(self,a,b):
        self.X = self.dataset.iloc[:,a:b].values
        return self.X
    
    def rango_variable_dependiente_all_rows(self,a,b):
   
        self.Y = self.dataset.iloc[:,a:b].values
        return self.Y

    def limpiar_na_med(self,a,b):
        #na 
        #axis = 0 para seleccionar la media de la columna
        #axis = 1 para seleccionar la media de la fila

        imputer = SimpleImputer(missing_values = np.nan, strategy =  "mean")
        #reemplazo todos los nan por el promedio de la columna
        imputer = imputer.fit(self.X[:,a:b])
        #sobreEscribo los valores de X con los nuevos datos
        self.X[:,a:b] = imputer.transform(self.X[:,a:b])
    #Codificar datos categoricos
    def codificar_categoricos_x(self,col):
        le_X = preprocessing.LabelEncoder()
        self.X[:,col] = le_X.fit_transform(self.X[:,col])
     
    
    def codificar_categoricos_y(self,col):
        le_Y = preprocessing.LabelEncoder()
        self.Y[:,col] = le_Y.fit_transform(self.Y[:,col])
        
    #Codificar datos que no tienen un orden (nominales)
    def codificar_datos_nominales_x(self,col):
        ct = ColumnTransformer(
            [('one_hot_encoder', OneHotEncoder(categories='auto'), [col])],   
            remainder='passthrough')

        self.X = np.array(ct.fit_transform(self.X), dtype=np.float64)
    
    def codificar_datos_nominales_y(self,col):
        ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories='auto'), [col])],   
        remainder='passthrough')

        self.Y = np.array(ct.fit_transform(self.Y), dtype=np.float64)
   
    #Dividir el data set en conjunto de entrenamiento y conjunto de testing    
    def conjunto_de_entrenamiento(self,test_size,random_state):
        #La función train_test_split ya no forma parte de sklearn.cross_validation, ahora debe cargarse desde el paquete 
        #sklearn.model_selection
        self.X_train, self.X_test , self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=test_size, random_state = random_state)
        #Escalado de variables
    
    def escalar_x(self):
        sc_X = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test =sc_X.transform(self.X_test)
    
    def escalar_y(self):
        sc_Y = StandardScaler()
        self.Y_train = sc_Y.fit_transform(self.Y_train)
        self.Y_test =sc_Y.transform(self.Y_test)
