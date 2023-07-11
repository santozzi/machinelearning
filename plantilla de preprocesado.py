# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar dataset
datos = "Datos/Data.csv"
dataset = pd.read_csv(datos)
# filas desde : hasta, columnas desde : hasta -1 es la ultima
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#na 
#axis = 0 para seleccionar la media de la columna
#axis = 1 para seleccionar la media de la fila
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy =  "mean")
#reemplazo todos los nan por el promedio de la columna
imputer = imputer.fit(X[:,1:3])
#sobreEscribo los valores de X con los nuevos datos
X[:,1:3] = imputer.transform(X[:,1:3])


#Codificar datos categoricos
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
le_Y = preprocessing.LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
y = le_Y.fit_transform(y)

#Codificar datos que no tienen un orden
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float64)

#Dividir el data set en conjunto de entrenamiento y conjunto de testing

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
#La función train_test_split ya no forma parte de sklearn.cross_validation, ahora debe cargarse desde el paquete 
#sklearn.model_selection
X_train, X_test , Y_train, Y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)

