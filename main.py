# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 20:08:43 2023

@author: SERGIO
"""
from preprocessing import *
from simple_lineal_regression import *

if __name__ == "__main__":
     pp = Preprocessing("Datos/Salary_Data.csv")
     dataset = pp.import_dataset()
     X = pp.rango_variable_independiente_all_rows(0,1)
     Y = pp.rango_variable_dependiente_all_rows(1,2)
     pp.conjunto_de_entrenamiento(1/3,0)
    
     
     
     yt = pp.Y_test
     Y_train = pp.Y_train
     xt= pp.X_test
     X_train = pp.X_train
    
     modeloDeRegressionLineal = SimpleLinearRegression(X_train, Y_train,"Pronostico de sueldos 2023","Años de experiencia","Sueldos")
     modeloDeRegressionLineal.y_pred(xt)
     y_predict = modeloDeRegressionLineal.ypred
     modeloDeRegressionLineal.graph()
     mde = SimpleLinearRegression(xt, yt, "Pronostico test", "experiencia", "Sueldo")
     mde.graph()
     
     """
      pp.escalar_x()
     pp.escalar_y()
     
  
     pp.limpiar_na_med(1,3)
    # pp.codificar_categoricos_x(0)
     pp.codificar_categoricos_y(0)
     pp.codificar_datos_nominales_x(0)
     #pp.codificar_datos_nominales_y(0)
     X = pp.X
     Y = pp.Y
     """