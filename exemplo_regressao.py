"""
Created on Fri Aug 19 10:17:36 2022
Código exemplo.
@author: Daniel Pordeus Menezes
"""

from classemultiteste import MultiTeste
import pandas as pd
import numpy as np

dataset = pd.read_excel("Dataset_covid_orto_reg.xlsx")
dataset = dataset.dropna()

y = np.array(dataset["internacao"])

X = np.array(dataset)
coluna_final = X.shape[1]
X = X[:,:coluna_final-1]
X = np.array(X, dtype=float)

MT = MultiTeste(X,y,'regressao')
resposta = MT.Regressao()
ordenada = MT.OrdenaMetrica(resposta,'RMSE')

resposta

ordenada
