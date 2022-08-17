# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:17:36 2022

Código exemplo.

@author: Daniel Pordeus Menezes
"""

from classemultiteste import MultiTeste
import pandas as pd
import numpy as np


dataset = pd.read_excel("Metricas_VFC_Covid_Modificado.xlsx")
dataset = dataset.dropna()
y = np.array(dataset["gravidade"])
X = np.array(dataset)
X = X[:,:44]
X = np.array(X, dtype=float)
MT = MultiTeste(X,y,'multiclasse')
resposta = MT.Teste()
ordenada = MT.OrdenaMetrica('acurácia')

resposta

ordenada = MT.OrdenaMetrica('acurácia')
ordenada