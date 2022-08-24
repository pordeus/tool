# -*- coding: utf-8 -*-
"""Tratamento de Dados Chagas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1936ZRdLa3thPB3y7j5XST8IzwwZtzJhL
"""

import pandas as pd
import numpy as np
import random
from typing import Counter
from ClasseMultiTeste import MultiTeste

dataset = pd.read_excel("Banco de Dados Geral rep-Chagas Modificado_class.xlsx")
dataset = dataset.drop(columns=["NYHA", "FE Teicholz"])

Pacientes = Counter(dataset['Nome do Paciente'])

Pacientes_Numpy = np.array(list(Pacientes.items()))

soma_exames = 0
for x in Pacientes_Numpy:
    soma_exames += int(x[1]) #somo a quantidade de exames de cada paciente

percent = 0.8 #percentual de exames para treinamento
qtd_exames_treino = round(percent*soma_exames)
qtd_exames_teste = soma_exames - qtd_exames_treino

dataset_numpy = np.array(dataset)

sorteados_teste = 0
pacientes_teste = []
while sorteados_teste < qtd_exames_teste:
    sorteado = random.choice(list(Pacientes.items()))
    sorteados_teste += int(sorteado[1])
    Pacientes.pop(sorteado[0])
    for exame in np.where(dataset_numpy == sorteado[0])[0]:
        pacientes_teste.append(dataset_numpy[exame])

#pacientes teste - 20%
pacientes_teste = np.array(pacientes_teste)
pacientes_teste.shape

#montando conjunto diferença - pacientes treinamento
pacientes_treinamento = []
for restante in Pacientes:
    #print(restante)
    for exame in np.where(dataset_numpy == restante)[0]:
        pacientes_treinamento.append(dataset_numpy[exame])
        #pacientes_treinamento.append(exame)
pacientes_treinamento = np.array(pacientes_treinamento)

pacientes_treinamento = np.array(pacientes_treinamento)
pacientes_treinamento.shape

paciente_treino_df = pd.DataFrame(pacientes_treinamento)
paciente_teste_df = pd.DataFrame(pacientes_teste)

#eliminando coluna com nomes e coluna com classificação de risco rassi
paciente_treino_df = paciente_treino_df.drop(0, axis=1)
paciente_teste_df = paciente_teste_df.drop(0, axis=1)

#montagem dos y's
y_treino = paciente_treino_df[19]
y_treino = np.array(y_treino, dtype=int)
y_teste = paciente_teste_df[19]
y_teste = np.array(y_teste, dtype=int)

#montagem dos X's
X_treino = np.array(paciente_treino_df, dtype=float)
X_teste = np.array(paciente_teste_df, dtype=float)

## MAIN - Execução
MT = MultiTeste(X_treino,y_treino, X_teste, y_teste,'multiclasse')
resposta = MT.ClassificadorMedico()
ordenada = MT.OrdenaMetrica(resposta, 'f1')
print(ordenada)




