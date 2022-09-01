# -*- coding: utf-8 -*-
"""Tratamento de Dados Chagas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1936ZRdLa3thPB3y7j5XST8IzwwZtzJhL
"""

import pandas as pd
from ClasseMultiTeste import MultiTeste

dataset = pd.read_excel("Banco de Dados Geral rep-Chagas Modificado_class.xlsx")
dataset = dataset.drop(columns=["NYHA", "Rassi pontos"])
coluna = 'Nome do Paciente'

## MAIN - Execução
MT = MultiTeste(dataset, coluna, 70, "exames")
X_treino, X_teste, y_treino, y_teste = MT.sorteiaExames(dataset, coluna, 80)
MT.ClassificadorMultiClasse(classes=['Baixo','Leve', 'Moderado','Grave'])

resultado = MT.ClassificadorMedico()
ordenada = MT.OrdenaMetrica(resultado,'revogação',"sim")
print(ordenada)
