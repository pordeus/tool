# -*- coding: utf-8 -*-
"""Original file is located at
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

morteSubita = pd.read_excel("Metricas_Morte_Subita_Chagas.xlsx")
col = 'PACIENTE'
MT = MultiTeste(morteSubita, col, 70, "exames")
X_treino, X_teste, y_treino, y_teste = MT.sorteiaExames(dataset, coluna, 80)
MS = MT.Classificacao()
MS2 = MT.ClassificadorMedico()
print(MS)
print(MS2)

ordenadaMS = MT.OrdenaMetrica(MS,'revogação',"sim")
print(ordenadaMS)
