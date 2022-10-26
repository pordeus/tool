# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:54:04 2022

@author: cp37

Código para Teste Covid

"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from ClasseMultiTeste import MultiTeste

## MAIN - Execução
print("Classificação Binária (1+2 x 3)")
dataset = pd.read_excel("Metricas_VFC_Covid_Modificado_bin.xlsx")
#dataset = dataset.drop(columns=["PACIENTES"])
coluna = 'PACIENTE'

inicio = time.time()
MT = MultiTeste()
MT.Sorteio(dataset, coluna, 80, "exames")
#X_treino, X_teste, y_treino, y_teste = MT.sorteiaExames(dataset, coluna, 80)
#MT.ClassificadorMultiClasse(classes=['Baixo', 'Moderado','Grave'])

rodadas = 10
resultado = []
for i in range(rodadas):
    resultado.append(MT.ClassificadorMedico('binary'))

resultado = np.array(resultado)
#resultado_final = []
resultado_final = pd.DataFrame(columns=['algoritmo', 'revogação', 'precisão', 'f1', 'acurácia', 'roc auc'])
recall = []
precisao = []
f1 = []
acuracia = []
roc = []
algoritmos = []
for k in range(20): #qtd algoritmos
    media_precisao = 0
    media_f1 = 0
    media_recall = 0
    media_acuracia = 0
    media_roc = 0
    for j in range(rodadas):
        media_recall += resultado[j,k,1]
        media_precisao += resultado[j,k,2]
        media_f1 += resultado[j,k,3]
        media_acuracia += resultado[j,k,4]
        media_roc += resultado[j,k,5]
    algoritmos.append(resultado[j,k,0])
    precisao.append(media_precisao/rodadas)
    f1.append(media_f1/rodadas)
    recall.append(media_recall/rodadas)
    acuracia.append(media_acuracia/rodadas)
    roc.append(media_roc/rodadas)
    #print(resultado[j,k,0])
    #print(f"Recall = {media_recall} Precisao = {media_precisao} F1 Score = {media_f1}")
resultado_final['algoritmo'] = algoritmos
resultado_final['revogação'] = recall
resultado_final['precisão'] = precisao
resultado_final['f1'] = f1
resultado_final['acurácia'] = acuracia
resultado_final['roc auc'] = roc

print(f"Tempo decorrido: {(time.time() - inicio)/60:.2f} min")
print()
print("RECALL")
ordenada = MT.OrdenaMetrica(resultado_final,'revogação',"sim")
print(ordenada)
print()

print("Precisão")
ordenada = MT.OrdenaMetrica(resultado_final,'precisão',"sim")
print(ordenada)
print()
print("F1 Score")
ordenada = MT.OrdenaMetrica(resultado_final,'f1',"sim")
print(ordenada)


ordenada.to_excel("resultado.xlsx")

for matrix in MT.MatrizConf:
    print(matrix[0])
    print(matrix[1])
    print(matrix[2])
    #(ConfusionMatrixDisplay(confusion_matrix=matrix[1])).plot()
#plt.show()


X_treino, X_valid, X_teste, y_treino, y_valid, y_teste = MT.sorteiaExamesValidacao(dataset, coluna)

# meu gridsearch para GBR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
#GBR = GradientBoostingRegressor()
inicio = time.time()
best_mse_gbr = 0
pontos = []
for estado in range(1,9):
    for alpha in [0.13, 0.14, 0.15]:
        for samples in range(2, 11, 1):#, 10, 12]:
            gbr = GradientBoostingClassifier(learning_rate=alpha, min_samples_split=samples, min_samples_leaf=50, max_depth=10, max_features='sqrt', random_state=estado)
            gbr.fit(X_treino, y_treino)
            y_prev = gbr.predict(X_valid)
            #y_prev = inverse_output_transform(output_scaler.inverse_transform(y_prev))
            #pontuacao = mean_absolute_percentage_error(y_teste, y_prev)
            erro_abs = recall_score(y_valid, y_prev)# mean_absolute_percentage_error(y_valid, y_prev)
            pontos.append(erro_abs)
            if erro_abs > best_mse_gbr:
                best_mse_gbr = erro_abs
                best_a = alpha
                best_sample = samples
                best_estado = estado

print('Tempo decorrido: {:.0f}min'.format((time.time() - inicio)/60))
print('melhor resultado: Estado Rand = {}, Alpha={}, Sample = {}, Recall Score = {:.4f}'.format(best_estado, best_a, best_sample, best_mse_gbr))

print("GBR com Melhores Parametros")
gbr_melhor = GradientBoostingClassifier(learning_rate=best_a, min_samples_split=best_sample, min_samples_leaf=50, max_depth=10, max_features='sqrt', random_state=best_estado)
gbr_melhor.fit(X_treino, y_treino)
y_prev_melhor = gbr_melhor.predict(X_teste)
#y_prev_melhor = inverse_output_transform(output_scaler.inverse_transform(y_prev_melhor))
#recall_score(y_teste, y_prev_melhor) #np.abs( (y_teste - y_prev_melhor) / y_teste)
#recallScore = recall_score(y_teste, y_prev_melhor)
#recallScore = recall_score(y_teste, y_prev_melhor)
#recallScore = recall_score(y_teste, y_prev_melhor)
matriz_confusao = confusion_matrix(y_teste, y_prev_melhor)
print(f"Revogação  = {recall_score(y_teste, y_prev_melhor)}")
print(f"F1  = {f1_score(y_teste, y_prev_melhor)}")
print(f"Precisão  = {precision_score(y_teste, y_prev_melhor)}")
print(f"Acurácia = {accuracy_score(y_teste, y_prev_melhor)}")
print(f"ROC AUC = {roc_auc_score(y_teste, y_prev_melhor)}")

(ConfusionMatrixDisplay(confusion_matrix=matriz_confusao)).plot()
plt.show()


# Gridsearch árvore de decisão
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
tree_clas = DecisionTreeClassifier(random_state=1024)
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(X_treino, y_treino)
final_model = grid_search.best_estimator_
final_model
tree_clas = DecisionTreeClassifier(ccp_alpha=0.1, max_depth=5, max_features='auto',
                       random_state=1024)
tree_clas.fit(X_treino, y_treino)
y_predict = tree_clas.predict(X_valid)
print(f"Revogação  = {recall_score(y_valid, y_predict)}")
print(f"F1  = {f1_score(y_valid, y_predict)}")
print(f"Precisão  = {precision_score(y_valid, y_predict)}")
print(f"Acurácia = {accuracy_score(y_valid, y_predict)}")
print(f"ROC AUC = {roc_auc_score(y_valid, y_predict)}")

# meu gridsearch para DecisionTree

inicio = time.time()
medicao_dtc = 0
pontos = []
for criterio in ['gini','entropy']:
    for alpha in [0.5, 0.1, 0.01, .001]:
        for depth in range(5, 11):#, 10, 12]:
            dtc = DecisionTreeClassifier(ccp_alpha=alpha, max_depth=depth, criterion=criterio)
            dtc.fit(X_treino, y_treino)
            y_prev = dtc.predict(X_valid)
            #y_prev = inverse_output_transform(output_scaler.inverse_transform(y_prev))
            #pontuacao = mean_absolute_percentage_error(y_teste, y_prev)
            pontuacao = f1_score(y_valid, y_prev)# mean_absolute_percentage_error(y_valid, y_prev)
            pontos.append(pontuacao)
            if pontuacao > medicao_dtc:
                medicao_dtc = pontuacao
                best_a = alpha
                best_depth = depth
                best_criterio = criterio

print('Tempo decorrido: {:.0f}min'.format((time.time() - inicio)/60))
print('melhor resultado: Criterio = {}, Alpha={}, Depth = {}, F1 Score = {:.4f}'.format(best_criterio, best_a, best_depth, medicao_dtc))                
                
                
melhor_dtc = DecisionTreeClassifier(ccp_alpha=best_a, max_depth=best_depth, criterion=best_criterio)
melhor_dtc.fit(X_treino, y_treino)
y_prev_melhor = dtc.predict(X_teste)
matriz_confusao = confusion_matrix(y_teste, y_prev_melhor)
print(f"Revogação  = {recall_score(y_teste, y_prev_melhor)}")
print(f"F1  = {f1_score(y_teste, y_prev_melhor)}")
print(f"Precisão  = {precision_score(y_teste, y_prev_melhor)}")
print(f"Acurácia = {accuracy_score(y_teste, y_prev_melhor)}")
print(f"ROC AUC = {roc_auc_score(y_teste, y_prev_melhor)}")

(ConfusionMatrixDisplay(confusion_matrix=matriz_confusao)).plot()
plt.show()


## Avaliação com Feature Selection
#from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

estimator = melhor_dtc
#for numero in range(5,int(dataset.shape[0])):
#selector = RFE(estimator, n_features_to_select=numero, step=1)
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X_treino, y_treino)
print(f"Suporte: {selector.support_} ")
print(f"Ranking: {selector.ranking_}")




















