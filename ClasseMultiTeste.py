# -*- coding: utf-8 -*-
"""ClasseMultiTeste.ipynb
Classe para auxiliar na execução de testes
utilizando diversos algoritmos de Machine Learning.
Nesta primeira versão (18/08/2022) estão disponiveis os recursos:
    - Rodar testes com 16 algoritmos de classificação
    - A classificação pode ser binária ou multiclasse
    - Os resultados incluem as métricas de acurácia, revogaçao (recall),
    precisão e F1-Score
    - Função para ordenar a saída por uma das métricas
v2 (19/08/2022):
    - Disponibilizado função de regressão.
v3 (25/08/2022)
    - Regressão especifica para dados médicos.
v4 (31/08/2022)
    - Classificação multiclasses
    - sorteio de base de dados para treino e teste gerais
    - sorteio de base de dados para treino e teste médicos,
        considerando mais de um exame por paciente
v5 (30/09/2022)
    - Método para sortear dados para treino, validação e teste
    em GridSearch
v6 (14/10/2022)
    - Calculo da Matriz de Confusão, que fica armazenado numa
    lista para ser utilizada quando desejável.
    - Método para sortear dados para treino, validação e teste
    em GridSearch com segrega;áo de exames por paciente.


Desenvolvido por Daniel Pordeus Menezes
Disponível em
    https://github.com/pordeus/tool/
"""
import warnings
warnings.filterwarnings('ignore')

#GPU
from numba import jit, cuda

#Apoio
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import random
from typing import Counter
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
#import utils

#Algoritmos classificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import ExtraTreeClassifier
#from sklearn.multioutput import ClassifierChain
#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
#from sklearn.semi_supervised import LabelPropagation
#from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.neighbors import NearestCentroid
#from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
#from sklearn.linear_model import SGDClassifier

#algoritmos regressores
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix

#Apoio especifico
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, median_absolute_error#, root_mean_squared_error
from sklearn.metrics import classification_report
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.gaussian_process.kernels import RBF
#from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

class MultiTeste:

    classificadores = [
        SVC(gamma='auto'),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        LinearSVC(),
        SGDClassifier(max_iter=100, tol=1e-3),
        KNeighborsClassifier(),
        LogisticRegression(solver='lbfgs'),
        #LogisticRegressionCV(cv=3),
        BaggingClassifier(),
        ExtraTreesClassifier(n_estimators=300),
        RandomForestClassifier(max_depth=5, n_estimators=300, max_features=1),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        OneVsRestClassifier(LinearSVC(random_state=0, dual=False)), #multiclass
        LGBMClassifier(),
        GradientBoostingClassifier()
    ]

    regressores = [
        LinearRegression(),
        LGBMRegressor(),
        SGDRegressor(),
        KernelRidge(),
        ElasticNet(),
        BayesianRidge(),
        GradientBoostingRegressor(),
        SVR(),
        MLPRegressor()
    ]

    MatrizConf = []

    def __init__(self):#, bancoDados, coluna, divisao_treino, tipoEstudo):
        pass
        #if (tipoEstudo == 'exames'):
        #    self.X_treino, self.X_teste, self.y_treino, self.y_teste = self.sorteiaExames(bancoDados, coluna, divisao_treino)
        #else:
        #    self.X_treino, self.X_teste, self.y_treino, self.y_teste = self.sorteioTreinoTeste(bancoDados, divisao_treino)
            #self.setvalues(X_treino, X_teste, y_treino, y_teste)
        #self.X_treino = preprocessing.normalize(X_treino, norm='l2')
        #self.X_teste = preprocessing.normalize(X_teste, norm='l2')
        #self.y_treino = y_treino
        #self.y_teste = y_teste
        #self.tipoDado = tipoDado
        #if self.tipoDado == 'multiclasse':
        #    self.y_treino = LabelBinarizer().fit_transform(y_treino)
        #    self.y_teste = LabelBinarizer().fit_transform(y_teste)

    def setvalues(self, X_treino, X_teste, y_treino, y_teste):
        self.X_treino = preprocessing.normalize(X_treino, norm='l2')
        self.X_teste = preprocessing.normalize(X_teste, norm='l2')
        self.y_treino = y_treino
        self.y_teste = y_teste

    def Sorteio(self, bancoDados, coluna, divisao_treino, tipoEstudo='normal'):
        if (tipoEstudo.lower() == 'exames'):
            self.X_treino, self.X_teste, self.y_treino, self.y_teste = self.sorteiaExames(bancoDados, coluna, divisao_treino)
        else:
            self.X_treino, self.X_teste, self.y_treino, self.y_teste = self.sorteioTreinoTeste(bancoDados, divisao_treino)

    ##
    # Função que executa todos os testes de Classificação.
    # Ao instanciar a classe, informar no parametro
    # tipoDado se é 'binaria' ou 'multiclasse'. Isso fará
    # o tratamento correto do vetor y para o case de multiplas
    # categorias de classificação. A saída da função é o um dataframe
    # com os resultados.
    ##
    def Classificador(self):
        seed = 10
        splits = 10
        qtd_modelos = 0
        algoritmos = []
        acuracia = []
        #roc_auc = []
        revogacao = []
        precisao = []
        f1 = []
        resultados = pd.DataFrame(columns=['algoritmo','acurácia', 'revogação', 'precisão', 'f1'])
        metricas_class = ['accuracy', 'recall', 'precision', 'f1']

        for modelo in self.classificadores:
            #print(f"Processando {modelo.__class__.__name__}")
            qtd_modelos += 1
            kfold = model_selection.KFold(n_splits=splits, random_state=seed, shuffle=True)
            algoritmos.append(modelo.__class__.__name__)
            qual_metrica = 0
            for metrica in metricas_class:
                cv_results = model_selection.cross_val_score(modelo, self.X_treino, self.y_treino, cv=kfold, scoring=metrica)
                if qual_metrica == 0:
                    acuracia.append(cv_results.mean())
                if qual_metrica == 1:
                    revogacao.append(cv_results.mean())
                if qual_metrica == 2:
                    precisao.append(cv_results.mean())
                if qual_metrica == 3:
                    f1.append(cv_results.mean())
                qual_metrica += 1
        #print("Fim de Processamento.")

        resultados['algoritmo'] = algoritmos
        #resultados['roc_auc'] = roc_auc
        resultados['acurácia'] = acuracia
        resultados['revogação'] = revogacao
        resultados['precisão'] = precisao
        resultados['f1'] = f1
        return resultados


    # tipoDado = [binary, multiclasse=[weighted, sampled, etc]]
    @jit(target_backend='cuda')
    def ClassificadorMedico(self):
        #qtd_modelos = 0
        algoritmos = []
        revogacao = []
        precisao = []
        acuracia = []
        #roc = []
        f1 = []
        resultados = pd.DataFrame(columns=['algoritmo', 'acurácia', 'f1', 'revogação', 'precisão'])#, 'roc auc'])
        #metricas_class = [recall_score, precision_score, f1_score, accuracy_score]#, roc_auc_score]
        metricas_class = ["revogacao", "precisao", "f1", "acuracia"]#, roc_auc_score]
        for modelo in self.classificadores:
            #print(f"Processando {modelo.__class__.__name__}")
            algoritmos.append(modelo.__class__.__name__)
            #qual_metrica = 0
            modelo.fit(self.X_treino, self.y_treino)
            y_pred_teste = modelo.predict(self.X_teste)
            falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo = self.metricasBasicas(self.y_teste, y_pred_teste)
            #self.MatrizConf.append([modelo.__class__.__name__, metrica, confusion_matrix(self.y_teste, y_pred_teste)])
            for metrica in metricas_class:
                if metrica == "revogacao": #qual_metrica == 0:
                    if (verdadeiroPositivo > 0 and verdadeiroNegativo > 0):
                        revogacao.append(verdadeiroPositivo/(verdadeiroPositivo+verdadeiroNegativo))
                    else:
                        #revogacao.append(0)#'NaN')
                        revogacao.append(recall_score(self.y_teste, y_pred_teste))
                    #revogacao.append()
                if metrica == "precisao":# qual_metrica == 1:
                    if (verdadeiroPositivo > 0 and falsoPositivo > 0):
                        precisao.append(verdadeiroPositivo/(verdadeiroPositivo+falsoPositivo))
                    else:
                        #precisao.append(0)#'NaN')
                        precisao.append(precision_score(self.y_teste, y_pred_teste))
                if metrica == "f1": #qual_metrica == 2:
                    if (verdadeiroPositivo > 0 or falsoPositivo > 0 or falsoNegativo > 0):
                        f1.append(2*verdadeiroPositivo/(2*verdadeiroPositivo+falsoPositivo+falsoNegativo))
                    else:
                        #f1.append(0)#'NaN')
                        f1.append(f1_score(self.y_teste, y_pred_teste))
                if metrica == "acuracia": #qual_metrica == 3:
                    if (verdadeiroPositivo > 0 or falsoPositivo > 0 or falsoNegativo > 0 or verdadeiroNegativo > 0):
                        acuracia.append((verdadeiroPositivo+verdadeiroNegativo)/(verdadeiroPositivo+verdadeiroNegativo+falsoPositivo+falsoNegativo))
                    else:
                        #acuracia.append(0)#'NaN')
                        acuracia.append(accuracy_score(self.y_teste, y_pred_teste))
                #if qual_metrica == 4:
                #    roc.append(roc_auc_score(self.y_teste, y_pred_teste))
                #qual_metrica += 1
        #print("Fim de Processamento.")
        #print(f"Shape algoritmos: {len(algoritmos)}")
        #print(f"Revog Revoação: {len(revogacao)}")
        #print(f"Shape Precisão: {len(precisao)}")
        #print(f"Shape F1: {len(f1)}")
        #print(f"Shape Acurácia: {len(acuracia)}")
        resultados['algoritmo'] = algoritmos
        resultados['revogação'] = revogacao
        resultados['precisão'] = precisao
        resultados['f1'] = f1
        resultados['acurácia'] = acuracia
        #resultados['roc auc'] = roc
        return resultados

    # tipoDado = [binary, multiclasse=[weighted, sampled, etc]]
    # Utilizart para casos de pequenas bases de dados
    # quando há chance de dividsão por zero. 
    # aviso: Há risco de erro por isso. 
    # Este código precisa ser revisto.
    @jit(target_backend='cuda')
    def ClassificadorMedicoPeqDataset(self, listaModelos):
        algoritmos = []
        revogacao = []
        precisao = []
        acuracia = []
        f1 = []
        resultados = pd.DataFrame(columns=['algoritmo', 'acurácia', 'f1', 'revogação', 'precisão'])
        metricas_class = ["revogacao", "precisao", "f1", "acuracia"]
        for modelo in listaModelos:
            algoritmos.append(modelo.__class__.__name__)
            modelo.fit(self.X_treino, self.y_treino)
            y_pred_teste = modelo.predict(self.X_teste)
            falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo = self.metricasBasicas(self.y_teste, y_pred_teste)
            for metrica in metricas_class:
                if metrica == "revogacao":
                    if (verdadeiroPositivo > 0 and verdadeiroNegativo > 0):
                        revogacao.append(verdadeiroPositivo/(verdadeiroPositivo+verdadeiroNegativo))
                    else:
                        if (verdadeiroNegativo > 0):
                            revogacao.append(0)
                        else:
                            revogacao.append(recall_score(self.y_teste, y_pred_teste))
                if metrica == "precisao":
                    if (verdadeiroPositivo > 0 and falsoPositivo > 0):
                        precisao.append(verdadeiroPositivo/(verdadeiroPositivo+falsoPositivo))
                    else:
                        precisao.append(precision_score(self.y_teste, y_pred_teste))
                if metrica == "f1":
                    if (verdadeiroPositivo > 0 or falsoPositivo > 0 or falsoNegativo > 0):
                        f1.append(2*verdadeiroPositivo/(2*verdadeiroPositivo+falsoPositivo+falsoNegativo))
                    else:
                        f1.append(f1_score(self.y_teste, y_pred_teste))
                if metrica == "acuracia":
                    if (verdadeiroPositivo > 0 or falsoPositivo > 0 or falsoNegativo > 0 or verdadeiroNegativo > 0):
                        acuracia.append((verdadeiroPositivo+verdadeiroNegativo)/(verdadeiroPositivo+verdadeiroNegativo+falsoPositivo+falsoNegativo))
                    else:
                        acuracia.append(accuracy_score(self.y_teste, y_pred_teste))

        resultados['algoritmo'] = algoritmos
        resultados['revogação'] = revogacao
        resultados['precisão'] = precisao
        resultados['f1'] = f1
        resultados['acurácia'] = acuracia
        return resultados

    def ClassificadorMedico2(self, listaModelos):
        algoritmos = []
        revogacao = []
        precisao = []
        acuracia = []
        f1 = []
        resultados = pd.DataFrame(columns=['algoritmo', 'acurácia', 'f1', 'revogação', 'precisão'])
        metricas_class = ["revogacao", "precisao", "f1", "acuracia"]
        for modelo in listaModelos:
            algoritmos.append(modelo.__class__.__name__)
            modelo.fit(self.X_treino, self.y_treino)
            y_pred_teste = modelo.predict(self.X_teste)
            falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo = self.metricasBasicas(self.y_teste, y_pred_teste)
            for metrica in metricas_class:
                if metrica == "revogacao":
                    revogacao.append(recall_score(self.y_teste, y_pred_teste))
                if metrica == "precisao":
                    precisao.append(precision_score(self.y_teste, y_pred_teste))
                if metrica == "f1":
                    f1.append(f1_score(self.y_teste, y_pred_teste))
                if metrica == "acuracia":
                    acuracia.append(accuracy_score(self.y_teste, y_pred_teste))
        resultados['algoritmo'] = algoritmos
        resultados['revogação'] = revogacao
        resultados['precisão'] = precisao
        resultados['f1'] = f1
        resultados['acurácia'] = acuracia
        return resultados

    
    def ClassificadorMultiClasse(self, classes):
        #qtd_modelos = 0
        for modelo in self.classificadores:
            print(f"Algoritmo {modelo.__class__.__name__}")
            #qtd_modelos += 1
            self.avaliaClassificadorMultiClasse(modelo, self.X_treino, self.y_treino, self.X_teste, self.y_teste, classes)


    def metricasBasicas(self, y_original, y_previsto):
        fP = 0
        vP = 0
        fN = 0
        vN = 0
        for x in range(y_original.shape[0]):
            if y_original[x] == 0:
                if y_previsto[x] == 0:
                    vN = vN + 1
                else:
                    fN = fN + 1
            if y_original[x] == 1:
                if y_previsto[x] == 1:
                    vP = vP + 1
                else:
                    fP = fP + 1

        return fP, vP, fN, vN

    def medicoesManuais(self, falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo):
        #Recall
        if (verdadeiroPositivo > 0):
            print(f"-->Revogação: {verdadeiroPositivo/(verdadeiroPositivo+verdadeiroNegativo)}")
        else:
            if (verdadeiroNegativo > 0):
                print("-->Revogação: 0")
            else:
                print(f"-->Erro na Revogação. VP={verdadeiroPositivo} VN={verdadeiroNegativo}")
        #Precisão
        if (verdadeiroPositivo > 0):
            print(f"-->Precisão: {verdadeiroPositivo/(verdadeiroPositivo+falsoPositivo)}")
        else:
            if(falsoPositivo > 0):
                print("-->Precisão: 0")
            else:
                print(f"-->Erro na Precisão. VP={verdadeiroPositivo} FP={falsoPositivo}")
        #F1 Score
        if (verdadeiroPositivo > 0 or falsoPositivo > 0 or falsoNegativo > 0):
            print(f"-->F1 Score: {(2*verdadeiroPositivo/(2*verdadeiroPositivo+falsoPositivo+falsoNegativo))}")
        else:
            print(f"-->Erro no F1 Score. VP={verdadeiroPositivo} FN={falsoNegativo} FP={falsoPositivo}")
        #Acurácia
        if (verdadeiroPositivo > 0 or falsoPositivo > 0 or falsoNegativo > 0 or verdadeiroNegativo > 0):
            print(f"-->Acurácia: {(verdadeiroPositivo+verdadeiroNegativo)/(verdadeiroPositivo+verdadeiroNegativo+falsoPositivo+falsoNegativo)}")
        else:
            print(f"-->Erro na Acurácia. VP={verdadeiroPositivo} FN={falsoNegativo} FP={falsoPositivo} VN={verdadeiroNegativo}")

    def formataSaida(self, valor):
        saidaFormatada = "{:.2f}".format(valor*100)
        return saidaFormatada + "%"

    ## Avaliador padrão
    def avaliaClassificadorGeral(self, clf, kf, X, y, f_metrica):
        metrica_val = []
        metrica_train = []
        for train, valid in kf.split(X,y):
            x_train = X[train]
            y_train = y[train]
            x_valid = X[valid]
            y_valid = y[valid]
            clf.fit(x_train, y_train)
            y_pred_val = clf.predict(x_valid)
            y_pred_train = clf.predict(x_train)
            metrica_val.append(f_metrica(y_valid, y_pred_val))
            metrica_train.append(f_metrica(y_train, y_pred_train))
        return np.array(metrica_val).mean(), np.array(metrica_train).mean()

    ## Avaliador considerando exames médicos
    # Não há rodadas de cross validation pq não há
    # como garantir a separação correta de exames e pacientes.
    def avaliaClassificadorExames(self, clf, X_treino, y_treino, X_teste, y_teste, f_metrica):
        #metrica_val = []
        #metrica_train = []
        clf.fit(X_treino, y_treino)
        #y_pred_train = clf.predict(X_treino)
        y_pred_val = clf.predict(X_teste)
        metrica_teste = f_metrica(y_teste, y_pred_val)
        #metrica_treino = f_metrica(y_treino, y_pred_train)
        #print(f"Score Treino: {clf.score(X_treino, y_pred_train)}")
        #print(f"Score Teste: {clf.score(X_teste, y_pred_val)}")
        return metrica_teste#metrica_treino, metrica_teste

    def avaliaClassificadorMultiClasse(self, clf, X_treino, y_treino, X_teste, y_teste, classes):
        clf.fit(X_treino, y_treino)
        y_pred_val = clf.predict(X_teste)
        #print(classification_report(y_treino, y_pred_train, target_names=classes))
        #print("Teste - Validação")
        print(classification_report(y_teste, y_pred_val, target_names=classes))
        #return metrica_treino, metrica_teste

    def apresentaMetrica(self, nome_metrica, metrica_val, metrica_train, percentual = False):
        c = 100.0 if percentual else 1.0
        print('{} (validação): {}{}'.format(nome_metrica, metrica_val * c, '%' if percentual else ''))
        print('{} (treino): {}{}'.format(nome_metrica, metrica_train * c, '%' if percentual else ''))


    ##
    # Função para ordenar pela métrica a saída da Função Teste.
    # Não faz sentido ser usada antes desta função.
    # O parametro 'metrica' é uma string e assume os valores
    # 'accuracy', 'recall', 'precision', 'f1', 'MAE', 'MSE' ou 'RMSE'
    ##
    def OrdenaMetrica(self, saida, metrica, descendente):
        self.saida = saida
        ascendente = True
        if (descendente.lower() == "sim"):
            ascendente = False
        return self.saida.sort_values(metrica, ascending=ascendente, ignore_index=True)

    def Regressao(self):
        seed = 10
        splits = 10
        qtd_modelos = 0
        algoritmos = []
        MSE = []
        MAE = []
        RMSE = []
        resultados = pd.DataFrame(columns=['algoritmo','MSE', 'MAE', 'RMSE'])
        metricas_reg = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']

        for modelo in self.regressores:
            #print(f"Processando {modelo.__class__.__name__}")
            qtd_modelos += 1
            kfold = model_selection.KFold(n_splits=splits, random_state=seed, shuffle=True)
            algoritmos.append(modelo.__class__.__name__)
            qual_metrica = 0
            for metrica in metricas_reg:
                cv_results = model_selection.cross_val_score(modelo, self.X_treino, self.y_treino, cv=kfold, scoring=metrica)
                if qual_metrica == 0:
                    MSE.append(cv_results.mean())
                if qual_metrica == 1:
                    MAE.append(cv_results.mean())
                if qual_metrica == 2:
                    RMSE.append(cv_results.mean())
                qual_metrica += 1
        #print("Fim de Processamento.")

        resultados['algoritmo'] = algoritmos
        resultados['MSE'] = MSE
        resultados['MAE'] = MAE
        resultados['RMSE'] = RMSE
        return resultados

    def RegressaoMedica(self):
        qtd_modelos = 0
        algoritmos = []
        MSE = []
        MAE = []
        #RMSE = []
        R2 = []
        MEDIANA = []
        resultados = pd.DataFrame(columns=['Algoritmo','MSE', 'MAE', 'R2', 'MEDIANA EA'])
        #metricas_reg = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']
        metricas_reg = [mean_squared_error, mean_absolute_error, r2_score, median_absolute_error]#,root_mean_squared_error]

        for modelo in self.regressores:
            #print(f"Processando {modelo.__class__.__name__}")
            qtd_modelos += 1
            #kfold = model_selection.KFold(n_splits=splits, random_state=seed, shuffle=True)
            algoritmos.append(modelo.__class__.__name__)
            qual_metrica = 0
            for metrica in metricas_reg:
                #print(f"Metrica {metrica}")
                resultado_teste = self.avaliaClassificadorExames(modelo, self.X_treino, self.y_treino,
                                                                                   self.X_teste, self.y_teste, metrica)
                if qual_metrica == 0:
                    MSE.append(resultado_teste)
                    #RMSE.append(np.sqrt(resultado_teste))
                if qual_metrica == 1:
                    MAE.append(resultado_teste)
                if qual_metrica == 2:
                   R2.append(resultado_teste)
                if qual_metrica == 3:
                   MEDIANA.append(resultado_teste)
                qual_metrica += 1
        #print("Fim de Processamento.")

        resultados['Algoritmo'] = algoritmos
        resultados['MSE'] = MSE
        resultados['MAE'] = MAE
        resultados['R2'] = R2
        resultados['MEDIANA EA'] = MEDIANA
        return resultados

    def RegressaoMedicaAlgoritmo(self, modelo):
        algoritmos = []
        MSE = []
        MAE = []
        R2 = []
        MEDIANA = []
        resultados = pd.DataFrame(columns=['Algoritmo','MSE', 'MAE', 'R2', 'MEDIANA EA'])
        #metricas_reg = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']
        metricas_reg = [mean_squared_error, mean_absolute_error, r2_score, median_absolute_error]#,root_mean_squared_error]

        #print(f"Processando {modelo.__class__.__name__}")
        algoritmos.append(modelo.__class__.__name__)
        qual_metrica = 0
        for metrica in metricas_reg:
            #print(f"Metrica {metrica}")
            resultado_teste = self.avaliaClassificadorExames(modelo, self.X_treino, self.y_treino,
                                                                               self.X_teste, self.y_teste, metrica)
            if qual_metrica == 0:
                MSE.append(resultado_teste)
                #RMSE.append(np.sqrt(resultado_teste))
            if qual_metrica == 1:
                MAE.append(resultado_teste)
            if qual_metrica == 2:
               R2.append(resultado_teste)
            if qual_metrica == 3:
               MEDIANA.append(resultado_teste)
            qual_metrica += 1
        #print("Fim de Processamento.")

        resultados['Algoritmo'] = algoritmos
        resultados['MSE'] = MSE
        resultados['MAE'] = MAE
        resultados['R2'] = R2
        resultados['MEDIANA EA'] = MEDIANA
        return resultados


    ## procedimento de sorteio de exames
    # considera como entrada o dataframe completo.
    # Deve ser informado o nome da coluna em que se encontram
    # o nome dos pacientes para que o código faça a contagem.
    # O percentual solicitado é a divisão que se deseja entre
    # treino e teste.
    ##
    def sorteiaExames(self, bancoDados, coluna, percentual_treino):
        Pacientes = Counter(bancoDados[coluna])
        Pacientes_Numpy = np.array(list(Pacientes.items()))

        soma_exames = 0
        for x in Pacientes_Numpy:
            soma_exames += int(x[1]) #somo a quantidade de exames de cada paciente

        percent = percentual_treino/100 #percentual de exames para treinamento
        qtd_exames_treino = int(round(percent*soma_exames))
        qtd_exames_teste = soma_exames - qtd_exames_treino

        #dataset_numpy = np.array(bancoDados)
        dataset_numpy = np.random.permutation(bancoDados)

        sorteados_teste = 0
        pacientes_teste = []
        while sorteados_teste < qtd_exames_teste:
            sorteado = random.choice(list(Pacientes.items()))
            sorteados_teste += int(sorteado[1])
            Pacientes.pop(sorteado[0])
            for exame in np.where(dataset_numpy == sorteado[0])[0]:
                pacientes_teste.append(dataset_numpy[exame])

        #pacientes teste
        pacientes_teste = np.array(pacientes_teste)

        #montando conjunto diferença - pacientes treinamento
        pacientes_treinamento = []
        for restante in Pacientes:
            #print(restante)
            for exame in np.where(dataset_numpy == restante)[0]:
                pacientes_treinamento.append(dataset_numpy[exame])
                #pacientes_treinamento.append(exame)
        pacientes_treinamento = np.array(pacientes_treinamento)

        pacientes_treinamento = np.array(pacientes_treinamento)
        #pacientes_treinamento.shape

        paciente_treino_df = pd.DataFrame(pacientes_treinamento)
        paciente_teste_df = pd.DataFrame(pacientes_teste)

        #eliminando coluna dos pacientes, considerando que é a 1a
        paciente_treino_df = paciente_treino_df.drop(0, axis=1)
        paciente_teste_df = paciente_teste_df.drop(0, axis=1)

        #montagem dos y's
        cols = paciente_treino_df.shape[1]
        y_treino = paciente_treino_df[paciente_treino_df.columns[cols-1]]
        y_treino = np.array(y_treino, dtype=int)
        y_teste = paciente_teste_df[paciente_treino_df.columns[cols-1]]
        y_teste = np.array(y_teste, dtype=int)

        #eliminando ultima coluna dos X
        paciente_treino_df = paciente_treino_df.drop(cols, axis=1)
        paciente_teste_df = paciente_teste_df.drop(cols, axis=1)

        #montagem dos X's
        X_treino = np.array(paciente_treino_df, dtype=float)
        X_teste = np.array(paciente_teste_df, dtype=float)
        return X_treino, X_teste, y_treino, y_teste

    ## procedimento de sorteio padrão para treinamento e teste.
    # Tem como entrada o dataframe completo.
    # Deve ser informado o nome da coluna em que se encontram
    # É considerado que a coluna Y seja sempre a última do dataset.
    # O percentual solicitado é a divisão que se deseja entre
    # treino e teste.
    ##
    def sorteioTreinoTeste(self, bancoDados, percentual_treino):
        percent = percentual_treino/100
        base_numpy = np.array(bancoDados)
        qtd_dados = base_numpy.shape[0]
        qtd_dados_treino = int(np.round(qtd_dados*percent))

        embaralhado = np.random.permutation(base_numpy)
        Treino = embaralhado[0:qtd_dados_treino,:]
        Teste = embaralhado[qtd_dados_treino:,:]

        coluna = base_numpy.shape[1] - 1
        #X
        X_treino = Treino[:,:coluna]
        X_teste = Teste[:,:coluna]

        #Y
        y_treino = Treino[:,coluna] #tb pode ser usado o -1
        y_teste = Teste[:,coluna]

        return X_treino, X_teste, y_treino, y_teste

    ## procedimento de sorteio para treinamento e teste
    # em GRIDSEARCH. Está fixo o valor de 60/20/20 como padrão
    # para a separação do conjunto de dados.
    ##
    def sorteioTreinoValidacaoTeste(self, bancoDados):
        percent_treino = 0.6
        percent_valid = 0.2
        #percent_teste = 0.2
        base_numpy = np.array(bancoDados)
        qtd_dados = base_numpy.shape[0]
        qtd_dados_treino = int(np.round(qtd_dados*percent_treino))
        qtd_dados_valid = int(np.round(qtd_dados*percent_valid))

        embaralhado = np.random.permutation(base_numpy)
        Treino = embaralhado[0:qtd_dados_treino,:]
        Validacao = embaralhado[qtd_dados_treino:qtd_dados_valid,:]
        Teste = embaralhado[qtd_dados_treino+qtd_dados_valid:,:]

        coluna = base_numpy.shape[1] - 1
        #X
        X_treino = Treino[:,:coluna]
        X_valid = Validacao[:,:coluna]
        X_teste = Teste[:,:coluna]

        #Y
        y_treino = Treino[:,coluna] #tb pode ser usado o -1
        y_valid = Validacao[:,coluna]
        y_teste = Teste[:,coluna]

        return X_treino, X_valid, X_teste, y_treino, y_valid, y_teste


    ## procedimento de sorteio de exames
    # considera como entrada o dataframe completo.
    # Deve ser informado o nome da coluna em que se encontram
    # o nome dos pacientes para que o código faça a contagem.
    # O percentual solicitado é a divisão que se deseja entre
    # treino e teste.
    ##
    def sorteiaExamesValidacao(self, bancoDados, coluna):
        Pacientes = Counter(bancoDados[coluna])
        Pacientes_Numpy = np.array(list(Pacientes.items()))

        soma_exames = 0
        for x in Pacientes_Numpy:
            soma_exames += int(x[1]) #somo a quantidade de exames de cada paciente

        percent = 0.6 #percentual de exames para treinamento
        qtd_exames_treino = int(round(percent*soma_exames))
        qtd_exames_valid = int(round(0.2*soma_exames))
        qtd_exames_teste = soma_exames - qtd_exames_treino - qtd_exames_valid

        dataset_numpy = np.array(bancoDados)

        sorteados_valid = 0
        pacientes_valid = []
        while sorteados_valid < qtd_exames_valid:
            sorteado = random.choice(list(Pacientes.items()))
            sorteados_valid += int(sorteado[1])
            Pacientes.pop(sorteado[0])
            for exame in np.where(dataset_numpy == sorteado[0])[0]:
                pacientes_valid.append(dataset_numpy[exame])

        #pacientes validação
        pacientes_valid = np.array(pacientes_valid)


        sorteados_teste = 0
        pacientes_teste = []
        while sorteados_teste < qtd_exames_teste:
            sorteado = random.choice(list(Pacientes.items()))
            sorteados_teste += int(sorteado[1])
            Pacientes.pop(sorteado[0])
            for exame in np.where(dataset_numpy == sorteado[0])[0]:
                pacientes_teste.append(dataset_numpy[exame])

        #pacientes teste
        pacientes_teste = np.array(pacientes_teste)
        #pacientes_teste.shape

        #montando conjunto diferença - pacientes treinamento
        pacientes_treinamento = []
        for restante in Pacientes:
            #print(restante)
            for exame in np.where(dataset_numpy == restante)[0]:
                pacientes_treinamento.append(dataset_numpy[exame])
                #pacientes_treinamento.append(exame)
        pacientes_treinamento = np.array(pacientes_treinamento)

        paciente_treino_df = pd.DataFrame(pacientes_treinamento)
        paciente_teste_df = pd.DataFrame(pacientes_teste)
        paciente_valid_df = pd.DataFrame(pacientes_valid)

        #eliminando coluna dos pacientes, considerando que é a 1a
        paciente_treino_df = paciente_treino_df.drop(0, axis=1)
        paciente_teste_df = paciente_teste_df.drop(0, axis=1)
        paciente_valid_df = paciente_valid_df.drop(0, axis=1)

        #montagem dos y's
        cols = paciente_treino_df.shape[1]
        y_treino = paciente_treino_df[paciente_treino_df.columns[cols-1]]
        y_treino = np.array(y_treino, dtype=int)
        y_teste = paciente_teste_df[paciente_teste_df.columns[cols-1]]
        y_teste = np.array(y_teste, dtype=int)
        y_valid = paciente_valid_df[paciente_valid_df.columns[cols-1]]
        y_valid = np.array(y_valid, dtype=int)

        #eliminando ultima coluna dos X
        paciente_treino_df = paciente_treino_df.drop(cols, axis=1)
        paciente_teste_df = paciente_teste_df.drop(cols, axis=1)
        paciente_valid_df = paciente_valid_df.drop(cols, axis=1)

        #conversao dos X's
        X_treino = np.array(paciente_treino_df, dtype=float)
        X_teste = np.array(paciente_teste_df, dtype=float)
        X_valid = np.array(paciente_valid_df, dtype=float)
        return X_treino, X_valid, X_teste, y_treino, y_valid, y_teste

    ## procedimento de treinamento onde escolhe-se a janela minima
     # de teste, retira-a da base e executa o treinamento com o restante.
     # Retorna a média da métrica selecionada.
    ##
    def deixaUmFora(self, modelo, bancoDados, qtdRodadas, metrica):
        bancoDados = np.random.permutation(bancoDados)
        #tamJanela = np.ceil(len(bancoDados)/qtdRodadas)

        resultado_metrica = 0
        tamanho = len(bancoDados)
        comprimento = tamanho//qtdRodadas
        for i in range(0,tamanho-1, comprimento):
            #print(i)
            teste = bancoDados[i:(i+comprimento)]
            #print(teste)
            auxiliar = np.arange(i,(i+comprimento))
            treino = np.delete(bancoDados, auxiliar,0)
            #print(treino)
            X_treino  = treino[:,1:-1]
            y_treino = treino[:,-1]
            #y_treino = y_treino.reshape(y_treino.shape[0],1)
            X_teste = teste[:,1:-1]
            y_teste = teste[:,-1]
            #print(f"Alvo Y Treino: {utils.multiclass.type_of_target(y_treino)}")
            #y_teste = y_teste.reshape(y_teste.shape[0],1)
            #print(f"Formato Y treino: {type(y_treino)}")
            #print(f"Formato X treino: {type(X_treino)}")
            modelo.fit(X_treino, y_treino)
            previsao = modelo.predict(X_teste)
            resultado_metrica += metrica(y_teste, previsao)

        return resultado_metrica/qtdRodadas

    ## procedimento de treinamento onde escolhe-se a janela minima
     # de teste, retira-a da base e executa o treinamento com o restante.
     # Retorna os vetores para treino e teste.
    ##
    @jit(target_backend='cuda')
    def deixaUmForaXY(self, bancoDados, qtdRodadas):
        bancoDados = np.random.permutation(bancoDados)
        #tamJanela = np.ceil(len(bancoDados)/qtdRodadas)
        Xs_treino = []
        Xs_teste = []
        ys_treino = []
        ys_teste = []
        tamanho = len(bancoDados)
        comprimento = tamanho//qtdRodadas
        for i in range(0,tamanho-1, comprimento):
            #print(i)
            teste = bancoDados[i:(i+comprimento)]
            #print(teste)
            #auxiliar = np.arange(i,(i+comprimento))
            if (i+comprimento) > len(bancoDados):
                auxiliar = np.arange(i,(i+comprimento-1))
            else:
                auxiliar = np.arange(i,(i+comprimento))
            treino = np.delete(bancoDados, auxiliar,0)
            #print(treino)
            X_treino  = treino[:,1:-1]
            y_treino = treino[:,-1]
            #y_treino = y_treino.reshape(y_treino.shape[0],1)
            X_teste = teste[:,1:-1]
            y_teste = teste[:,-1]
            #y_teste = y_teste.reshape(y_teste.shape[0],1)
            #print(f"Formato Y treino: {type(y_treino)}")
            #print(f"Formato X treino: {type(X_treino)}")
            ys_treino.append(y_treino)
            ys_teste.append(y_teste)
            Xs_teste.append(X_teste)
            Xs_treino.append(X_treino)
        return np.array(Xs_treino), np.array(ys_treino), np.array(Xs_teste), np.array(ys_teste)
        #return np.array(Xs_treino, dtype=float), np.array(ys_treino, dtype=int), np.array(Xs_teste, dtype=float), np.array(ys_teste, dtype=int)


    ## Código enviado pelo Pedro Ribeiro.
    # Correção do método Classificador
    ##
    def Classificador2(self):
        #seed = 10
        splits = len(self.X_treino)
        feature = self.X_treino
        target = self.y_treino

        algorithm = []
        accuracy = []
        #roc_auc = []
        recall = []
        precision = []
        f1 = []
        resultados = pd.DataFrame(columns=['algorithm','accuracy', 'recall', 'precision', 'f1'])
        metricas_class = ['accuracy', 'recall', 'precision', 'f1']

        for modelo in self.classificadores:
            #print(f"Processando {modelo.__class__.__name__}")

            kfold = model_selection.KFold(n_splits=splits, shuffle=False, random_state=None)
            algorithm.append(modelo.__class__.__name__)
            cv_results = model_selection.cross_val_predict(modelo, feature, target, cv=kfold)#,error_score=1)

            tru = 0
            fal = 0
            truPos = 0
            falNeg = 0
            for i in range(len(target)):
                if target[i] == cv_results[i]:
                    tru += 1
                else:
                    fal += 1

                if target[i] == cv_results[i] and target[i] == max(target):
                    truPos += 1

                if target[i] != cv_results[i] and target[i] == max(target):
                    falNeg += 1

            truNeg = tru-truPos

            falPos = fal - falNeg

            if (tru+fal) > 0:
                acc = tru/(tru+fal)
            else:
                acc = 'nan'

            if (truPos+falPos) > 0:
                prec = truPos/(truPos+falPos)
            else:
                prec = 'nan'

            if (truPos+falNeg) > 0:
                rec = truPos/(truPos + falNeg)
            else:
                rec = 'nan'

            if (2*truPos + fal) > 0:
                f1score = (2*truPos)/(2*truPos + fal)
            else:
                f1score = 'nan'

            accuracy.append(acc)
            recall.append(rec)
            precision.append(prec)
            f1.append(f1score)

        #print("Fim de Processamento.")

        resultados['algorithm'] = algorithm
        #resultados['roc_auc'] = roc_auc
        resultados['accuracy'] = accuracy
        resultados['recall'] = recall
        resultados['precision'] = precision
        resultados['f1'] = f1
        return resultados

