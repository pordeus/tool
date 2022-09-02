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
v5 (?)
    - Gridsearch (?)
    
    
Desenvolvido por Daniel Pordeus Menezes
Disponível em
    https://github.com/pordeus/tool/
"""
import warnings
warnings.filterwarnings('ignore')

#Apoio
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import random
from typing import Counter
from sklearn import model_selection

#Algoritmos classificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import ExtraTreeClassifier
#from sklearn.multioutput import ClassifierChain
#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.multiclass import OutputCodeClassifier
#from sklearn.multiclass import OneVsOneClassifier
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

#Apoio especifico
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import mean_squared_error,mean_absolute_error#, root_mean_squared_error
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
        LogisticRegressionCV(cv=3),
        BaggingClassifier(), 
        ExtraTreesClassifier(n_estimators=300),
        RandomForestClassifier(max_depth=5, n_estimators=300, max_features=1),
        GaussianNB(), 
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        OneVsRestClassifier(LinearSVC(random_state=0)), #multiclass
        LGBMClassifier(),
        GradientBoostingClassifier(),
        SGDClassifier(),

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

    def __init__(self, bancoDados, coluna, divisao_treino, tipoEstudo):
        if (tipoEstudo == 'exames'):
            self.X_treino, self.X_teste, self.y_treino, self.y_teste = self.sorteiaExames(bancoDados, coluna, divisao_treino)
        else:
            self.X_treino, self.X_teste, self.y_treino, self.y_teste = self.sorteioTreinoTeste(bancoDados, divisao_treino)
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
            print(f"Processando {modelo.__class__.__name__}")
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
        print("Fim de Processamento.")

        resultados['algoritmo'] = algoritmos
        #resultados['roc_auc'] = roc_auc
        resultados['acurácia'] = acuracia
        resultados['revogação'] = revogacao
        resultados['precisão'] = precisao
        resultados['f1'] = f1
        return resultados
    
    
    # tipoDado = [binary, multiclasse=[weighted, sampled, etc]]
    def ClassificadorMedico(self, tipoDado):
        qtd_modelos = 0
        algoritmos = []
        revogacao = []
        precisao = []
        f1 = []
        resultados = pd.DataFrame(columns=['algoritmo', 'revogação', 'precisão', 'f1'])
        metricas_class = [recall_score, precision_score, f1_score]
                    
        for modelo in self.classificadores:
            print(f"Processando {modelo.__class__.__name__}")
            qtd_modelos += 1
            #kfold = model_selection.KFold(n_splits=splits, random_state=seed, shuffle=True)
            algoritmos.append(modelo.__class__.__name__)
            qual_metrica = 0
            for metrica in metricas_class:
                #_, resultado_teste = self.avaliaClassificadorExames(modelo, self.X_treino, self.y_treino, 
                #                                                                   self.X_teste, self.y_teste, metrica)
                modelo.fit(self.X_treino, self.y_treino)
                y_pred_teste = modelo.predict(self.X_teste)
                if qual_metrica == 0:
                    revogacao.append(recall_score(self.y_teste, y_pred_teste, average=tipoDado))
                if qual_metrica == 1:
                    precisao.append(precision_score(self.y_teste, y_pred_teste, average=tipoDado))
                if qual_metrica == 2:
                    f1.append(f1_score(self.y_teste, y_pred_teste, average=tipoDado))
                qual_metrica += 1
        print("Fim de Processamento.")

        resultados['algoritmo'] = algoritmos
        resultados['revogação'] = revogacao
        resultados['precisão'] = precisao
        resultados['f1'] = f1
        return resultados
    
    def ClassificadorMultiClasse(self, classes):
        qtd_modelos = 0
        for modelo in self.classificadores:
            print(f"Algoritmo {modelo.__class__.__name__}")
            qtd_modelos += 1
            self.avaliaClassificadorMultiClasse(modelo, self.X_treino, self.y_treino, self.X_teste, self.y_teste, classes)

    def F1_score(self, revocacao, precisao):
        return 2*(revocacao*precisao)/(revocacao+precisao)
    
    def metricasBasicas(self, y_original, y_previsto):
        falsoPositivo = 0
        verdadeiroPositivo = 0
        falsoNegativo = 0
        verdadeiroNegativo = 0
        for x in range(y_original.shape[0]):
            if y_original[x] == 0:
                if y_previsto[x] == 0:
                    verdadeiroNegativo = verdadeiroNegativo + 1
                else:
                    falsoNegativo = falsoNegativo + 1
            if y_original[x] == 1:
                if y_previsto[x] == 1:
                    verdadeiroPositivo = verdadeiroPositivo + 1
                else:
                    falsoPositivo = falsoPositivo + 1
    
        return falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo

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
        y_pred_train = clf.predict(X_treino)
        y_pred_val = clf.predict(X_teste)
        metrica_teste = f_metrica(y_teste, y_pred_val)
        metrica_treino = f_metrica(y_treino, y_pred_train)
        return metrica_treino, metrica_teste
    
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
            print(f"Processando {modelo.__class__.__name__}")
            qtd_modelos += 1
            kfold = model_selection.KFold(n_splits=splits, random_state=seed, shuffle=True)
            algoritmos.append(modelo.__class__.__name__)
            qual_metrica = 0
            for metrica in metricas_reg:
                cv_results = model_selection.cross_val_score(modelo, self.X, self.y, cv=kfold, scoring=metrica)
                if qual_metrica == 0:
                    MSE.append(cv_results.mean())
                if qual_metrica == 1:
                    MAE.append(cv_results.mean())
                if qual_metrica == 2:
                    RMSE.append(cv_results.mean())
                qual_metrica += 1
        print("Fim de Processamento.")

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
        RMSE = []
        resultados = pd.DataFrame(columns=['algoritmo','MSE', 'MAE', 'RMSE'])
        #metricas_reg = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']
        metricas_reg = [mean_squared_error,mean_absolute_error]#,root_mean_squared_error]
        
        for modelo in self.regressores:
            print(f"Processando {modelo.__class__.__name__}")
            qtd_modelos += 1
            #kfold = model_selection.KFold(n_splits=splits, random_state=seed, shuffle=True)
            algoritmos.append(modelo.__class__.__name__)
            qual_metrica = 0
            for metrica in metricas_reg:
                resultado_treino, resultado_teste = self.avaliaClassificadorExames(modelo, self.X_treino, self.y_treino, 
                                                                                   self.X_teste, self.y_teste, metrica)
                if qual_metrica == 0:
                    MSE.append(resultado_teste)
                    RMSE.append(np.sqrt(resultado_teste))
                if qual_metrica == 1:
                    MAE.append(resultado_teste)
                #if qual_metrica == 2:
                #   RMSE.append(resultado_teste)
                qual_metrica += 1
        print("Fim de Processamento.")

        resultados['algoritmo'] = algoritmos
        resultados['MSE'] = MSE
        resultados['MAE'] = MAE
        resultados['RMSE'] = RMSE
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
        
        dataset_numpy = np.array(bancoDados)
        
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
        
        #eliminando coluna dos pacientes, considerando que é a 1a
        paciente_treino_df = paciente_treino_df.drop(0, axis=1)
        paciente_teste_df = paciente_teste_df.drop(0, axis=1)
        
        #montagem dos y's
        cols = paciente_treino_df.shape[1]        
        y_treino = paciente_treino_df[paciente_treino_df.columns[cols-1]]
        y_treino = np.array(y_treino, dtype=int)
        y_teste = paciente_teste_df[paciente_treino_df.columns[cols-1]]
        y_teste = np.array(y_teste, dtype=int)
        
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
        base_numpy = np.array(bancoDados)
        qtd_dados = base_numpy.shape[0]
        qtd_dados_treino = int(np.round(qtd_dados*percentual_treino))

        embaralhado = np.random.permutation(base_numpy)
        Treino = embaralhado[0:qtd_dados_treino,:]
        Teste = embaralhado[qtd_dados_treino:,:]
        
        coluna = base_numpy.shape[1] - 1
        #X
        X_treino = Treino[:,:coluna]
        X_teste = Teste[:,:coluna]
        
        #Y
        y_treino = Treino[:,coluna]
        y_teste = Teste[:,coluna]

        return X_treino, X_teste, y_treino, y_teste

        

