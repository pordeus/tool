# -*- coding: utf-8 -*-
"""ClasseMultiTeste.ipynb

Classe para auxiliar na execução de testes 
utilizando diversos algoritmos de Machine Learning. 
Nesta primeira vesão estão disponiveis os recursos:
    - Rodar testes com 16 algoritmos de classificação
    - A classificação pode ser binária ou multiclasse
    - Os resultados incluem as métricas de acurácia, revogaçao (recall), 
    precisão e F1-Score
    - Função para ordenar a saída por uma das métricas


Desenvolvido por Daniel Pordeus Menezes

Disponível em
    
"""

import warnings
warnings.filterwarnings('ignore')

#Apoio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

#Algortimos
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

#Apoio especifico
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

class MultiTeste:

    resultados = pd.DataFrame(columns=['algoritmo','acurácia', 'revogação', 'precisão', 'f1'])
    metricas = ['accuracy', 'recall', 'precision', 'f1']

    modelos = [
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
        OneVsRestClassifier(LinearSVC(random_state=0)) #multiclass
    ]

    def __init__(self, X, y, tipoDado):
        self.X = preprocessing.normalize(X, norm='l2')
        self.y = y
        self.tipoDado = tipoDado
        if self.tipoDado == 'multiclasse':
            self.y = LabelBinarizer().fit_transform(y)


    ##
    # Função que executa todos os testes. 
    # Ao instanciar a classe, informar no parametro
    # tipoDado se é 'binaria' ou 'multiclasse'. Isso fará
    # o tratamento correto do vetor y para o case de multiplas 
    # categorias de classificação. A saída da função é o um dataframe
    # com os resultados.
    ##     
    def Teste(self):
        seed = 10
        splits = 10
        qtd_modelos = 0
        algoritmos = []
        acuracia = []
#        roc_auc = []
        revogacao = []
        precisao = []
        f1 = []
        
        for modelo in self.modelos:
            print(f"Processando {modelo.__class__.__name__}")
            qtd_modelos += 1
            kfold = model_selection.KFold(n_splits=splits, random_state=seed, shuffle=True)
            algoritmos.append(modelo.__class__.__name__)
            qual_metrica = 0
            for metrica in self.metricas:
                cv_results = model_selection.cross_val_score(modelo, self.X, self.y, cv=kfold, scoring=metrica)
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

        self.resultados['algoritmo'] = algoritmos
        #resultados['roc_auc'] = roc_auc
        self.resultados['acurácia'] = acuracia
        self.resultados['revogação'] = revogacao
        self.resultados['precisão'] = precisao
        self.resultados['f1'] = f1
        return self.resultados

    ##
    # Função para ordenar pela métrica a saída da Função Teste. 
    # Não faz sentido ser usada antes desta função.
    # O parametro 'metrica' é uma string e assume os valores
    # 'accuracy', 'recall', 'precision', 'f1'
    ##
    def OrdenaMetrica(self, metrica):
        return self.resultados.sort_values(metrica, ascending=False, ignore_index=True)

