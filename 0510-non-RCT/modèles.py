import seaborn as sns
import matplotlib
#from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from generate_synth import *
import pandas as pd
from sklift.metrics import qini_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import kuplift as kp
import warnings
import importlib
import subprocess
import sys
from évaluation import *
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import re

#Random

def random_expected_outcome(X, traitement):
    uplift_list = []
    traitement_values = X[traitement].unique()
    
    for _ in range(len(X)):
        random_values = np.random.uniform(0, 1, len(traitement_values))
        uplift_list.append(random_values)
    
    return uplift_list

def expected_to_uplift(expected):
    uplift = []

    for sublist in expected:
        diffs = np.array(sublist[1:]) - sublist[0]
        uplift.append(diffs)

    return uplift

def random_uplift(X, traitement):
    uplift_list = []
    traitement_values = X[traitement].unique()
    
    for _ in range(len(X)):
        random_values = np.random.uniform(-1, 1, len(traitement_values)-1)
        uplift_list.append(random_values)
    
    return uplift_list

def random_policy(X, traitement):
    traitements_uniques = X[traitement].unique()
    return [random.choice(traitements_uniques) for _ in range(len(X))]

def uplift_to_policy(uplift):
    indices = np.argmax(uplift, axis=1) + 1
    for i, values in enumerate(uplift):
        if np.all(values <= 0):
            indices[i] = 0 
    return indices

#random_uplift = expected_to_uplift(random_expected_outcome(Hillstrom_test, 'segment'))
#random_policy = uplift_to_policy(random_uplift)


#S learner

class S_Learner(BaseEstimator, TransformerMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        
    def fit(self, X_train, y_train,nom_col_trait):
        self.classifier.fit(X_train, y_train)
        self.nom_col_trait=nom_col_trait
        self.traitements = np.unique(X_train[nom_col_trait])
        return self
        
    def predict_uplift(self, X):
        X_test=X.copy()
        self.X_test_list = []
        for value in self.traitements:
            X_test_copy = X_test.copy()
            X_test_copy[self.nom_col_trait] = value
            self.X_test_list.append(X_test_copy)

        probabilities_0 = self.classifier.predict_proba(self.X_test_list[0])[:, 1]
        probabilities_1 = self.classifier.predict_proba(self.X_test_list[1])[:, 1]
        uplift_values=[]
        for i in range(len(probabilities_0)):
            uplift_values.append([probabilities_1[i] - probabilities_0[i]])
        for X_test in self.X_test_list[2:]:
            probabilities = self.classifier.predict_proba(X_test)[:, 1]
            uplift = probabilities - probabilities_0
            for i in range(len(uplift_values)):
                l=[]
                ll=[]
                for j in uplift_values[i]:
                    ll.append(j)
                ll.append(uplift[i])
                uplift_values[i] = ll
        #return np.array(self.X_test_list)
        uplift_values = np.array(uplift_values)
        return uplift_values
    
    def predict_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmax(uplift_values, axis=1) + 1
        for i, values in enumerate(uplift_values):
            if np.all(values <= 0):
                indices[i] = 0 
        return indices

    def predict_worst_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmin(uplift_values, axis=1) + 1
        for i, values in enumerate(uplift_values):
            if np.all(values >= 0):
                indices[i] = 0 
        return indices

'''

class S_Learner(BaseEstimator, TransformerMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        
    def fit(self, X_train, y_train, nom_col_trait):
        self.classifier.fit(X_train, y_train)
        self.nom_col_trait = nom_col_trait
        self.traitements = np.unique(X_train[nom_col_trait])
        return self
        
    def predict_uplift(self, X):
        # Création d'une matrice pour stocker les probabilités
        uplift_values = np.zeros((X.shape[0], len(self.traitements)))

        # Calcul des probabilités pour chaque traitement
        for i, value in enumerate(self.traitements):
            X_copy = X.copy()
            X_copy[self.nom_col_trait] = value
            uplift_values[:, i] = self.classifier.predict_proba(X_copy)[:, 1]

        # Calculer l'uplift directement
        uplift_values = uplift_values - uplift_values[:, [0]]  # Soustraire les probabilités du traitement de référence

        return uplift_values
    
    def predict_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmax(uplift_values, axis=1)
        # Assurez-vous que les indices sont appropriés
        indices[np.all(uplift_values <= 0, axis=1)] = 0 
        return indices  # Ajuster pour des indices à partir de 1

    def predict_worst_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmin(uplift_values, axis=1)
        indices[np.all(uplift_values >= 0, axis=1)] = 0 
        return indices  # Ajuster pour des indices à partir de 1


    #T learner (M learner ?)


'''

class T_Learner(BaseEstimator, TransformerMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        
    def fit(self, X, nom_col_trait,nom_col_outcome):
        self.classifiers_dict = {}
        self.nom_col_trait = nom_col_trait
        self.nom_col_outcome = nom_col_outcome
        # Obtenir les valeurs uniques de la colonne spécifiée
        self.traitements = np.unique(X[self.nom_col_trait])
        
        # Ajuster un classificateur pour chaque valeur unique
        for traitement in self.traitements:
            # Filtrer les données pour le traitement actuel
            X_trait = X[X[nom_col_trait] == traitement]
            X_train_trait = X_trait.drop(columns=[self.nom_col_outcome, self.nom_col_trait])
            y_train_trait = X_trait[self.nom_col_outcome]
            # Créer et ajuster un classificateur pour ce traitement
            classifier_clone = clone(self.classifier)  
            classifier_clone.fit(X_train_trait, y_train_trait)
            # Stocker le classificateur dans un dictionnaire
            self.classifiers_dict[traitement] = classifier_clone
        return self
        
    def predict_uplift(self, X):
        X_test=X.copy()
        self.X_test_list = []
        for value in self.traitements:
            X_test_copy = X_test.copy()
            X_test_copy[self.nom_col_trait] = value
            X_test_copy = X_test_copy.drop(columns=[self.nom_col_trait])
            self.X_test_list.append(1)   
        probabilities_0 = self.classifiers_dict[self.traitements[0]].predict_proba(X_test_copy)[:, 1]
        probabilities_1 = self.classifiers_dict[self.traitements[1]].predict_proba(X_test_copy)[:, 1]
        uplift_values=[]
        for i in range(len(probabilities_0)):
            uplift_values.append([probabilities_1[i] - probabilities_0[i]])
        i = 2
        for X_test in self.X_test_list[2:]:
            probabilities = self.classifiers_dict[self.traitements[i]].predict_proba(X_test_copy)[:, 1]
            uplift = probabilities - probabilities_0
            for k in range(len(uplift_values)):
                l=[]
                ll=[]
                for j in uplift_values[k]:
                    ll.append(j)
                ll.append(uplift[k])
                uplift_values[k] = ll
            i += 1
        #return np.array(self.X_test_list)
        uplift_values = np.array(uplift_values)
        return uplift_values
    
    def predict_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmax(uplift_values, axis=1) + 1
        for i, values in enumerate(uplift_values):
            if np.all(values <= 0):
                indices[i] = 0 
        return indices
    
    def predict_worst_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmin(uplift_values, axis=1) + 1
        for i, values in enumerate(uplift_values):
            if np.all(values >= 0):
                indices[i] = 0 
        return indices
    
'''

class T_Learner(BaseEstimator, TransformerMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        
    def fit(self, X, nom_col_trait, nom_col_outcome):
        self.classifiers_dict = {}
        self.nom_col_trait = nom_col_trait
        self.nom_col_outcome = nom_col_outcome
        
        # Obtenir les valeurs uniques de la colonne spécifiée
        self.traitements = np.unique(X[self.nom_col_trait])
        
        # Ajuster un classificateur pour chaque valeur unique
        for traitement in self.traitements:
            # Filtrer les données pour le traitement actuel
            X_trait = X[X[self.nom_col_trait] == traitement]
            X_train_trait = X_trait.drop(columns=[self.nom_col_outcome, self.nom_col_trait])
            y_train_trait = X_trait[self.nom_col_outcome]
            
            # Créer et ajuster un classificateur pour ce traitement
            classifier_clone = clone(self.classifier)  
            classifier_clone.fit(X_train_trait, y_train_trait)
            
            # Stocker le classificateur dans un dictionnaire
            self.classifiers_dict[traitement] = classifier_clone
            
        return self
        
    def predict_uplift(self, X):
        # Stockage des probabilités pour chaque traitement
        probabilities = np.array([
            self.classifiers_dict[traitement].predict_proba(X.drop(columns=[self.nom_col_trait]))[:, 1]
            for traitement in self.traitements
        ])
        
        # Calcul des uplift values
        uplift_values = probabilities - probabilities[0]  # Uplift par rapport à la première colonne
        
        return uplift_values
    
    def predict_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmax(uplift_values, axis=0)
        indices[np.all(uplift_values <= 0, axis=0)] = 0  # Réinitialisation pour des uplifts <= 0
        return indices
    
    def predict_worst_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmin(uplift_values, axis=0)
        indices[np.all(uplift_values >= 0, axis=0)] = 0  # Réinitialisation pour des uplifts >= 0
        return indices

'''

#Policy optimale
'''
def determine_value(row):
    # Obtenir les colonnes pertinentes qui commencent par 'E_y|T'
    values = {col: row[col] for col in row.index if col.startswith('E_y|T') and not col.endswith('rand')}
    
    # Trouver la colonne ayant la valeur maximale
    max_col = max(values, key=values.get)  # Renvoie la colonne avec la valeur maximale
    
    # Extraire le nombre après 'E_y|T' dans le nom de la colonne
    number_after_prefix = int(max_col.split('|T')[1])  # Convertir en entier
    
    return number_after_prefix
'''

def determine_value(row):
    # Obtenir les colonnes pertinentes qui commencent par 'E_y|T'
    values = {col: row[col] for col in row.index if col.startswith('E_y|T') and not col.endswith('rand')}
    
    # Trouver la colonne ayant la valeur maximale
    max_col = max(values, key=values.get)  # Renvoie la colonne avec la valeur maximale
    # Utiliser une expression régulière pour extraire le nombre après 'E_y|T'
    match = re.search(r'E_y\|T(\d+)', max_col)  # Extraire le nombre après 'E_y|T'
    
    if match:
        number_after_prefix = int(match.group(1))  # Convertir en entier
    else:
        raise ValueError(f"Le format de la colonne {max_col} est incorrect.")
    
    return number_after_prefix


#def determine_value(row):
#    if row['E_y|T0'] > row['E_y|T1'] and row['E_y|T0'] > row['E_y|T2']:
#        return 0
#    elif row['E_y|T1'] > row['E_y|T2'] and row['E_y|T1'] > row['E_y|T0']:
#        return 1
#    else:
#        return 2
    
def determine_value_2(row):
    if row['E_y|T0'] < row['E_y|T1'] and row['E_y|T0'] < row['E_y|T2']:
        return 0
    elif row['E_y|T1'] < row['E_y|T2'] and row['E_y|T1'] < row['E_y|T0']:
        return 1
    else:
        return 2




class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    
    def fit(self, x, y):
        input_size = 3
        hidden_size = 100000
        output_size = 2 
        self.model = NeuralNetworkClassifier(input_size, hidden_size, output_size)
        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        self.optimizer.zero_grad()
        self.outputs = self.model(x_tensor)
        self.loss = self.criterion(self.outputs, y_tensor)
        self.loss.backward()
        self.optimizer.step()
        return self

    def predict_proba(self,x):
        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        outputs = model(x_tensor)
        outputs = outputs.detach().numpy()
        return outputs
    

class MeanPredictor:
    def fit(self, X):
        df = X.copy()
        self.mean_values = df.groupby('T')['Y'].mean().to_dict()
    
    def predict_E_Y(self,X):
        predictions = []
        for _, row in X.iterrows():
            predictions.append(list(self.mean_values.values()))
        return np.array(predictions)
    
    def predict_uplift(self, X):
        predictions = []
        predictions_uplift=[]
        for _, row in X.iterrows():
            predictions.append(list(self.mean_values.values()))
            pred=[]
            for i in range(len(predictions[0])-1):
                pred.append(float(predictions[0][i+1])-float(predictions[0][0]))
            predictions_uplift.append(pred)
        return np.array(predictions_uplift)
    
    def predict_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmax(uplift_values, axis=1) + 1
        for i, values in enumerate(uplift_values):
            if np.all(values <= 0):
                indices[i] = 0 
        return indices

class X_Learner:
    def __init__(self, classifier):
        self.classifier = classifier
        self.models = {}
    def fit(self, X_train, nom_col_y,nom_col_trait):
        valeurs_uniques_T = X_train[nom_col_trait].unique().tolist()
        self.valeurs_uniques_T = valeurs_uniques_T
        valeurs_uniques_T = [x for x in valeurs_uniques_T if x != 0]
        self.nom_col_trait = nom_col_trait
        self.nom_col_y = nom_col_y
        for i in valeurs_uniques_T:
            model = XLearner(models=self.classifier)
            X_train_copy=X_train.copy()
            X_train_filtered = X_train_copy[X_train_copy[nom_col_trait].isin([0, i])]
            X_train_without_T = X_train_filtered.drop(columns=[nom_col_trait])
            X_train_without_T = X_train_without_T.drop(columns=[nom_col_y])
            T = X_train_filtered[[nom_col_trait]].replace({2: 1})
            y_train = X_train_filtered[[nom_col_y]]
            model.fit(y_train, T, X=X_train_without_T)
            self.models[i] = model
        return self
        
    def predict_uplift(self, X):
        X_test=X.copy()
        self.X_test_list = []
        predict_uplift=[]
        for value, model in self.models.items():
            #X_test = X_test.drop(columns=[self.nom_col_trait])
            pred=model.effect(X_test)
            predict_uplift.append(pred)
        predict_uplift = [list(elements) for elements in zip(*predict_uplift)]
        predict_uplift = [[float(arr[0]) for arr in sublist] for sublist in predict_uplift]
        predict_uplift = np.array(predict_uplift)

        return predict_uplift
    
    def predict_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmax(uplift_values, axis=1) + 1
        for i, values in enumerate(uplift_values):
            if np.all(values <= 0):
                indices[i] = 0 
        return indices

    def predict_worst_policy(self, X):
        uplift_values = self.predict_uplift(X)
        indices = np.argmin(uplift_values, axis=1) + 1
        for i, values in enumerate(uplift_values):
            if np.all(values >= 0):
                indices[i] = 0 
        return indices