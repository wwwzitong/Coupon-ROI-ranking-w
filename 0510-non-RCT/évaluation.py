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
from modèles import *
from sklearn.model_selection import train_test_split

def calculate_rmse(true_values, predictions):
    return np.sqrt(np.mean((true_values - predictions) ** 2))

#Expected outcome

def expected_outcome(X, traitement, outcome, policy=None):

    total_outcome = 0
    total_count = len(X)
    traitement_counts = X[traitement].value_counts(normalize=True).to_dict()
    X_copy = X.copy()
    X_copy.reset_index(drop=True, inplace=True)

    if total_count == 0:
        return 0
    
    if policy is None:
            if traitement is not None:
                policy = X[traitement].tolist()
                for i in range(total_count):
                    if X_copy.loc[i, traitement] == policy[i]:
                        total_outcome += X_copy.loc[i, outcome]
                return total_outcome / total_count
            else:
                raise ValueError("Le paramètre 'traitement' doit être fourni.")
    
    
    else:
        for i in range(total_count):
            if X_copy.loc[i, traitement] == policy[i]:
                total_outcome += X_copy.loc[i, outcome]/traitement_counts[policy[i]]
   
    return total_outcome / total_count

#Expected outcome 2

def expected_outcome2(X, traitement, outcome, policy=None):

    total_outcome = 0
    total_count = 0
    traitement_counts = X[traitement].value_counts(normalize=True).to_dict()
    X_copy = X.copy()
    X_copy.reset_index(drop=True, inplace=True)

    if len(X) == 0:
        return 0
    
    if policy is None:
            if traitement is not None:
                policy = X[traitement].tolist()
                for i in range(len(X)):
                    if X_copy.loc[i, traitement] == policy[i]:
                        total_outcome += X_copy.loc[i, outcome]
                        total_count+=1
                return total_outcome / total_count
            else:
                raise ValueError("Le paramètre 'traitement' doit être fourni.")
    
    
    else:
        for i in range(len(X)):
            if X_copy.loc[i, traitement] == policy[i]:
                total_outcome += X_copy.loc[i, outcome]
                total_count+=1
    return total_outcome / total_count

#Calibration

def estim_calibration(X, uplift, traitement, outcome):
    max_uplift_sum = 0
    X_copy = X.copy()
    X_copy.reset_index(drop=True, inplace=True)
    uplift_estim = 0
    for i in range(len(uplift)):
        max_index = np.argmax(uplift[i])
        uplift_estim += uplift[i][max_index]
    expectedoutcome=expected_outcome(X,traitement,outcome,policy=uplift_to_policy(uplift))
    expectedoutcome0 = expected_outcome(X,traitement,outcome,policy= [0] * len(uplift_to_policy(uplift)))
    #A changer oubli de E[Y|T=0]
    return expectedoutcome - expectedoutcome0 - uplift_estim/len(X)

#Uplift per decile

def decile_uplift(X,uplift,treatment,outcome):
    n = len(uplift[0])
    X_prime = X.copy()
    decile=[0]*n
    for i in range(n):
        X_sorted_Ti = X.copy()
        column_name = f"T{i+1}"
        X_sorted_Ti[column_name] = [arr[i] for arr in uplift]
        X_sorted_Ti = X_sorted_Ti.sort_values(by=f"T{i+1}", ascending=False)
        X_sorted_Ti['decile'] = pd.qcut(X_sorted_Ti[f"T{i+1}"], q=10, labels=False, precision=0, duplicates='drop')
        X_segment_1 = X_sorted_Ti[X_sorted_Ti[treatment] == i+1]
        X_segment_0 = X_sorted_Ti[X_sorted_Ti[treatment] == 0]
        decile_avg_visit_1 = X_segment_1.groupby('decile')[outcome].mean()
        decile_avg_visit_0 = X_segment_0.groupby('decile')[outcome].mean()
        decile[i] = decile_avg_visit_1-decile_avg_visit_0
        X_sorted_Ti.drop(columns=[f"T{i+1}"], inplace=True)
        X_sorted_Ti.drop(columns=['decile'], inplace=True)
    return decile
#decile_uplift(Hillstrom_test,uplift_T_learner_RF)
#decile_uplift(Hillstrom_test,uplift_T_learner_RF).plot(kind='bar', xlabel='Décile', ylabel='Uplift', title='Calibration par décile T learner RF pour T = 1')
#plt.show()


#Uplift per decile (ecart par decile)

def decile_uplift2(X,uplift):
    X_prime = X.copy()
    X_prime['T1'] = [arr[0] for arr in uplift]
    X_prime['T2'] = [arr[1] for arr in uplift]
    X_sorted_T1 = X_prime.sort_values(by='T1', ascending=False)
    X_sorted_T1['decile'] = pd.qcut(X_sorted_T1['T1'], q=10, labels=False, precision=0)
    X_segment_1 = X_sorted_T1[X_sorted_T1['segment'] == 1]
    X_segment_0 = X_sorted_T1[X_sorted_T1['segment'] == 0]
    decile_avg_visit_1 = X_segment_1.groupby('decile')['visit'].mean()
    decile_avg_visit_0 = X_segment_0.groupby('decile')['visit'].mean()
    decile = decile_avg_visit_1-decile_avg_visit_0
    first_elements = [sublist[0] for sublist in uplift]
    liste_triee = sorted(first_elements)
    deciles = np.array_split(liste_triee, 10)
    decile_uplift = [np.mean(decile) for decile in deciles]
    return decile - decile_uplift 

def uplift_to_policy(uplift):
    indices = np.argmax(uplift, axis=1) + 1
    for i, values in enumerate(uplift):
        if np.all(values <= 0):
            indices[i] = 0 
    return indices

def calibre_per_decile(X,uplift,treatment,outcome):
    n = len(uplift[0])
    X_prime = X.copy()
    decile=[0]*n
    for i in range(n):
        X_sorted_Ti = X.copy()
        column_name = f"T{i+1}"
        X_sorted_Ti[column_name] = [arr[i] for arr in uplift]
        X_sorted_Ti = X_sorted_Ti.sort_values(by=f"T{i+1}", ascending=False)
        X_sorted_Ti['decile'] = pd.qcut(X_sorted_Ti[f"T{i+1}"], q=10, labels=False, precision=0, duplicates='drop')
        X_segment_1 = X_sorted_Ti[X_sorted_Ti[treatment] == i+1]
        X_segment_0 = X_sorted_Ti[X_sorted_Ti[treatment] == 0]
        decile_avg_visit_1 = X_segment_1.groupby('decile')[outcome].mean()
        decile_avg_visit_0 = X_segment_0.groupby('decile')[outcome].mean()

        decile_avg_visit_tau = X_sorted_Ti.groupby('decile')[column_name].mean()
        #pas ca, la c'est l'uplift des i
        decile[i] = decile_avg_visit_1-decile_avg_visit_0 - decile_avg_visit_tau
        X_sorted_Ti.drop(columns=[f"T{i+1}"], inplace=True)
        X_sorted_Ti.drop(columns=['decile'], inplace=True)
    return decile

def uplift_hat_uplift(X,treatment_name,outcome_name,name_uplift,taille_groupe=30,T0=0,T=1,pas=1):
    moyennes_tau_hat = []
    moyennes_Y = []
    i = 0
    while i*pas+taille_groupe <= len(X):
        debut = i * pas
        fin = i * pas + taille_groupe

        groupe_T = X.iloc[debut:fin].loc[X[treatment_name] == T]
        groupe_T0 = X.iloc[debut:fin].loc[X[treatment_name] == T0]
        
        moyenne_Y_T = groupe_T[outcome_name].mean()
        moyenne_Y_T0 = groupe_T0[outcome_name].mean()
        moyennes_tau_hat.append(moyenne_Y_T - moyenne_Y_T0)

        groupe = X.iloc[debut:fin]
        moyenne_tau = groupe[name_uplift].mean()
        moyennes_Y.append(moyenne_tau)
        i+=1
    return moyennes_Y, moyennes_tau_hat

def sort_lists(list1, list2):
    # Récupérer les indices triés de la liste 1
    sorted_indices = sorted(range(len(list1)), key=lambda i: list1[i])
    # Réorganiser les deux listes en fonction des indices triés
    sorted_list1 = [list1[i] for i in sorted_indices]
    sorted_list2 = [list2[i] for i in sorted_indices]
    return sorted_list1, sorted_list2

def calculate_auc(x, y):
    auc = 0
    for i in range(1, len(x)):
        auc += (x[i] - x[i-1])*((x[i] - x[i-1])/2 + abs(y[i-1] - x[i]) + abs(y[i] - y[i-1])/2)
    return auc

def uplift_hat_uplift(X,treatment_name,outcome_name,name_uplift,taille_groupe=30,T0=0,T=1,pas=1,method='mean'):
    moyennes_tau_hat = []
    moyennes_Y = []
    i = 0
    while i*pas+taille_groupe <= len(X):
        debut = i * pas
        fin = i * pas + taille_groupe

        groupe_T = X.iloc[debut:fin].loc[X[treatment_name] == T]
        groupe_T0 = X.iloc[debut:fin].loc[X[treatment_name] == T0]
        
        moyenne_Y_T = groupe_T[outcome_name].mean()
        moyenne_Y_T0 = groupe_T0[outcome_name].mean()
        moyennes_tau_hat.append(moyenne_Y_T - moyenne_Y_T0)

        groupe = X.iloc[debut:fin]
        if method=='mean':
            moyenne_tau = groupe[name_uplift].mean()
        elif method=='mediane':
            moyenne_tau = groupe[name_uplift].median()
        moyennes_Y.append(moyenne_tau)
        i+=1
    return moyennes_Y, moyennes_tau_hat

def calculate_ATE(X,treatment_name,outcome_name,T0=0,T=1):
    groupe_T = X.loc[X[treatment_name] == T]
    groupe_T0 = X.loc[X[treatment_name] == T0]
    moyenne_Y_T = groupe_T[outcome_name].mean()
    moyenne_Y_T0 = groupe_T0[outcome_name].mean()
    return moyenne_Y_T - moyenne_Y_T0

def courbe_uplift_hat_uplift(X,treatment_name,outcome_name,name_uplift,taille_groupe=30,T0=0,T=1,pas=1,method='mean',ATE=True,AUC=True,plot=True):
    moyennes_Y_tau, moyennes_Y_tau_hat = uplift_hat_uplift(X,treatment_name,outcome_name,name_uplift,taille_groupe=taille_groupe,T0=T0,T=T, pas=pas, method = method)
    moyennes_Y_tau_hat, moyennes_Y_tau = sort_lists(moyennes_Y_tau_hat, moyennes_Y_tau)

    ate=calculate_ATE(X,treatment_name,outcome_name)
    ate_hat=X[name_uplift].mean()

    auc = calculate_auc(moyennes_Y_tau_hat, moyennes_Y_tau)


    if plot:
        if ATE:
            plt.text(0.05, 0.95, f'ATE = {ate:.2f}', ha='left', va='top', transform=plt.gca().transAxes)
            plt.text(0.05, 0.90, f'ATE estimée = {ate_hat:.2f}', ha='left', va='top', transform=plt.gca().transAxes)
        if AUC:     
            plt.text(0.05, 0.85, f'AUC = {auc:.2f}', ha='left', va='top', transform=plt.gca().transAxes)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.plot(moyennes_Y_tau_hat, moyennes_Y_tau_hat, color='red', linestyle='--')
        plt.plot(moyennes_Y_tau_hat, moyennes_Y_tau, marker='.', linestyle='-', markersize=0, linewidth=0.5,label='Uplift en fonction de l\'uplift estimé')
        plt.xlabel('Uplift réel')  
        plt.ylabel('Uplift estimé') 
        plt.legend()
        plt.show()
    return moyennes_Y_tau, moyennes_Y_tau_hat


def calibration(X,treatment_name,outcome_name,name_uplift,taille_groupe=50,pas=50,method='mediane',plot=False):
    
    y_true, y_pred = courbe_uplift_hat_uplift(X,treatment_name,outcome_name,name_uplift,taille_groupe=taille_groupe,pas=pas,method=method,plot=plot)
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    y_calibre = iso_reg.fit_transform(y_pred, y_true)
    calibre = []
    for item in y_calibre:
        calibre.extend([item] * taille_groupe)
    suppr = len(X) - len(calibre)

    last_element = calibre[-1]
    calibre = calibre + [last_element] * suppr
    return calibre

def statistics_elements(dictionary):
    stats = {}
    for key, list_of_lists in dictionary.items():
        # Convertir la liste de listes en un array numpy pour faciliter les calculs
        array = np.array(list_of_lists)
        # Calculer la moyenne, le maximum et le minimum de chaque colonne
        column_means = np.mean(array, axis=0)
        column_maxs = np.max(array, axis=0)
        column_mins = np.min(array, axis=0)
        stats[key] = {
            'mean': column_means.tolist(),
            'max': column_maxs.tolist(),
            'min': column_mins.tolist()
        }
    return stats

def true_expected_outcome(X,policy):
    X = X.reset_index(drop=True)
    sum = 0
    unique_values_list = set(policy) 


    column_names = X.columns.tolist()
        
    for i, row in X.iterrows():
        sum += row['E_y|T'+str(policy[i])]
    return sum/len(X)

def bons_traitements(df,E_y_T='E_Y|T'):
    e_y_t_columns = [col for col in df.columns if col.startswith('E_y|T') and not col.endswith('_no_rand')]
    df_filtered = df[e_y_t_columns]
    max_value_columns = df_filtered.idxmax(axis=1)
    column_mapping = {col: int(col.split('|T')[1]) for col in e_y_t_columns}
    max_value_indices = max_value_columns.map(column_mapping)
    return max_value_indices

def nb_bons_traitements(true,hat):
    compteur = 0
    for i in range(len(true)):
        if true[i] == hat[i]:
            compteur += 1
    return compteur

def get_E_y_value(row,colonne="Bon traitement"):
    col_name = f"E_y|T{int(row[colonne])}"
    return row[col_name]

def policy_to_esperance(X,policy,name="policy"):
    X_copy=X.copy()
    esperance = []
    for idx in range(len(policy)):
        bons_traitements_value = policy[idx]
        col_name = f"E_y|T{bons_traitements_value}"
        value = X.loc[idx, col_name] 
        esperance.append(value)
    X["E_y|T_"+name] = esperance
    return X