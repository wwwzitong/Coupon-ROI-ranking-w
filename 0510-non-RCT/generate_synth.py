import itertools
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import numpy as np
#from causalml.dataset import synthetic_data
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
from datetime import datetime
import os
from tqdm import tqdm

def objective_function(coefficients):
    a, b, c = coefficients
    return 0

def find_uplift_inter_simple(x1_val, x2_val,uplift_values,intervals_x1,intervals_x2,initial_guess = [0.1, 0.1, 0],a_b_c=None):
    for i, (interval_x2, interval_x1) in enumerate(itertools.product(intervals_x1, intervals_x2)):
        if interval_x1[0] <= x1_val < interval_x1[1] and interval_x2[0] <= x2_val < interval_x2[1]:
            E_y=-0.01
            initial_guess = [0.1, 0.1, 0]
            while E_y>1 or E_y<0:
                ligne=len(uplift_values) - 1 - i//len(uplift_values[0])
                colonne=i%len(uplift_values[0])
                cell = uplift_values[ligne, colonne]
                rand = np.random.normal(loc=0, scale=cell[2])
                if a_b_c[ligne][colonne] == [0,0,0]:
                    constraints = ({'type': 'eq', 'fun': lambda coefficients: cell[1] - (coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[1] + coefficients[2])}, 
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[0] + coefficients[2] - cell[0]})  
                    result = minimize(objective_function, initial_guess, constraints=constraints)
                    result_list = result.x.tolist()
                    a_b_c[ligne][colonne] = result_list
                #constraints = ({'type': 'eq', 'fun': lambda coefficients: cell[1] - (coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[1] + coefficients[2])}, 
                #   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[0] + coefficients[2] - cell[0]})  
                #result = minimize(objective_function, initial_guess, constraints=constraints)
                #E_y = rand+result.x[0]*x1_val+result.x[1]*x2_val+result.x[2] 
                E_y = rand+a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2]  
                E_y_no_rand = a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2] 
            #print(a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2])
            return max(min(E_y, 1), 0), a_b_c, E_y_no_rand
    return 0 

def generate_synthetic_data_MT_simple(uplift_values_0,uplift_values_1,uplift_values_2,n=1000,min_x1=0,max_x1=1,min_x2=0,max_x2=1,initial_guess = [0.1, 0.1, 0.1]):
    x1 = np.random.uniform(low=0, high=1, size=n)
    x2 = np.random.uniform(low=0, high=1, size=n)
    nb_intervalles_x1 = len(uplift_values_1[0])
    nb_intervalles_x2 = len(uplift_values_1)
    amplitude_intervalle_x1 = (max_x1 - min_x1)/nb_intervalles_x1
    amplitude_intervalle_x2 = (max_x2 - min_x2)/nb_intervalles_x2
    intervals_x1 = [(i * amplitude_intervalle_x1, (i + 1) * amplitude_intervalle_x1) for i in range(nb_intervalles_x1)]
    intervals_x2 = [(i * amplitude_intervalle_x2, (i + 1) * amplitude_intervalle_x2) for i in range(nb_intervalles_x2)]
    a_b_c = [[[0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    #a_b_c = np.array(a_b_c)

    E_y_T0 = []
    E_y_no_rand_T0 = []
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c, E_y_no_rand = find_uplift_inter_simple(x1_val, x2_val, uplift_values_0, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c)
        E_y_T0.append(E_y_val)
        E_y_no_rand_T0.append(E_y_no_rand)
    E_y_T1 = []
    E_y_no_rand_T1 = []
    a_b_c = [[[0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c, E_y_no_rand = find_uplift_inter_simple(x1_val, x2_val, uplift_values_1, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c)
        E_y_T1.append(E_y_val)
        E_y_no_rand_T1.append(E_y_no_rand)
    a_b_c = [[[0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    E_y_T2 = []
    E_y_no_rand_T2 = []
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c, E_y_no_rand = find_uplift_inter_simple(x1_val, x2_val, uplift_values_2, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c)
        E_y_T2.append(E_y_val)
        E_y_no_rand_T2.append(E_y_no_rand)


    #E_y_T0 = np.array([find_uplift_inter(x1_val, x2_val,uplift_values_1,intervals_x1,intervals_x2,initial_guess,a_b_c=a_b_c) for x1_val, x2_val in zip(x1, x2)])
    #E_y_T1 = np.array([find_uplift_inter(x1_val, x2_val,uplift_values_2,intervals_x1,intervals_x2,initial_guess,a_b_c=a_b_c) for x1_val, x2_val in zip(x1, x2)])

    E_y_T0 = np.array(E_y_T0)
    E_y_T1 = np.array(E_y_T1)
    E_y_T2 = np.array(E_y_T2)
    E_y_no_rand_T0 = np.array(E_y_no_rand_T0)
    E_y_no_rand_T1 = np.array(E_y_no_rand_T1)
    E_y_no_rand_T2 = np.array(E_y_no_rand_T2)
    tau_1_0 = E_y_T1 - E_y_T0
    tau_2_0 = E_y_T2 - E_y_T0

    data = pd.DataFrame({'X1': x1, 'X2': x2, 'E_y|T0': E_y_T0, 'E_y|T1': E_y_T1, 'E_y|T2': E_y_T2, 'E_y_no_rand_T0':E_y_no_rand_T0,'E_y_no_rand_T1':E_y_no_rand_T1,'E_y_no_rand_T2':E_y_no_rand_T2,'tau_1_0':E_y_T1 - E_y_T0, 'tau_2_0':E_y_T2 - E_y_T0})
    return data


def heatmap_uplift(data,colonne='E_y|T0', bins=5):
    heatmap, xedges, yedges = np.histogram2d(data['X1'], data['X2'], bins=bins, weights=data[colonne])
    counts, _, _ = np.histogram2d(data['X1'], data['X2'], bins=bins)
    
    # Calculer la moyenne de l'uplift pour chaque bin en évitant les divisions par zéro
    mean_uplift = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts != 0)
    
    # Créer la heatmap avec les valeurs moyennes de l'uplift
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_uplift.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis', aspect='auto')
    plt.colorbar(label='Mean Uplift')
    plt.title(f'2D Heatmap of Mean {colonne} based on X1 and X2')
    plt.xlabel('X1')
    plt.ylabel('X2')

    # Ajouter les annotations textuelles pour chaque cellule de la heatmap
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            plt.text((xedges[i] + xedges[i + 1]) / 2, (yedges[j] + yedges[j + 1]) / 2, f'{mean_uplift[i, j]:.2f}', ha='center', va='center', color='black')

    plt.show()


def objective_function_x1x2(coefficients):
    a, b, c, d = coefficients
    return 0


def find_uplift_inter_x1x2(x1_val, x2_val,uplift_values,intervals_x1,intervals_x2,initial_guess = [0.1, 0.1, 0],a_b_c=None):
    for i, (interval_x2, interval_x1) in enumerate(itertools.product(intervals_x1, intervals_x2)):
        if interval_x1[0] <= x1_val < interval_x1[1] and interval_x2[0] <= x2_val < interval_x2[1]:
            E_y=-0.01
            initial_guess = [0.1, 0.1, 0.1,0]
            #while E_y>1 or E_y<0:
            ligne=len(uplift_values) - 1 - i//len(uplift_values[0])
            colonne=i%len(uplift_values[0])
            cell = uplift_values[ligne, colonne]
            rand = np.random.normal(loc=0, scale=cell[4])
            if a_b_c[ligne][colonne] == [0,0,0]:
                constraints = ({'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[0] + coefficients[2] + coefficients[3] * interval_x1[0] * interval_x2[0] - cell[0]}, 
                {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[0] + coefficients[2] + coefficients[3] * interval_x1[1] * interval_x2[0] - cell[1]},
                {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[1] + coefficients[2] + coefficients[3] * interval_x1[1] * interval_x2[1] - cell[2]},
                {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[1] + coefficients[2] + coefficients[3] * interval_x1[0] * interval_x2[1] - cell[3]})  
                result = minimize(objective_function_x1x2, initial_guess, constraints=constraints)
                result_list = result.x.tolist()
                a_b_c[ligne][colonne] = result_list
            #constraints = ({'type': 'eq', 'fun': lambda coefficients: cell[1] - (coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[1] + coefficients[2])}, 
            #   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[0] + coefficients[2] - cell[0]})  
            #result = minimize(objective_function, initial_guess, constraints=constraints)
            #E_y = rand+result.x[0]*x1_val+result.x[1]*x2_val+result.x[2] 
            E_y = rand+a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2]+a_b_c[ligne][colonne][3]*x1_val*x2_val
            E_y_no_rand = a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2]+a_b_c[ligne][colonne][3]*x1_val*x2_val
            #print(a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2])
            return max(min(E_y, 1), 0), a_b_c, E_y_no_rand
    return 0 

def generate_synthetic_data_MT_x1x2(uplift_values_0,uplift_values_1,uplift_values_2,n=1000,min_x1=0,max_x1=1,min_x2=0,max_x2=1,initial_guess = [0.1, 0.1, 0.1]):
    x1 = np.random.uniform(low=0, high=1, size=n)
    x2 = np.random.uniform(low=0, high=1, size=n)
    nb_intervalles_x1 = len(uplift_values_1[0])
    nb_intervalles_x2 = len(uplift_values_1)
    amplitude_intervalle_x1 = (max_x1 - min_x1)/nb_intervalles_x1
    amplitude_intervalle_x2 = (max_x2 - min_x2)/nb_intervalles_x2
    intervals_x1 = [(i * amplitude_intervalle_x1, (i + 1) * amplitude_intervalle_x1) for i in range(nb_intervalles_x1)]
    intervals_x2 = [(i * amplitude_intervalle_x2, (i + 1) * amplitude_intervalle_x2) for i in range(nb_intervalles_x2)]
    a_b_c = [[[0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    #a_b_c = np.array(a_b_c)

    E_y_T0 = []
    E_y_no_rand_T0 = []
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c, E_y_no_rand = find_uplift_inter_x1x2(x1_val, x2_val, uplift_values_0, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c)
        E_y_T0.append(E_y_val)
        E_y_no_rand_T0.append(E_y_no_rand)
    a_b_c = [[[0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    E_y_T1 = []
    E_y_no_rand_T1 = []
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c, E_y_no_rand = find_uplift_inter_x1x2(x1_val, x2_val, uplift_values_1, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c)
        E_y_T1.append(E_y_val)
        E_y_no_rand_T1.append(E_y_no_rand)
    a_b_c = [[[0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    E_y_T2 = []
    E_y_no_rand_T2 = []
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c, E_y_no_rand = find_uplift_inter_x1x2(x1_val, x2_val, uplift_values_2, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c)
        E_y_T2.append(E_y_val)
        E_y_no_rand_T2.append(E_y_no_rand)

    E_y_T0 = np.array(E_y_T0)
    E_y_T1 = np.array(E_y_T1)
    E_y_T2 = np.array(E_y_T2)
    E_y_no_rand_T0 = np.array(E_y_no_rand_T0)
    E_y_no_rand_T1 = np.array(E_y_no_rand_T1)
    E_y_no_rand_T2 = np.array(E_y_no_rand_T2)
    tau_1_0 = E_y_T1 - E_y_T0
    tau_2_0 = E_y_T2 - E_y_T0
    data = pd.DataFrame({'X1': x1, 'X2': x2, 'E_y|T0': E_y_T0, 'E_y|T1': E_y_T1, 'E_y|T2': E_y_T2, 'E_y_no_rand_T0':E_y_no_rand_T0,'E_y_no_rand_T1':E_y_no_rand_T1,'E_y_no_rand_T2':E_y_no_rand_T2,'tau_1_0':E_y_T1 - E_y_T0, 'tau_2_0':E_y_T2 - E_y_T0})

    return data

def create_column_T(data, k):
    random_numbers = np.random.randint(0, k, size=len(data))
    data['T'] = random_numbers
    return data

def create_column_Y(data):
    data['Y'] = data.apply(lambda row: np.random.choice([0, 1], p=[1 - row[f'E_y|T{int(row["T"])}'], row[f'E_y|T{int(row["T"])}']]), axis=1)
    return data


def objective_function_x1x2_grosse_eq(coefficients):
    a, b, c, d, e,f,g,h,i,j,k,l = coefficients
    return 0


def find_uplift_inter_x1x2_grosse_eq(x1_val, x2_val,uplift_values_0,uplift_values_1,uplift_values_2,intervals_x1,intervals_x2,initial_guess = [0.1, 0.1, 0],a_b_c=None,T=0):
    for i, (interval_x2, interval_x1) in enumerate(itertools.product(intervals_x1, intervals_x2)):
        if interval_x1[0] <= x1_val < interval_x1[1] and interval_x2[0] <= x2_val < interval_x2[1]:
            E_y=-0.01
            initial_guess = [0.1, 0.1, 0.1, 0,0.1, 0.1, 0.1, 0,0.1, 0.1, 0.1, 0]
            while E_y>1 or E_y<0:
                ligne=len(uplift_values_0) - 1 - i//len(uplift_values_0[0])
                colonne=i%len(uplift_values_0[0])
                cell_0 = uplift_values_0[ligne, colonne]
                cell_1 = uplift_values_1[ligne, colonne]
                cell_2 = uplift_values_2[ligne, colonne]
                rand_0 = np.random.normal(loc=0, scale=cell_0[4])
                rand_1 = np.random.normal(loc=0, scale=cell_1[4])
                rand_2 = np.random.normal(loc=0, scale=cell_2[4])
                if a_b_c[ligne][colonne] == [0,0,0,0,0,0,0,0,0,0,0,0]:
                    constraints = ({'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[0] + coefficients[2] + coefficients[3] * interval_x1[0] * interval_x2[0] - cell_0[0]}, 
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[0] + coefficients[2] + coefficients[3] * interval_x1[1] * interval_x2[0] - cell_0[1]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[1] + coefficients[2] + coefficients[3] * interval_x1[1] * interval_x2[1] - cell_0[2]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[1] + coefficients[2] + coefficients[3] * interval_x1[0] * interval_x2[1] - cell_0[3]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[0] + coefficients[2] + coefficients[3] * interval_x1[0] * interval_x2[0] + coefficients[4] * interval_x1[0] + coefficients[5] * interval_x2[0] + coefficients[6] + coefficients[7] * interval_x1[0] * interval_x2[0] - cell_1[0]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[0] + coefficients[2] + coefficients[3] * interval_x1[1] * interval_x2[0] + coefficients[4] * interval_x1[1] + coefficients[5] * interval_x2[0] + coefficients[6] + coefficients[7] * interval_x1[1] * interval_x2[0] - cell_1[1]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[1] + coefficients[2] + coefficients[3] * interval_x1[1] * interval_x2[1] + coefficients[4] * interval_x1[1] + coefficients[5] * interval_x2[1] + coefficients[6] + coefficients[7] * interval_x1[1] * interval_x2[1] - cell_1[2]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[1] + coefficients[2] + coefficients[3] * interval_x1[0] * interval_x2[1] + coefficients[4] * interval_x1[0] + coefficients[5] * interval_x2[1] + coefficients[6] + coefficients[7] * interval_x1[0] * interval_x2[1]- cell_1[3]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[0] + coefficients[2] + coefficients[3] * interval_x1[0] * interval_x2[0] + coefficients[8] * interval_x1[0] + coefficients[9] * interval_x2[0] + coefficients[10] + coefficients[11] * interval_x1[0] * interval_x2[0] - cell_2[0]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[0] + coefficients[2] + coefficients[3] * interval_x1[1] * interval_x2[0] + coefficients[8] * interval_x1[1] + coefficients[9] * interval_x2[0] + coefficients[10] + coefficients[11] * interval_x1[1] * interval_x2[0] - cell_2[1]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[1] + coefficients[2] + coefficients[3] * interval_x1[1] * interval_x2[1] + coefficients[8] * interval_x1[1] + coefficients[9] * interval_x2[1] + coefficients[10] + coefficients[11] * interval_x1[1] * interval_x2[1] - cell_2[2]},
                   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[1] + coefficients[2] + coefficients[3] * interval_x1[0] * interval_x2[1] + coefficients[8] * interval_x1[0] + coefficients[9] * interval_x2[1] + coefficients[10] + coefficients[11] * interval_x1[0] * interval_x2[1] - cell_2[3]},)  
                    result = minimize(objective_function_x1x2_grosse_eq, initial_guess, constraints=constraints)
                    result_list = result.x.tolist()
                    a_b_c[ligne][colonne] = result_list
                #constraints = ({'type': 'eq', 'fun': lambda coefficients: cell[1] - (coefficients[0] * interval_x1[1] + coefficients[1] * interval_x2[1] + coefficients[2])}, 
                #   {'type': 'eq', 'fun': lambda coefficients: coefficients[0] * interval_x1[0] + coefficients[1] * interval_x2[0] + coefficients[2] - cell[0]})  
                #result = minimize(objective_function, initial_guess, constraints=constraints)
                #E_y = rand+result.x[0]*x1_val+result.x[1]*x2_val+result.x[2] 
                if T==1:
                    E_y = rand_1+a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2]+a_b_c[ligne][colonne][3]*x1_val*x2_val+a_b_c[ligne][colonne][4]*x1_val+a_b_c[ligne][colonne][5]*x2_val+a_b_c[ligne][colonne][6]+a_b_c[ligne][colonne][7]*x1_val*x2_val
                elif T==2:
                    E_y = rand_2+a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2]+a_b_c[ligne][colonne][3]*x1_val*x2_val+a_b_c[ligne][colonne][8]*x1_val+a_b_c[ligne][colonne][9]*x2_val+a_b_c[ligne][colonne][10]+a_b_c[ligne][colonne][11]*x1_val*x2_val
                elif T==0:
                    E_y = rand_0+a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2]+a_b_c[ligne][colonne][3]*x1_val*x2_val
            #print(a_b_c[ligne][colonne][0]*x1_val+a_b_c[ligne][colonne][1]*x2_val+a_b_c[ligne][colonne][2])
            return max(min(E_y, 1), 0), a_b_c
    return 0 

def generate_synthetic_data_MT_x1x2_grosse_eq(uplift_values_0,uplift_values_1,uplift_values_2,n=1000,min_x1=0,max_x1=1,min_x2=0,max_x2=1,initial_guess = [0.1, 0.1, 0.1]):
    x1 = np.random.uniform(low=0, high=1, size=n)
    x2 = np.random.uniform(low=0, high=1, size=n)
    nb_intervalles_x1 = len(uplift_values_1[0])
    nb_intervalles_x2 = len(uplift_values_1)
    amplitude_intervalle_x1 = (max_x1 - min_x1)/nb_intervalles_x1
    amplitude_intervalle_x2 = (max_x2 - min_x2)/nb_intervalles_x2
    intervals_x1 = [(i * amplitude_intervalle_x1, (i + 1) * amplitude_intervalle_x1) for i in range(nb_intervalles_x1)]
    intervals_x2 = [(i * amplitude_intervalle_x2, (i + 1) * amplitude_intervalle_x2) for i in range(nb_intervalles_x2)]
    a_b_c = [[[0,0,0,0,0,0,0,0,0,0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    #a_b_c = np.array(a_b_c)

    E_y_T0 = []
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c = find_uplift_inter_x1x2_grosse_eq(x1_val, x2_val, uplift_values_0,uplift_values_1,uplift_values_2, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c, T=0)
        E_y_T0.append(E_y_val)
    a_b_c = [[[0,0,0,0,0,0,0,0,0,0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    E_y_T1 = []
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c = find_uplift_inter_x1x2_grosse_eq(x1_val, x2_val, uplift_values_0,uplift_values_1,uplift_values_2, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c,T=1)
        E_y_T1.append(E_y_val)
    a_b_c = [[[0,0,0,0,0,0,0,0,0,0,0,0] for _ in range(len(l))] for l in uplift_values_1]
    E_y_T2 = []
    for x1_val, x2_val in zip(x1, x2):
        E_y_val, a_b_c = find_uplift_inter_x1x2_grosse_eq(x1_val, x2_val, uplift_values_0,uplift_values_1,uplift_values_2, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c,T=2)
        E_y_T2.append(E_y_val)

    E_y_T0 = np.array(E_y_T0)
    E_y_T1 = np.array(E_y_T1)
    E_y_T2 = np.array(E_y_T2)
    tau_1_0 = E_y_T1 - E_y_T0
    tau_2_0 = E_y_T2 - E_y_T0
    data = pd.DataFrame({'X1': x1, 'X2': x2, 'E_y|T0': E_y_T0, 'E_y|T1': E_y_T1, 'E_y|T2': E_y_T2, 'tau_1_0':E_y_T1 - E_y_T0, 'tau_2_0':E_y_T2 - E_y_T0})

    return data

def create_column_T(data, k):
    random_numbers = np.random.randint(0, k, size=len(data))
    data['T'] = random_numbers
    return data

def create_column_Y(data):
    data['Y'] = data.apply(lambda row: np.random.choice([0, 1], p=[1 - row[f'E_y|T{int(row["T"])}'], row[f'E_y|T{int(row["T"])}']]), axis=1)
    return data

def suppression_carre_per_cent(X,coord=[1,1],cote=5,X1="X1",X2="X2",percent=100):
    stats_X1 = X['X1'].describe()
    stats_X2 = X['X2'].describe()
    amp_X1 = stats_X1['max'] - stats_X1['min']
    amp_X2 = stats_X2['max'] - stats_X2['min']
    condition = ((X['X1'] >= coord[0]*amp_X1/cote) & 
             (X['X1'] <= (coord[0]+1)*amp_X1/cote) & 
             (X['X2'] >= coord[1]*amp_X1/cote) & 
             (X['X2'] <= (coord[1]+1)*amp_X1/cote))

    indices_to_remove = X[condition].index
    np.random.seed(42)
    indices_to_remove_percent = np.random.choice(indices_to_remove, size=len(indices_to_remove)*percent//100, replace=False)
    filtered_df = X.drop(indices_to_remove_percent)
    return filtered_df

def motif_to_int(motif,int_x1=[0,1],int_x2=[0,1],decoupage=4):
    ampl_x1=(int_x1[1] - int_x1[0])/decoupage
    ampl_x2=(int_x2[1] - int_x2[0])/decoupage
    new_int=[]
    for i in motif:
        new_int.append([[(i[0]-1)*ampl_x1,i[0]*ampl_x1],[(i[1]-1)*ampl_x1,i[1]*ampl_x1]])
    return new_int

def is_in_int(X,int1,x1='X1',x2='X2',NRA='NRA'):
    X_prime = X.copy()
    X_prime[NRA]=0
    for index, row in X_prime.iterrows():
        for i in int1:
            if (i[0][0] <= row[x1] <= i[0][1]) & (i[1][0] <= row[x2] <= i[1][1]):
                X_prime.at[index, NRA] = 1
    return X_prime

def drop_nra(X,k=100,NRA='NRA'):
    df = X.copy()
    indices_NRA_1 = df[df[NRA] == 1].index.tolist()
    n_to_remove = int(len(indices_NRA_1)*k/100)
    remove_indices = np.random.choice(indices_NRA_1, n_to_remove, replace=False)
    df = df.drop(remove_indices)
    
    return df

def generate_ind(n=1000,min_x1=0,max_x1=1,min_x2=0,max_x2=1):
    x1 = np.random.uniform(low=0, high=1, size=n)
    x2 = np.random.uniform(low=0, high=1, size=n)
    data = pd.DataFrame({'X1': x1, 'X2': x2})
    return data

def generate_ind_NRA(motif,n=1000,min_x1=0,max_x1=1,min_x2=0,max_x2=1,NRA='NRA',X1='X1',X2='X2',k=100,decoupage=4):
    x1 = np.random.uniform(low=0, high=1, size=n)
    x2 = np.random.uniform(low=0, high=1, size=n)
    data = pd.DataFrame({X1: x1, X2: x2})
    data[NRA]=0
    motif = motif_to_int(motif,decoupage=decoupage)
    for index, row in data.iterrows():
        for i in motif:
            if (i[0][0] <= row[X1] <= i[0][1]) & (i[1][0] <= row[X2] <= i[1][1]):
                data.at[index, NRA] = 1
    data = drop_nra(data,k=k,NRA='NRA')
    while len(data) < n:
        x1 = np.random.uniform(low=0, high=1, size=1)
        x2 = np.random.uniform(low=0, high=1, size=1)
        bool=1
        for i in motif:
            if (i[0][0] <= x1 <= i[0][1]) & (i[1][0] <= x2 <= i[1][1]):
                bool=0
        if bool:
            nouvel_individu = pd.DataFrame({X1: x1, X2 : x2, NRA : 0})
            data = pd.concat([data, nouvel_individu], ignore_index=True)
    return data

def gen_synt_E_Y(X,E_Y,name_col='E_y|T0',min_x1=0,max_x1=1,min_x2=0,max_x2=1,initial_guess=[0,0,0]):
    df = X.copy()
    nb_intervalles_x1 = len(E_Y[0])
    nb_intervalles_x2 = len(E_Y)
    amplitude_intervalle_x1 = (max_x1 - min_x1)/nb_intervalles_x1
    amplitude_intervalle_x2 = (max_x2 - min_x2)/nb_intervalles_x2
    intervals_x1 = [(i * amplitude_intervalle_x1, (i + 1) * amplitude_intervalle_x1) for i in range(nb_intervalles_x1)]
    intervals_x2 = [(i * amplitude_intervalle_x2, (i + 1) * amplitude_intervalle_x2) for i in range(nb_intervalles_x2)]
    a_b_c = [[[0,0,0] for _ in range(len(l))] for l in E_Y]
    E_y = []
    E_y_no_rand = []
    for x1_val, x2_val in zip(df['X1'], df['X2']):
        E_y_val, a_b_c, E_y_no_rand_val = find_uplift_inter_x1x2(x1_val, x2_val, E_Y, intervals_x1, intervals_x2, initial_guess, a_b_c=a_b_c)
        E_y.append(E_y_val)
        E_y_no_rand.append(E_y_no_rand_val)
    E_y = np.array(E_y)
    E_y_no_rand = np.array(E_y_no_rand)
    df[name_col]=E_y
    df[name_col+'_no_rand']=E_y_no_rand
    return df

def calculate_uplift(X,test_name,controle_name,tau_name=None):
    if tau_name is None:
        tau_name='tau_'+test_name+'_'+controle_name
    df = X.copy()
    uplift=df[test_name] - df[controle_name]
    df[tau_name]=uplift
    return df

def carre_random(min_val=0,max_val=1,bruit=0,type='tout',prob=0.5):
    if type == 'diag':
        options = [[max_val, min_val, max_val, min_val,bruit],
                   [min_val, max_val, min_val, max_val,bruit]]
        return random.choice(options)
    elif type == 'droit':
        options = [[max_val, max_val, min_val, min_val,bruit],
                    [min_val, max_val, max_val, min_val,bruit],
                    [min_val, min_val, max_val, max_val,bruit],
                    [max_val, min_val, min_val, max_val,bruit]]
        return random.choice(options)
    elif type == 'trois':
        options = [[max_val, max_val, max_val, min_val,bruit],
                    [min_val, max_val, max_val, max_val,bruit],
                    [max_val, min_val, max_val, max_val,bruit],
                    [max_val, max_val, min_val, max_val,bruit]]
        return random.choice(options)
    elif type == 'un':
        options = [[max_val, min_val, min_val, min_val,bruit],
                    [min_val, max_val, min_val, min_val,bruit],
                    [min_val, min_val, max_val, min_val,bruit],
                    [min_val, min_val, min_val, max_val,bruit]]
        return random.choice(options)
    elif type == 'plein':
        options = [[max_val, max_val, max_val, max_val, bruit],
                    [min_val, min_val, min_val, min_val, bruit]]
        return random.choice(options)
    elif type == 'quadrillage':
        options = [
            [max_val, max_val, max_val, max_val, bruit],
            [min_val, min_val, min_val, min_val, bruit]
        ]
        return random.choices(options, weights=[prob, 1-prob], k=1)[0]

    else:
        return [np.random.choice([min_val, max_val]), np.random.choice([min_val, max_val]),
                np.random.choice([min_val, max_val]), np.random.choice([min_val, max_val]), bruit]


def generate_random_motif(dim=4,min_val=0,max_val=1,nb_t=3,bruit=0,type='tout',prob=0.5):
    motifs=[]
    while len(motifs) < nb_t:
        motif=[]
        for j in range(dim):
            liste=[]
            for k in range(dim):
                liste.append(carre_random(min_val=min_val,max_val=max_val,bruit=bruit,type=type,prob=prob))
            motif.append(liste)
        motif=np.array(motif)
        if not any(np.array_equal(motif, m) for m in motifs):
            motifs.append(motif)
    return motifs

def generate_plein_de_motifs(nb=1,dim=4,min_val=0,max_val=1,nb_t=3,bruit=0,type='tout',prob=0.5):
    MOTIFS = []
    while len(MOTIFS) < nb:
        motif=generate_random_motif(dim=dim,min_val=min_val,max_val=max_val,nb_t=nb_t,bruit=bruit,type=type,prob=prob)
        if not any(np.array_equal(motif, m) for m in MOTIFS):
            MOTIFS.append(motif)
    return MOTIFS

def generate_plein_de_datasets(motifs,plot=False,nb_ind_per_t=10000,NRA=0,suffixe='',suffixe_motif='',titre=''):
    parent_directory = "Datasets_générés"+str(suffixe_motif)
    os.makedirs(parent_directory, exist_ok=True)
    timestamp = datetime.now().strftime('%H-%M_%d-%m-%Y')
    name_directory = 'df'+str(suffixe)+timestamp
    path = os.path.join(parent_directory, name_directory)
    os.makedirs(path, exist_ok=True)
    c=0
    for EY in tqdm(motifs):
        c+=1
        nb_t = len(EY)
        motif = []
        X_T0 = generate_ind_NRA(motif,n=nb_ind_per_t,k=NRA)
        X_T0['NRA'] = 0
        X_T0['T'] = 0
        train_final = X_T0.copy()
        for j in range(nb_t-1):
            motif = [[1,1],[1,2],[2,1],[2,2],[3,1],[3,2],[1,3],[2,3]]
            X_T0 = generate_ind_NRA(motif,n=nb_ind_per_t,k=NRA)
            X_T0['NRA'] = 0
            X_T0['T'] = j+1
            train_final = pd.concat([train_final,X_T0], ignore_index=True)
        train_final  = gen_synt_E_Y(train_final,EY[0],name_col='E_y|T0')
        for j in range(nb_t-1):
            train_final  = gen_synt_E_Y(train_final,EY[j+1],name_col='E_y|T'+str(j+1))
            train_final = calculate_uplift(train_final,'E_y|T'+str(j+1),'E_y|T0',tau_name='tau_'+str(j+1)+'_0')
        train_final = create_column_Y(train_final)
        if titre!= '':
            csv_path = os.path.join(path, titre)
        else:
            csv_path = os.path.join(path, 'dataset'+str(c)+'.csv')
        train_final.to_csv(csv_path, index=False)
        if plot:
            for i in range(nb_t):
                X1 = train_final['X1']
                X2 = train_final['X2']
                E_y_T = train_final['E_y|T'+str(i)]
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, X2, c=E_y_T, cmap='viridis')
                plt.colorbar(label='E_y_T0')
                plt.title('Distribution de E_y_T'+str(i)+'en fonction de X1 et X2')
                plt.xlabel('X1')
                plt.ylabel('X2')
                plt.show()

def random_motif(dim=1, min_val=0, max_val=1, nb_t=3,variance=0):
    MOTIFS = []
    for i in range(nb_t):
        liste=[]
        motifs = []
        for _ in range(dim):
            liste=[]
            for _ in range(dim):
                liste.append([np.random.uniform(min_val, max_val),np.random.uniform(min_val, max_val),np.random.uniform(min_val, max_val),np.random.uniform(min_val, max_val),variance])
            motif = liste
            motifs.append(motif)
        motifs = np.array(motifs)
        MOTIFS.append(motifs)

    return MOTIFS

def generer_motif_random(valeur_min_int, valeur_max_int, nb_inter):
    motif = []
    for i in range(valeur_min_int, valeur_max_int + 1):
        for j in range(valeur_min_int, valeur_max_int + 1):
            motif.append([i, j])
    random.shuffle(motif)  # Mélange les paires de manière aléatoire
    return motif[:nb_inter]