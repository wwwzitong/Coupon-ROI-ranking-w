import seaborn as sns
import matplotlib
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
import random
from scipy.stats import norm
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import preprocessing
import kuplift as kp
import warnings
import importlib
import subprocess
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from modèles import *
from évaluation import *
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
warnings.filterwarnings('ignore')
import string
import time
from sklearn.isotonic import IsotonicRegression
from multiprocessing import Pool
import json
from datetime import datetime
import ray
import os

@ray.remote
def generate_for_t(nb_t,nb_df,nb_ind_per_t):
    #motifs=generate_plein_de_motifs(nb=1,dim=1,nb_t=nb_t,type='plein')
    #generate_plein_de_datasets(motifs,plot=False)
    
    motifs=generate_plein_de_motifs(nb=1,dim=1,nb_t=nb_t,type='diag')
    generate_plein_de_datasets(motifs,plot=False)
    
    motifs=generate_plein_de_motifs(nb=1,dim=1,nb_t=nb_t,type='droit')
    generate_plein_de_datasets(motifs,plot=False)
    
    motifs=generate_plein_de_motifs(nb=1,dim=1,nb_t=nb_t,type='trois')
    generate_plein_de_datasets(motifs,plot=False)
    
    
    motifs=generate_plein_de_motifs(nb=1,dim=1,nb_t=nb_t,type='un')
    generate_plein_de_datasets(motifs,plot=False,nb_ind_per_t=nb_ind_per_t)


@ray.remote
def generate_plein_de_datasets(motifs, plot=False, nb_ind_per_t=10000, NRA=0, suffixe='', suffixe_motif='', titre=''):
    parent_directory = "/root/autodl-tmp/Coupon-ROI-ranking-w/0510-non-RCT/data"
    os.makedirs(parent_directory, exist_ok=True)

    timestamp = datetime.now().strftime('%H-%M_%d-%m-%Y')
    name_directory = 'df' + str(suffixe) + timestamp
    path = os.path.join(parent_directory, name_directory)
    os.makedirs(path, exist_ok=True)

    print("当前工作目录:", os.getcwd())
    print("输出目录:", path)

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
            csv_path = os.path.join(path, titre+str(c)+'.csv')
        else:
            csv_path = os.path.join(path, 'dataset'+str(c)+'.csv')
        train_final.to_csv(csv_path, index=False)
        print("已保存:", csv_path)
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



ray.init(ignore_reinit_error=True)

refs = []

# nb_t_values = [3, 5, 9, 17, 33]
nb_t_values = [1]
val_values = [[0,1], [0.2,0.8], [0.4,0.6]]

for nb_t in tqdm(nb_t_values, desc="Progression nb_t"):
    for val in val_values:
        motif = generate_plein_de_motifs(
            nb=10,
            dim=10,
            nb_t=nb_t,
            type='quadrillage',
            prob=0.25,
            min_val=val[0],
            max_val=val[1]
        )

        titre = str(nb_t) + "_" + str(val[0]) + str(val[1])
        titre = titre.replace(".", "")

        ref = generate_plein_de_datasets.remote(
            motif,
            nb_ind_per_t=20000,
            plot=False,
            titre=titre
        )
        refs.append(ref)

ray.get(refs)
print("全部 CSV 生成完成")
        