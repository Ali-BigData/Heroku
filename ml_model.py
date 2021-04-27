import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from scipy import  stats 
from sklearn.ensemble import GradientBoostingRegressor
import pickle


def rename(df):

    df = df.rename(columns={'Cement (kg per  m^3 of  mixture)':'Cement','Water  (kg per  m^3 of  mixture)':'Water','Coarse Aggregate  (kg per  m^3 of  mixture)':'Coarse Aggregate','Fly Ash (kg per  m^3 of  mixture)': 'Fly Ash','Superplasticizer (kg per  m^3 of  mixture)':'Superplasticizer',
                                 'Fine Aggregate (kg per  m^3 of  mixture)':'Fine Aggregate','Concrete compressive strength(MPa, megapascals) ':'Concrete Strenght',
                                 'Blast Furnace Slag (kg per  m^3 of  mixture)':'Blast Furnace Slag','Age (day)':'Age'},inplace = False)
    return df


def extract_y(df):
    return (df['Concrete Strenght'])

# Feature Engineering
def extract_x(df):
    for col in df.columns:
        if col== 'Concrete Strenght':
                 
            X = df.drop(columns=col)
        else:
            X = df
   

# Scalling data
    scaler = StandardScaler()

    scaler.fit(X)
    X = scaler.transform(X)
# convet it to dataframe.
    X = pd.DataFrame(columns= df.columns[:8], data=X)
    return(X)


def predict_concrete(config,model):
    if type(config)==dict:
        df = pd.DataFrame(config)
    else:
        df = config
        
    x = extract_x(df)
  
    y_pred = model.predict(x)
    return y_pred
        
        