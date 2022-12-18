from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib import messages

import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import ExtraTreesRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# import sklearn.metrics
# from pylab import rcParams

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Create your views here.
def home(request):
    model,premodel=get_model()
    x_scaled=get_input(premodel)
    a=model.predict(x_scaled)
    return render(request,"home.html",{'a':a})
def preprocess(df):
    df['fraud_reported'].replace(to_replace='Y', value=1, inplace=True)
    df['fraud_reported'].replace(to_replace='N',  value=0, inplace=True)
    df['csl_per_person'] = df.policy_csl.str.split('/', expand=True)[0]
    df['csl_per_accident'] = df.policy_csl.str.split('/', expand=True)[1]
    df['vehicle_age'] = 2018 - df['auto_year']
    bins = [-1, 3, 6, 9, 12, 17, 20, 24]
    names = ["past_midnight", "early_morning", "morning", 'fore-noon', 'afternoon', 'evening', 'night']
    df['incident_period_of_day'] = pd.cut(df.incident_hour_of_the_day, bins, labels=names).astype(object)
    df = df.drop(columns = [
        'policy_number', 
        'policy_csl',
        'insured_zip',
        'policy_bind_date', 
        'incident_date', 
        'incident_location', 
        '_c39', 
        'auto_year', 
        'incident_hour_of_the_day'])
    return df

def returnOneHotEncObj(df):
    premodel=OneHotEncoder(sparse=False,handle_unknown="ignore")
    premodel.fit(df[[
        'policy_state', 
        'insured_sex', 
        'insured_education_level',
        'insured_occupation', 
        'insured_hobbies', 
        'insured_relationship',
        'incident_type', 
        'incident_severity',
        'authorities_contacted', 
        'incident_state', 
        'incident_city',
        'auto_make', 
        'auto_model', 
        'csl_per_person', 
        'csl_per_accident',
        'incident_period_of_day']])
    return premodel
def transform_OneHotModel(premodel,df):
    dummies=pd.DataFrame(premodel.transform(df[[
        'policy_state', 
        'insured_sex', 
        'insured_education_level',
        'insured_occupation', 
        'insured_hobbies', 
        'insured_relationship',
        'incident_type', 
        'incident_severity',
        'authorities_contacted', 
        'incident_state', 
        'incident_city',
        'auto_make', 
        'auto_model', 
        'csl_per_person', 
        'csl_per_accident',
        'incident_period_of_day']]),columns=premodel.get_feature_names_out())
    dummies = dummies.join(df[[
        'collision_type', 
        'property_damage', 
        'police_report_available', 
        "fraud_reported"]])
    return dummies
def label_Encodeing_Scaling(X,dummies,df):
    X['collision_en'] = LabelEncoder().fit_transform(dummies['collision_type'])
    X[['collision_type', 'collision_en']]
    X['property_damage'].replace(to_replace='YES', value=1, inplace=True)
    X['property_damage'].replace(to_replace='NO', value=0, inplace=True)
    X['property_damage'].replace(to_replace='?', value=0, inplace=True)
    X['police_report_available'].replace(to_replace='YES', value=1, inplace=True)
    X['police_report_available'].replace(to_replace='NO', value=0, inplace=True)
    X['police_report_available'].replace(to_replace='?', value=0, inplace=True)
    X = X.drop(columns = ['collision_type'])
    X = pd.concat([X, df._get_numeric_data()], axis=1)
    X = X.drop(columns = ['fraud_reported'])
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns = X.columns)
    X_train_scaled = pd.DataFrame.to_numpy(X_train_scaled)
    return X_train_scaled


def get_model():
    df = pd.read_csv('Main/insurance_claims.csv')
    df=preprocess(df)
    premodel=returnOneHotEncObj(df)
    dummies=transform_OneHotModel(premodel,df)
    X = dummies.iloc[:, 0:-1]
    y = dummies.iloc[:, -1] 
    X_scaled=label_Encodeing_Scaling(X,dummies,df)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled,y)
    return lda,premodel

def get_input(premodel):
    ip = pd.read_csv('Main/new data.csv')
    ip=preprocess(ip)
    dummies=transform_OneHotModel(premodel,ip)
    X = dummies.iloc[:, 0:-1]
    X_scaled=label_Encodeing_Scaling(X,dummies,ip)
    return X_scaled
    

    

