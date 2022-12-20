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

def getInsureFormDetails(request):
    hobbies=['sleeping', 'reading', 'board-games', 'bungie-jumping',
       'base-jumping', 'golf', 'camping', 'dancing', 'skydiving',
       'movies', 'hiking', 'yachting', 'paintball', 'chess', 'kayaking',
       'polo', 'basketball', 'video-games', 'cross-fit', 'exercise']
    occupations=['craft-repair', 'machine-op-inspct', 'sales', 'armed-forces',
       'tech-support', 'prof-specialty',
       'priv-house-serv', 'exec-managerial', 'protective-serv',
       'transport-moving', 'handlers-cleaners', 'adm-clerical',
       'farming-fishing', 'other-service']
    educationList=['MD', 'PhD', 'Associate', 'Masters', 'High School', 'College',
       'JD']
    relationList=['husband', 'other-relative', 'own-child', 'unmarried', 'wife',
       'not-in-family']
    incidentType=['Single Vehicle Collision', 'Vehicle Theft',
       'Multi-vehicle Collision', 'Parked Car']
    collosionType=['Side Collision', 'Rear Collision', 'Front Collision']
    incidentSevierity=['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage']
    authoritiesContacted=['Police', 'Fire', 'Ambulance',  'Other','None']
    incidentCity=['Columbus', 'Riverwood', 'Arlington', 'Springfield', 'Hillsdale',
       'Northbend', 'Northbrook']
    incidentState=['SC', 'VA', 'NY', 'OH', 'WV', 'NC', 'PA']
    propertyDamage=['YES', 'NO']
    policeReportAvailable=['YES', 'NO']
    autoMakers=['Saab', 'Mercedes', 'Dodge', 'Chevrolet', 'Accura', 'Nissan',
       'Audi', 'Toyota', 'Ford', 'Suburu', 'BMW', 'Jeep', 'Honda',
       'Volkswagen']
    autoModel=['92x', 'E400', 'RAM', 'Tahoe', 'RSX', '95', 'Pathfinder', 'A5',
       'Camry', 'F150', 'A3', 'Highlander', 'Neon', 'MDX', 'Maxima',
       'Legacy', 'TL', 'Impreza', 'Forrestor', 'Escape', 'Corolla',
       '3 Series', 'C300', 'Wrangler', 'M5', 'X5', 'Civic', 'Passat',
       'Silverado', 'CRV', '93', 'Accord', 'X6', 'Malibu', 'Fusion',
       'Jetta', 'ML350', 'Ultima', 'Grand Cherokee']
    autoYears=[i for i in range(1995,2023)]
    incidentHourOfDay=[i for i in range(0,24)]
    data={'hobbies':hobbies,"occupations":occupations,"educationList":educationList,"relationList":relationList,"incidentType":incidentType,
            'collosionType':collosionType,"incidentSevierity":incidentSevierity,"authoritiesContacted":authoritiesContacted,"incidentCity":incidentCity,
            "incidentState":incidentState,"incidentHourOfDay":incidentHourOfDay,"propertyDamage":propertyDamage,"policeReportAvailable":policeReportAvailable,
            "autoMakers":autoMakers,"autoModel":autoModel,"autoYears":autoYears}
    return render(request,"InsuranceForm.html",data)

def makePrediction(request):
    
    return render(request,"prediction.html")

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
    

    

