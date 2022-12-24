from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Create your views here.
def home(request):
    return render(request,"home.html")

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
    if request.method=="POST":
        months_as_customer=int(request.POST["months_as_customer"])
        policy_number=int(request.POST["policy_number"])
        policy_state=request.POST["policy_state"]
        policy_deductable=int(request.POST["policy_deductable"])
        umbrella_limit=int(request.POST["umbrella_limit"])
        insured_sex=request.POST["insured_sex"]
        insured_occupation=request.POST["insured_occupation"]
        insured_relationship=request.POST["insured_relationship"]
        capital_loss=-1*int(request.POST["capital_loss"])
        incident_type=request.POST["incident_type"]
        incident_severity=request.POST["incident_severity"]
        incident_state=request.POST["incident_state"]
        incident_location=request.POST["incident_location"]
        number_of_vehicles_involved=int(request.POST["number_of_vehicles_involved"])
        bodily_injuries=int(request.POST["bodily_injuries"])
        police_report_available=request.POST["police_report_available"]
        injury_claim=int(request.POST["injury_claim"])
        vehicle_claim=int(request.POST["vehicle_claim"])
        auto_model=request.POST["auto_model"]
        age=int(request.POST["age"])
        policy_bind_date=request.POST["policy_bind_date"]
        policy_csl=request.POST["policy_csl"]
        policy_annual_premium=float(request.POST["policy_annual_premium"])
        insured_zip=int(request.POST["insured_zip"])
        insured_education_level=request.POST["insured_education_level"]
        insured_hobbies=request.POST["insured_hobbies"]
        capital_gains=int(request.POST["capital_gains"])
        incident_date=request.POST["incident_date"]
        collision_type=request.POST["collision_type"]
        authorities_contacted=request.POST["authorities_contacted"]
        incident_city=request.POST["incident_city"]
        incident_hour_of_the_day=int(request.POST["incident_hour_of_the_day"])
        property_damage=request.POST["property_damage"]
        witnesses=int(request.POST["witnesses"])
        total_claim_amount=int(request.POST["total_claim_amount"])
        property_claim=int(request.POST["property_claim"])
        auto_make=request.POST["auto_make"]
        auto_year=int(request.POST["auto_year"])
        _c39=float(request.POST["_c39"])
        fraud_reported="-"
        data={
            "months_as_customer":months_as_customer,
            "age":age,
            "policy_number":policy_number,
            "policy_bind_date":policy_bind_date,
            "policy_state":policy_state,
            "policy_csl":policy_csl,
            "policy_deductable":policy_deductable,
            "policy_annual_premium":policy_annual_premium,
            "umbrella_limit":umbrella_limit,
            "insured_zip":insured_zip,
            "insured_sex":insured_sex,
            "insured_education_level":insured_education_level,
            "insured_occupation":insured_occupation,
            "insured_hobbies":insured_hobbies,
            "insured_relationship":insured_relationship,
            "capital-gains":capital_gains,
            "capital-loss":capital_loss,
            "incident_date":incident_date,
            "incident_type":incident_type,
            "collision_type":collision_type,
            "incident_severity":incident_severity,
            "authorities_contacted":authorities_contacted,
            "incident_state":incident_state,
            "incident_city":incident_city,
            "incident_location":incident_location,
            "incident_hour_of_the_day":incident_hour_of_the_day,
            "number_of_vehicles_involved":number_of_vehicles_involved,
            "property_damage":property_damage,
            "bodily_injuries":bodily_injuries,
            "witnesses":witnesses,
            "police_report_available":police_report_available,
            "total_claim_amount":total_claim_amount,
            "injury_claim":injury_claim,
            "property_claim":property_claim,
            "vehicle_claim":vehicle_claim,
            "auto_make":auto_make,
            "auto_model":auto_model,
            "auto_year":auto_year,
            "fraud_reported":fraud_reported,
            "_c39":_c39
        }    
        inputdata=pd.DataFrame(data,index=[0])
        pred=predict(inputdata) 
        data=[]
        for i in range(len(pred)):
            data.append({"Sno":i+1,"polNo":policy_number,"pred":pred[i]!=1})
    return render(request,"prediction.html",{"data":data})

def predict(data):
    model,premodel=get_model()
    data=preProcessIP(data,premodel)
    pred=model.predict(data)
    return pred

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

def fileUpload(request):
    folder='insurance_files/' 
    if request.method == 'POST' and request.FILES['insurance_file']:
        myfile = request.FILES['insurance_file']
        fs = FileSystemStorage(location=folder) #defaults to   MEDIA_ROOT  
        filename = fs.save(myfile.name, myfile)
        data=predectFromFile(filename)
        
         
        return render(request, 'prediction.html',{"data":data})
    else:
         return render(request, 'home.html')

def predectFromFile(filename):
    model,premodel=get_model()
    x_scaled,ip=get_input(premodel,filename)  
    pred=model.predict(x_scaled)
    polNO=list(ip['policy_number'])
    data=[]
    for i in range(len(pred)):
        data.append({"Sno":i+1,"polNo":polNO[i],"pred":pred[i]!=1})
    return data 

def preProcessIP(ip,premodel):
    ip=preprocess(ip)
    dummies=transform_OneHotModel(premodel,ip)
    X = dummies.iloc[:, 0:]
    X_scaled=label_Encodeing_Scaling(X,dummies,ip)
    return X_scaled

def get_input(premodel,filename):
    ip=pd.read_csv('insurance_files/'+filename) 
    return preProcessIP(ip,premodel),ip
    

    

