import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL="clf_predicter.pkl"
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def Preprocessing(data):
    data = remove_outliers_iqr(data, ['Area', 'Rate per sqft', 'BHK_Count'])
    data=data[data['Flat Type'] != 'Penthouse']
    
    data['mult']=data['Area']*data['Rate per sqft']
    data=data.drop(['Area','Rate per sqft','Property Type','Socity','Builder Name','Company Name','Locality'],axis=1)
    binss=[0,10000000,20000000,30000000,40000000,50000000,np.inf]
    labelss=[1,2,3,4,5,6]
    data['mult_cat']=pd.cut(data["mult"],bins=binss,labels=labelss)#spliting column

    #Stratified Split
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_idx,test_idx in split.split(data,data["mult_cat"]):
        strat_train_set=data.iloc[train_idx]
        strat_test_set=data.iloc[test_idx]

    #Spliting features and labels
    strat_test_set=strat_test_set.drop('mult_cat',axis=1)
    strat_train_set=strat_train_set.drop('mult_cat',axis=1)
    train_labels=strat_train_set['Price']
    trainer=strat_train_set.drop('Price',axis=1)

    num_attrs=['BHK_Count','mult']
    cat_attr=['Status','RERA Approval','Flat Type']
    full_pipeline=ColumnTransformer([
        ("num",StandardScaler(),num_attrs),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_attr)
    ])
    return full_pipeline.fit_transform(data),train_labels

if not os.path.exists(MODEL):
    data=pd.read_csv("data of gurugram real Estate.csv")
    trainer_prepared,train_labels=Preprocessing(data)
    RanReg=RandomForestRegressor(random_state=42)
    RanReg.fit(trainer_prepared,train_labels)
    joblib.dump(RanReg,"ggn_predictor.pkl")
    print("Model Saved!")

else:
    model=joblib.load(MODEL)

    input=pd.read_csv("data of gurugram real Estate.csv")
    input_processed,labels=Preprocessing(input)
    predictions=model.predict(input_processed)
    input['expected']=labels
    input['predicted']=predictions
    output=input.drop('median_house_value',axis=1).copy()
    output.to_csv(predictions.csv)
