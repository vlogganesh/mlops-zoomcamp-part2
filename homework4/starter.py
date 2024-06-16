#!/usr/bin/env python
# coding: utf-8
##Please run the script with the command shown below.
#python3 starter.py 2023 04 for q5

import pickle
import pandas as pd
import sys
def read_data(filename):
    df = pd.read_parquet(filename)
    categorical = ['PULocationID', 'DOLocationID']
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df




year=sys.argv[1]
month=sys.argv[2]

def run():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
        categorical = ['PULocationID', 'DOLocationID']
        df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')
        dicts = df[categorical].to_dict(orient='records')
        X_val = dv.transform(dicts)
        y_pred = model.predict(X_val)
        df['predicted_duration']=y_pred
        df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
        df_result=pd.DataFrame()
        df_result['ride_id']=df['ride_id']
        df_result['predicted_duration']=df['predicted_duration']
        return df_result['predicted_duration'].mean()

if __name__ =="__main__":
    print(run())