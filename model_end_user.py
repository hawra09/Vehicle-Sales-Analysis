import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

load_model = joblib.load('linear_regression_model.pkl')
MonthEncoder = joblib.load('month_encoder.pkl')
MM_Scaler = joblib.load('min_max_scaler.pkl')
'''
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
sales_2023 = pd.read_csv('vehicle_sales_in_2023.csv')
refined_2023 = sales_2023[['Year', 'Months', 'New_MinMax', 'Used_MinMax', 'Total_Used_MinMax']]
y_2023 = sales_2023[['Total_New_MinMax']]
prediction_2023 = load_model.predict(refined_2023)
'''


def sales_prediction():
    print('Welcome to the Total New Sales Predictor for Vehicles:')
    year = (int(input('Please enter a year from 2002 to 2023:')))
    month = (input('Please enter a month using three letter notion, such as:'
                   '\n \'Jan\':')).strip().upper()
    new_vehicles = int(input('Please enter New vehicles:'))
    used_vehicles = int(input('Please enter Used vehicles:'))
    total_new_sales = 0
    total_used_sales = int(input('Please enter the Total Used Sales:'))
    prediction_values = {'Year': year,
                         'Month': month,
                         'New': new_vehicles,
                         'Used': used_vehicles,
                         'Total Sales New': total_new_sales,
                         'Total Sales Used': total_used_sales}
    vehicle_sales_count = pd.DataFrame([prediction_values])
    vehicle_sales_count['Months'] = MonthEncoder.fit_transform(vehicle_sales_count[['Month']])
    vehicle_sales_count['New_MinMax'] = MM_Scaler.fit_transform(vehicle_sales_count[['New']])
    vehicle_sales_count['Used_MinMax'] = MM_Scaler.fit_transform(vehicle_sales_count[['Used']])
    vehicle_sales_count['Total_New_MinMax'] = MM_Scaler.fit_transform(vehicle_sales_count[['Total Sales New']])
    vehicle_sales_count['Total_Used_MinMax'] = MM_Scaler.fit_transform(vehicle_sales_count[['Total Sales Used']])
    vehicle_sales_count = vehicle_sales_count[['Year', 'Months', 'New_MinMax', 'Used_MinMax', 'Total_Used_MinMax']]
    y_prediction = load_model.predict(vehicle_sales_count)
    reverse_prediction = MM_Scaler.inverse_transform(y_prediction).reshape(-1,1)
    print(f"Predicted Total New Sales: {reverse_prediction[0][0]}")


sales_prediction()
