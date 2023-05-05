# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:23:28 2023

@author: Akshays
"""

import pyodbc
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import altair as alt

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
# time series - statsmodels
from statsmodels.tsa.filters.hp_filter import hpfilter  # Hodrick Prescott filter for cyclic & trend separation
from statsmodels.tsa.seasonal import seasonal_decompose # Error Trend Seasonality decomposition
from pmdarima import auto_arima
# holt winters 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   # single exponential smoothing as in ewm of pandas
from statsmodels.tsa.holtwinters import ExponentialSmoothing # double and triple exponential smoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error,mean_squared_error

def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


def check_stationarity(series):
   

    result = adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")
        
st.header('DEMAND FORECASTING')

st.image('forecasting.jpg',use_column_width=True)

sb = st.selectbox('**Select the plant**',('Select an option',5000,5001,5003,5007,5008,5009,5010,5011,5012,5013,5014,5015,5016,5017,5018,5019,5020))
if(sb=='Select an option'):
    pass


if(sb==5000):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='5000',skiprows=1)

    finaldf.set_index('Year_Month',inplace= True)
    
    finaldf.index.freq='m'
    
    train = finaldf.copy()
    arima = sm.tsa.arima.ARIMA(train['Production'], order=(2,1,1))
    predictions = arima.fit().predict()
    
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
   
    
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)
    
    train1 = train.reset_index()
    train1.head()
  
    
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)

if(sb==5001):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='5001',skiprows=1)
    finaldf.set_index('Year_Month',inplace= True)
    finaldf.index.freq='m'
    train = finaldf.copy()
    best_model = SARIMAX( train['Production'], order=(3,0, 3), seasonal_order=(3, 0,3, 6)).fit(dis=-1)
    predictions = best_model.predict()
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
    
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)
    
    train1 = train.reset_index()
    train1.head()
  
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)
    
if(sb==5008):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='5008',skiprows=1)
    finaldf.set_index('Year_Month',inplace= True)
    finaldf.index.freq='m'
    finaldf['Production']=np.log(finaldf.Production)
    train = finaldf.copy()
    train['Dum']=train['exog_over']+train['exog_under']
    exogenous_features = train[['Dum']]
    best_model = SARIMAX( train['Production'], order=(2,0,5),exog=exogenous_features, seasonal_order=(5, 0,5, 6)).fit(dis=-1)
    predictions = best_model.predict()
    
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
    
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)
    
    train1 = train.reset_index()
    train1.head()
  
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)
    
if(sb==5009):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='5009',skiprows=1)
    finaldf.set_index('Year_Month',inplace= True)
    finaldf.index.freq='m'
    finaldf['Production']=np.log((finaldf.Production))
    train = finaldf.copy()
    best_model = SARIMAX( train['Production'], order=(6,0,5 ), seasonal_order=(6,0,5, 8)).fit(dis=-1)
    predictions = best_model.predict()
    
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
    
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)
    
    train1 = train.reset_index()
    train1.head()
  
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)

if(sb==5013):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='50013',skiprows=1)
    finaldf.set_index('Year_Month',inplace= True)
    finaldf.index.freq='m'
    train = finaldf.copy()
    arima = sm.tsa.arima.ARIMA(train['Production'], order=(5,1,2))
    stepwise_fit = auto_arima(train['Production'], trace=True,suppress_warnings=True)
    predictions = arima.fit().predict()
    
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
    
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)
    
    train1 = train.reset_index()
    train1.head()
  
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)

if(sb==5014):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='50014',skiprows=1)
    finaldf.set_index('Year_Month',inplace= True)
    finaldf.index.freq='m'
    train = finaldf.copy()
    best_model = SARIMAX( train['Production'], order=(2,0,5), seasonal_order=(2,0,5, 12)).fit(dis=-1)
    predictions = best_model.predict()
    
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
    
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)
    
    train1 = train.reset_index()
    train1.head()
  
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)

if(sb==5018):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='50018',skiprows=1)
    finaldf.set_index('Year_Month',inplace= True)
    finaldf.index.freq='m'
    finaldf['Production']=np.log((finaldf.Production))
    train = finaldf.copy()
    best_model = SARIMAX( (train['Production']), order=(5,0,6), seasonal_order=(0,0,0,8)).fit(dis=-1)
    predictions = best_model.predict()
    
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
    print(mape1)
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)
    
    train1 = train.reset_index()
    train1.head()
  
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)
    
if(sb==5019):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='50019',skiprows=1)
    finaldf.set_index('Year_Month',inplace= True)
    finaldf.index.freq='m'
    finaldf['Production']=np.log((finaldf.Production))
    train = finaldf.copy()
    best_model = SARIMAX( np.sqrt(train['Production']), order=(4,0,5), seasonal_order=(0,0,0,3)).fit(dis=-1)
    predictions = best_model.predict()
    
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
    #print(mape1)
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)
    
    train1 = train.reset_index()
    train1.head()
  
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)
    
if(sb==5020):
    finaldf = pd.read_excel('Forecast Data.xlsx',sheet_name='50020',skiprows=1)
    finaldf.set_index('Year_Month',inplace= True)
    finaldf.index.freq='m'
    finaldf['Production']=np.log((finaldf.Production))
    train = finaldf.copy()
    best_model = SARIMAX( train['Production'], order=(5,0,5), seasonal_order=(5,0,5, 12)).fit(dis=-1)
    predictions = best_model.predict()
    
    train['prediction'] = predictions.values
    mape1 = mape(train['Production'], train['prediction'])
    print(mape1)
    st.dataframe(finaldf)
    
    st.write('MAPE:',mape1)    
    train1 = train.reset_index()
    train1.head()
  
    import plotly.graph_objs as go
    st.subheader('Actual vs Predicted')
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['Production'], name='Actual Production', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train1['Year_Month'], y=train1['prediction'], name='Predicted Production', line=dict(color='orange')))
    st.plotly_chart(fig)
    

        
    
    


    

    

    
    
    