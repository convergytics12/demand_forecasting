# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:05:31 2023

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
#from pmdarima import auto_arima
# holt winters 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   # single exponential smoothing as in ewm of pandas
from statsmodels.tsa.holtwinters import ExponentialSmoothing # double and triple exponential smoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error,mean_squared_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go
def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = (np.mean(np.abs((y_test - pred) / y_test)))*100
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
file1 = st.file_uploader('Upload a file')
if file1 is not None:
    sb = st.selectbox('**Select the Plant**',('Select an option',5001,5007,5008,5009,5010,5011,5012,5013,5014,5015,5016,5017,5018,5019,5020))
    finaldf = pd.read_excel(file1)
    if(sb==5001):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(finaldf['Production'], order=(5, 0, 4), seasonal_order=(5, 0, 4, 12)).fit(dis=-1) 
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    if(sb==5007):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(finaldf['Production'], order=(3, 1, 3), seasonal_order=(3, 0, 0, 12)).fit(dis=-1) 
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    
    if(sb==5010):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(finaldf['Production'], order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)).fit(dis=-1) 
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    if(sb==5014):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(finaldf['Production'], order=(3, 0, 4), seasonal_order=(0, 0, 4, 5)).fit(dis=-1) 
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    
    if(sb==5018):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(finaldf['Production'], order=(2, 0, 2), seasonal_order=(2, 0, 2, 6)).fit(dis=-1) 
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    
    if(sb==5020):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(finaldf['Production'], order=(6, 0, 4), seasonal_order=(6, 0, 5, 12)).fit(dis=-1) 
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    
    if(sb==5019):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(np.log(finaldf['Production']), order=(7, 0, 5), seasonal_order=(7, 0, 5, 10)).fit(dis=-1) 
        pred = np.exp(model1.predict())
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    
    if(sb==5017):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(np.log(finaldf['Production']), order=(5, 0, 4), seasonal_order=(5, 0, 4, 12)).fit(dis=-1) 
        pred = np.exp(model1.predict())
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    
    if(sb==5016):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(np.log(finaldf['Production']), order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)).fit(dis=-1) 
        pred = np.exp(model1.predict())
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
        
        st.subheader('**Material Level Forecasting**')
        d = pd.read_excel("data1.xlsx",sheet_name='Data_new')
        sb1 = st.selectbox('**Select the Material no**',('Select an option',5016410000020,5016410000064,5016410000067,5016410000068,5016410000101,5016410000196,
                                                         5016410000209,5016410000225,5016410000226,5016410000241,5016410000242,5016410000270,5016410000271,
                                                         5016410000272,5016410000307,5016410000420,5016410000421,5016410000470,5016410001348,5016410001349,
                                                         5016410001351,5016410001533,5016420000009,5016420000030,5016420000043,5016410000082,5016410000295))
        list1 = [5016410000020,5016410000064,5016410000067,5016410000068,5016410000101,5016410000196,5016410000209,
                 5016410000225,5016410000226,5016410000241,5016410000242,5016410000270,5016410000271,5016410000272,
                 5016410000307,5016410000420,5016410000421,5016410000470,5016410001348,5016410001349,5016410001351,
                 5016410001533,5016420000009,5016420000030,5016420000043,5016410000082,5016410000295]
        if(sb1=='Select an option'):
            pass
        
        if sb1 in list1:
            d2 = d[d['SKU'].isin([sb1])]
            st.subheader('Information Provided')
            d10 = d2
            d10 = d10.rename(columns={'SKU':'Material no'})
            d10['Material no'] = d10['Material no'].astype('str')
            st.dataframe(d10)
            d2 = d2.reset_index(drop=True)
            d2['Yr-mt'] = pd.to_datetime(d2['Yr-mt'])
            d2['Year'] = d2['Yr-mt'].dt.year
            d2['Value'] = d2['Value'].astype(int)
            yearly_mean = d2.groupby('Year')['Value'].mean().reset_index()
            d3 = d2.merge(yearly_mean, on='Year', suffixes=('', '_mean'))
            d3['Seasonality_Avg'] = d3['Value']/d3['Value_mean']
            d3 = d3.set_index('Yr-mt')
            
            st.subheader('Seasonal Decompose')
            result = seasonal_decompose(x=d3['Value'], model='additive')

            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
            )
            
            fig.add_trace(
                go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
                row=4, col=1
            )
            
            fig.update_layout(height=600, width=1000)
            st.plotly_chart(fig, use_container_width=True)
            
            d4 = pd.read_excel('5016_sku_order.xlsx')
            meal_id = list(d4[d4['SKU']==sb1]['SKU'])[0]
            d_final = d3[d3['SKU']==meal_id]
            exogenous_features=pd.DataFrame(d3[['Seasonality_Avg']])
            d_final.dropna(inplace=True)
            d_final.reset_index(inplace=True)
            d_final=d_final.set_index('Yr-mt')
            d_final.index=pd.to_datetime(d_final.index)
            exog = d_final[['Seasonality_Avg']]
            order = list(d4[d4['SKU']==sb1]['order'])[0]
            order = str(order,)
            order = order.replace("(","")
            order = order.replace(")","")
            order = order.replace(","," ")
            order = order.replace("'","")
            order = order.split()
            order = list(map(int,order))
            order = tuple(order)
            model=sm.tsa.statespace.SARIMAX(d_final['Value'],order=order,exog=exog)
            model=model.fit()
            d_final['Forecast'] = model.predict(start=0,end=len(d_final)-1,dynamic=False)
            d_final = d_final.reset_index()
           
            
            mape2 = mape(d_final['Value'],d_final['Forecast'])
            st.write('MAPE:',round(mape2,2))
            forecast_sku=model.predict(start=len(d_final), end=71,exog=exog)
            forecast_sku=forecast_sku[:6]
            d_forecast=pd.DataFrame(data=list(forecast_sku.values),columns=['Forecasted SKU Production'],index=pd.Series((forecast_sku.index.values)))
            d_forecast=dff.reset_index().rename(columns={'index':'Date'})
            d_forecast=pd.DataFrame(data=list(forecast_sku.values),columns=['Forecasted SKU Production'],index=pd.Series((forecast_sku.index.values)))
            d_forecast=d_forecast.reset_index().rename(columns={'index':'Date'})
            
            a=pd.DataFrame(data=d_final.iloc[-1]['Forecast'],columns=['Forecasted SKU Production'],index=pd.Series(d_final['Yr-mt'].iloc[-1]))
            a=a.reset_index().rename(columns={'index':'Date'})
            
            d_forecast=pd.concat([a,d_forecast],ignore_index=True)
           
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d_final['Yr-mt'], y=d_final['Value'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=d_final['Yr-mt'], y=d_final['Forecast'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=d_forecast['Date'], y=d_forecast['Forecasted SKU Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
            
            
            

    
    if(sb==5009):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd['Year_Month'], y=dd['Production'], name='Production', line=dict(color='blue')))
        st.plotly_chart(fig)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        df=finaldf
             
        
        
        
        st.subheader('Seasonal Decompose')
        result = seasonal_decompose(x=df['Production'], model='multiplicative',)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name="Trend"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name="Seasonal"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('Actual Production Vs Predicted Production')
        model1 = SARIMAX(np.log(finaldf['Production']), order=(5, 0, 2), seasonal_order=(5, 0, 2, 12)).fit(dis=-1) 
        pred = np.exp(model1.predict())
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        df = df[:-1]
        a=pd.DataFrame(df[['Year_Month','Production']].iloc[34]).T
        a.rename(columns={'Year_Month':'Date','Production':'Forecasted Production'},inplace=True)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',round(mape1,2))
        start = len(df)
        end=48
        fore=model1.predict(start = start,end = end)
        fore = fore.shift(periods=-1)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        dff = pd.concat([a,dff],ignore_index=True)
        dff = dff[:-1]
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
            st.plotly_chart(fig)
            
       
        excel_button = st.button('Download as Excel')
        if excel_button:
            writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
            dff.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()
            with open('data.xlsx', 'rb') as f:
                excel_data = f.read()
                st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    
    
                    
    
        
    
            
        
        
        
        
        
        
        
        
        
        


        
    
    

