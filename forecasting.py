# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:21:55 2023

@author: Convergytics
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
    sb = st.selectbox('**Select the Plant**',('Select an option',5000,5001,5003,5007,5008,5009,5010,5011,5012,5013,5014,5015,5016,5017,5018,5019,5020))
    finaldf = pd.read_excel(file1)
    if(sb==5000):
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(2, 1, 1))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            st.write(df['Year_Month'].dtype)
            st.write(dff['Date'].dtype)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
    
    if(sb==5001):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(1, 1, 1))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5003):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(1, 1, 0))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
    if(sb==5007):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(2,1,1))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5008):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(1, 1, 1))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5010):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(1, 1, 0))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5011):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.statespace.SARIMAX(df['Production'],order=(4,0,1),seasonal_order=(0,0,0,6))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5013):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(5, 1, 2))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5015):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(1, 1, 0))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
    
    if(sb==5016):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(0, 1, 3))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5017):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(1, 1, 0))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5019):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.arima.ARIMA(df['Production'], order=(1, 1, 0))  
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    if(sb==5020):
        x=finaldf[finaldf['Plant']==sb][['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']]
        dd=pd.DataFrame(x.sum())
        dd=dd.reset_index()
        dd.rename(columns={'index':'Year_Month',0:'Production'},inplace=True)
        dd=dd.set_index('Year_Month')
        dd.index=pd.to_datetime(dd.index)
        finaldf=dd
        st.subheader('Information Provided')
        st.dataframe(finaldf)
        st.line_chart(finaldf)
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
        model = sm.tsa.statespace.SARIMAX(df['Production'],order=(5,0,5),seasonal_order=(5,0,5,6)) 
        model1 = model.fit()
        pred = model1.predict()
        df['prediction'] = pred.values
        df.reset_index(inplace=True)
        df['prediction'] = df['prediction'].shift(periods=-1)
        df['prediction'] = df['prediction'].replace(np.nan, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['prediction'], name='Predicted Production', line=dict(color='orange')))
        st.plotly_chart(fig)
        
        mape1 = mape(df['Production'], df['prediction'])
   
        st.write('MAPE:',mape1)
        fore=model1.forecast(steps=6)
        dff=pd.DataFrame(data=list(fore.values),columns=['Forecasted Production'],index=pd.Series((fore.index.values)))
        dff=dff.reset_index().rename(columns={'index':'Date'})
        
        if(st.button('Start Forecasting')):
            st.subheader('Forecasted Production')
            fig = go.Figure()
            #fig.add_trace(go.Scatter(x=df['Year_Month'], y=df['Production'], name='Actual Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='orange')))
            st.plotly_chart(fig)
            
       
            rad=st.radio('**Export Forecasted Production**',['.csv','.xlsx'])
            if(st.button('Download')):
                if rad=='.csv':
                    dff.to_csv(str(sb)+'_forecasted_data.csv')
                if rad=='.xlsx':
                    dff.to_excel(str(sb)+'_forecasted_data.xlsx')
                    
    
        
    
            
        
        
        
        
        
        
        
        
        
        


        
    
    

