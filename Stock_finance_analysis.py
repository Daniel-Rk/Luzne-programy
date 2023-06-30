from openbb_terminal.sdk import openbb
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from core import *
import plotly.graph_objects as go

def convert_string_to_int(string):
    if 'B' in string:
        return int(float(string[:-1]) * 1e9)
    elif 'M' in string:
        return int(float(string[:-1]) * 1e6)
    elif 'K' in string:
        return int(float(string[:-1]) * 1e3)
    elif 'T' in string:
        return int(float(string[:-1]) * 1e12)
    else:
        return int(string)
    


def ln_regres(dataframe, value_to_regres):
    dataframe[dataframe.columns[0]] = pd.to_datetime(dataframe[dataframe.columns[0]], format='%Y-%m-%d')
    x = dataframe.index.values.reshape((-1, 1))
    y = dataframe[value_to_regres]
    model = LinearRegression().fit(x, y) 
    y_pred = model.predict(x)

    return y_pred

def plot(dataframe, lista, linear_regres=None):
    for i,j in enumerate(lista):
        fig = px.bar(dataframe, x=dataframe.columns[0], y=j)
        fig.update_layout(barmode='group')

        if linear_regres is True:
            fig.add_traces(go.Line( x=dataframe.fiscalDateEnding, y=ln_regres(dataframe, j) ))
        st.plotly_chart(fig)




class Stock_Finance():
    def __init__(self, stock_array = ['PZU.WA'], stock_date = '2020-01-05'):
        self.stock_array = stock_array
        self.stock_date = stock_date
        self.df = yahoo(self.stock_array).cena(start_= self.stock_date)  
        self.m = {'K': 3, 'M': 6, 'B': 9, 'T': 12}
        
    def Info(self):
        pass

    def Income(self, quartly_status=False):
        x = openbb.stocks.fa.income(self.stock_array[0], source="AlphaVantage", quarterly=quartly_status).T
        x.drop(list(x.filter(regex='reportedCurrency')), axis=1, inplace=True) 
        for column in x.columns:
            x[column] = x[column].apply(convert_string_to_int)
        return x.reset_index()

    def Balance(self, quartly_status=False):
        x = openbb.stocks.fa.balance(self.stock_array[0], source="AlphaVantage", quarterly=quartly_status).T
        x.drop(list(x.filter(regex='reportedCurrency')), axis=1, inplace=True) 
        for column in x.columns:
            x[column] = x[column].apply(convert_string_to_int)

        x.eval('WorkingCapital = totalCurrentAssets - totalCurrentLiabilities', inplace=True)
        return x.reset_index()

    def Cash_flow(self, quartly_status=False):
        x = openbb.stocks.fa.cash(self.stock_array[0], source="AlphaVantage", quarterly=quartly_status).T
        x.drop(list(x.filter(regex='reportedCurrency')), axis=1, inplace=True) 
        for column in x.columns:
            x[column] = x[column].apply(convert_string_to_int)
        return x.reset_index()
    
    def Ratios(self):
        x = openbb.stocks.fa.ratios(self.stock_array[0])
        return x
    
    def Metrics(self):
        x = openbb.stocks.fa.metrics(self.stock_array[0])
        return x
    
    def Customers(self):
        x = openbb.stocks.fa.customer(self.stock_array)
        return x
    
    def Suppliers(self):
        x = openbb.stocks.fa.supplier(self.stock_array)
        return x
    
    def Dividends(self):
        x = openbb.stocks.fa.divs(self.stock_array)
        return x
    
    def Price_Target(self):
        x = openbb.stocks.fa.pt(self.stock_array)
        return x
    
    def Score(self):
        x = openbb.stocks.fa.score(self.stock_array)
        return x
    
    def Summary_table(self):
        # df = self.df
        # df_cagr = pd.DataFrame(cagr(df.values[-1], df.values[0], self.stock_date), columns=['CAGR'])
        # df_cagr['Return'] = stock_return(df.values[-1], df.values[0])
        
        # df_cagr.index = df.T.index
        # st.dataframe(df_cagr)
        pass