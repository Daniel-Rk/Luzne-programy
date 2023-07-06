import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from core import *
from sklearn.linear_model import LinearRegression
from Stock_price_analysis import *
from Stock_finance_analysis import *

st.title("Aplikacja")
page = st.sidebar.selectbox('Page', ['Cena', 'Wyniki'])

if page == 'Cena':
    array = st.text_input('Provide tickers [use ;]', value='PZU.WA').replace(" ","").split(";")
    date = st.sidebar.text_input('Provide date', value='2020-01-01').replace(" ","")
    mean = st.sidebar.checkbox('Show average')
    outlier = st.sidebar.checkbox('Show outliers')
    linear_regres = st.sidebar.checkbox('Show linear regres')
    quantile_regres = st.sidebar.checkbox('Show quantile regres')
    kde = st.sidebar.checkbox('Show KDE')
    correlation = st.sidebar.checkbox('Show Correlation')
    table_info = st.sidebar.checkbox("Show summary table")
    stock_to_benchmark = st.sidebar.checkbox("Show benchmark")

    # df = yahoo(array).cena()
    # st.dataframe(df)
    df = Stock_Price(stock_array = array, stock_date = date)

    df.price(show_average = mean,
                show_outlier = outlier, show_linear_regres = linear_regres,
                show_quantile_regres=quantile_regres, show_kde = kde, show_return_to_benchmark=stock_to_benchmark)

    if correlation is True:
        df.correlation()

    if table_info is True:
        df.show_table_info()



if page == "Wyniki":
    array = st.text_input('Provide tickers [use ;]', value='KO').replace(" ","").split(";")
    date = st.sidebar.text_input('Provide date', value='2020-01-01').replace(" ","")
    Q_status = st.sidebar.checkbox('Qaurtly')
    linear_regres_status = st.sidebar.checkbox('Show linear regres')
    table_info = st.sidebar.checkbox("Show summary table")


    df = Stock_Finance(stock_array = array, stock_date = date)  
    income = df.Income(quartly_status=Q_status)
    balance = df.Balance(quartly_status=Q_status)
    cash = df.Cash_flow(quartly_status=Q_status)

    #st.dataframe(income)
    st.dataframe(balance)
    #st.dataframe(cash)


    #plot(income, ['totalRevenue', 'grossProfit', 'operatingIncome', 'netIncome'], linear_regres=linear_regres_status)
    plot(balance, [['totalAssets','totalCurrentAssets', 'WorkingCapital'], 'totalShareholderEquity'], linear_regres=linear_regres_status)


    if table_info is True:
        # df.show_table_info()
        pass