from numpy import *
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import plotly.express as px
import streamlit as st


def cagr(start_data, end_data, n):
    """
    Compound Annual Growth Rate (CAGR)
    """
    cagr_value = (end_data/start_data)**(1/n) - 1
    return round(cagr_value*100, 2)


class percent_project:
    def __init__(self, kap_pocz, rate, n):
        self.kap_pocz = kap_pocz
        self.rate = rate
        self.n = n
        self.n_array = arange(n+1)
        self.simple_percent = kap_pocz*(1 + rate*self.n_array*0.01) 
        self.comp_percent = kap_pocz*(1 + rate*0.01)**self.n_array

    def simple_project(self):
        dict = {
            'x':self.n_array,
            'y':self.simple_percent
        }
        return dict

    def compound_project(self):
        dict = {
            'x':self.n_array,
            'y':self.comp_percent
        }
        return dict


# _____________________________________________________________________< Data Frame 
class yahoo:
    def __init__(self, ticker_array) -> None:
        self.ticker_array = ticker_array
        
    def cena(self, start_ = '2015-01-01', interval = '1wk'):
        data_container = []
        for i,j in enumerate(self.ticker_array):
            stock = yf.Ticker(j)
            data = stock.history(start=start_, interval = interval)
            close_price = data['Close']
            data_container.append(close_price)

        dictionary = dict(zip(self.ticker_array, data_container))
        self.df = pd.DataFrame(data=dictionary).dropna()

        return self.df 

    def volume(self, start_ = '2015-01-01', interval = '1wk'):
        data_container = []
        for i,j in enumerate(self.ticker_array):
            stock = yf.Ticker(j)
            data = stock.history(start=start_, interval=interval)
            volumen = data['Volume']
            data_container.append(volumen)

        dictionary = dict(zip(self.ticker_array, data_container))
        self.df_volume = pd.DataFrame(data=dictionary).dropna()

        return self.df_volume
        






class analysis:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.index_array = self.dataframe.columns.tolist()

    def price(self, regres = None):
        for i,j in enumerate(self.index_array):
            x = np.arange(len(self.dataframe[j].index))
            y = self.dataframe[j].values
        
            plt.figure()
            plt.plot(x,y, color='blue')

            if regres == True:
                slope, intercept, r, p, se = stats.linregress(x, y)
                plt.plot(x, intercept + slope*x, 'r', label='fitted line')

    def magnitude(self):
        for i,j in enumerate(self.index_array):
            self.dataframe[j] = self.dataframe[j]/self.dataframe[j][0]
        return self.dataframe


    def kde(self):
        avg = self.dataframe.mean()
        s = self.dataframe.std()

        for i,j in enumerate(self.index_array):
            plt.figure()
            plt.plot(self.dataframe[j], color='blue')
            plt.hlines(avg[j], self.dataframe.index[0], self.dataframe.index[-1], color='red')

            plt.figure()
            plt.vlines(self.dataframe[j].values[-1], 0, 0.3)
            plt.vlines(avg[j] - s[j],  0, 0.3, colors='green')
            plt.vlines(avg[j] + s[j], 0, 0.3, colors='green')
            self.dataframe[j].plot.kde()


    def average(self):
        pass 
    
    def anomaly(self):
        rng = np.random.RandomState(42)
        clf = IsolationForest(max_samples=100, random_state=rng)

        for i,j in enumerate(self.index_array):
            ticker = self.index_array[i]
            X = self.dataframe[ticker]
            X.index = range(0, len(X.index))
            X_train = X.reset_index().T.reset_index().T.values.tolist()
            X_train.pop(0)

            model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
            model.fit(self.dataframe[[ticker]])

            self.dataframe['scores']=model.decision_function(self.dataframe[[ticker]])
            self.dataframe['anomaly']=model.predict(self.dataframe[[ticker]])

            anomaly=self.dataframe.loc[self.dataframe['anomaly']==-1]
            anomaly_index=list(anomaly.index)

            plt.figure()
            plt.plot(self.dataframe[ticker], color = 'blue', label = j)
            plt.scatter(anomaly[ticker].index, anomaly[ticker].values, edgecolors='red')
            plt.xticks(rotation=50)
            plt.legend(loc='best')

    def dividend_yeld(self):
        avg = self.dataframe.mean()
        
