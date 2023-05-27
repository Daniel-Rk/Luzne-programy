import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from finance_core import *

sp500 = yf.Ticker("^SPX").history(period="5y")
st.sidebar.title('Econometrics app')

array = st.text_input('Provide tickers [use ;]', value='PZU.WA').replace(" ","").split(";")
date = st.sidebar.text_input('Provide date', value='2020-01-01').replace(" ","")
st.sidebar.write(array)

show_average = st.sidebar.checkbox('Show average')
show_outlier = st.sidebar.checkbox('Show outliers')

# fig = px.line(sp500, x=sp500.index, y=["Low", "High"])
fig = px.line(sp500, x=sp500.index, y="Close")
fig.update_traces(textposition="bottom right")


st.plotly_chart(fig)

# _____________________ CENA ANALYSIS
df = yahoo(array).cena(start_=date)
df.describe()


df_1 = analysis(df).magnitude()

fig = px.line(df_1, x=df_1.index, y=df_1.columns)
st.plotly_chart(fig)

df = yahoo(array).cena(start_=date)

rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)

def plot_seperate():
    for i,j in enumerate(df):
        fig = px.line(df, x=df.index, y=j)

        if show_average is True:
            fig.add_hline(mean(df[j]))

        if show_outlier is True:
            ticker = j
            X = df[ticker]
            X.index = range(0, len(X.index))
            X_train = X.reset_index().T.reset_index().T.values.tolist()
            X_train.pop(0)
            model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
            model.fit(df[[ticker]])
            df['scores']=model.decision_function(df[[ticker]])
            df['anomaly']=model.predict(df[[ticker]])
            anomaly=df.loc[df['anomaly']==-1]
            fig.add_scatter(x=anomaly[ticker].index, y=anomaly[ticker].values,
                            marker=dict(size=8, color="red"), mode="markers")
            
        st.plotly_chart(fig)


plot_seperate()