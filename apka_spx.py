import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px


sp500 = yf.Ticker("^SPX").history(period="5y")
st.sidebar.title('Econometrics app')


# fig = px.line(sp500, x=sp500.index, y=["Low", "High"])
fig = px.line(sp500, x=sp500.index, y="Close")
fig.update_traces(textposition="bottom right")


st.plotly_chart(fig)


sp500  