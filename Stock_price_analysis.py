import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from core import *
from sklearn.linear_model import LinearRegression
import datetime

#sp500 = yf.Ticker("^SPX").history(period="5y")
# class Tools:
#     def __init__(self, df): 
#         rng = np.random.RandomState(42)
#         clf = IsolationForest(max_samples=100, random_state=rng) 
#         self.df = df

#     # def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
#     #     """Kernel Density Estimation with Scikit-learn"""
#     #     kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
#     #     kde_skl.fit(x[:, np.newaxis])
#     #     # score_samples() returns the log-likelihood of the samples
#     #     log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
#     #     return np.exp(log_pdf)

#     def plot_seperate(self, show_average=None, show_outlier=None, 
#                       show_linear_regres=None, show_kde=None):
#         for i,j in enumerate(self.df):
#             self.fig = px.line(self.df, x=self.df.index, y=j)

#             if show_average is True:
#                 self.fig = px.line(self.df, x=self.df.index, y=j).add_hline(mean(self.df[j]))

#             if show_outlier is True:
#                 ticker = j
#                 X = self.df[ticker]
#                 X.index = range(0, len(X.index))
#                 X_train = X.reset_index().T.reset_index().T.values.tolist()
#                 X_train.pop(0)
#                 model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
#                 model.fit(self.df[[ticker]])
#                 self.df['scores']=model.decision_function(self.df[[ticker]])
#                 self.df['anomaly']=model.predict(self.df[[ticker]])
#                 anomaly=self.df.loc[self.df['anomaly']==-1]
#                 self.fig = px.line(self.df, x=self.df.index, y=j).add_scatter(x=anomaly[ticker].index, y=anomaly[ticker].values,
#                                 marker=dict(size=8, color="red"), mode="markers")
            

#             if show_linear_regres is True:
#                 x = np.arange(len(self.df[j].values)).reshape((-1,1))
#                 y = np.array([self.df[j]]).reshape((-1,1))

#                 reg = LinearRegression()
#                 reg.fit(x, y)

#                 x_range = np.linspace(x.min(), x.max(), len(x))
#                 y_range = reg.predict(x_range.reshape(-1, 1))

#                 self.df['Regres model'] = y_range

#                 self.fig = px.line(self.df, x=self.df.index, y=[j, 'Regres model'])


#             if show_kde is True:
#                 self.fig = px.histogram(self.df, x=j)
#                 st.plotly_chart(self.fig)

#             st.plotly_chart(self.fig)


def cagr(end, begin, date):
    n_to_read = datetime.datetime.strptime(date, '%Y-%m-%d').date()   
    n_current = datetime.date.today()
    n = int(n_current.year - n_to_read.year)
    cagr = ((end/begin)**(1/n) - 1)*100
    return cagr

def stock_return(end, begin):
    stock_return = (end/begin - 1)*100
    return stock_return



class Stock_Price():
    def __init__(self, stock_array = ['PZU.WA'], stock_date = '2020-01-05'):
        self.stock_array = stock_array
        self.stock_date = stock_date
        self.df = yahoo(self.stock_array).cena(start_= self.stock_date)    


    def price(self, show_average = None,
                show_outlier = None, show_linear_regres=None, show_quantile_regres=None, show_kde=None):
        
        self.df.describe()
        df_1 = analysis(self.df).magnitude()
        fig = px.line(df_1, x=df_1.index, y=df_1.columns)

        st.plotly_chart(fig)

        self.df = yahoo(self.stock_array).cena(start_= self.stock_date) 
        for i,j in enumerate(self.df):
            self.fig = px.line(self.df, x=self.df.index, y=j)

            if show_average is True:
                self.fig.add_hline(mean(self.df[j]))

            if show_outlier is True:
                ticker = j
                X = self.df[ticker]
                X.index = range(0, len(X.index))
                X_train = X.reset_index().T.reset_index().T.values.tolist()
                X_train.pop(0)
                model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
                model.fit(self.df[[ticker]])
                self.df['scores']=model.decision_function(self.df[[ticker]])
                self.df['anomaly']=model.predict(self.df[[ticker]])
                anomaly=self.df.loc[self.df['anomaly']==-1]
                self.fig = px.line(self.df, x=self.df.index, y=j).add_scatter(x=anomaly[ticker].index, y=anomaly[ticker].values,
                                marker=dict(size=8, color="red"), mode="markers")

            if show_linear_regres is True:
                x = np.arange(len(self.df[j].values)).reshape((-1,1))
                y = np.array([self.df[j]]).reshape((-1,1))
                reg = LinearRegression()
                reg.fit(x, y)
                x_range = np.linspace(x.min(), x.max(), len(x))
                y_range = reg.predict(x_range.reshape(-1, 1))
                self.df['Regres model'] = y_range
                self.fig = px.line(self.df, x=self.df.index, y=[j, 'Regres model'])



            if show_kde is True:
                    self.fig = px.histogram(self.df, x=j)

            if show_quantile_regres is True:
                st.plotly_chart(self.fig)
                
                from sklearn.utils.fixes import parse_version, sp_version
                rng = np.random.RandomState(42)
                x = self.df[j].values
                X = x[:, np.newaxis]
                y_true_mean = 10 + 0.5 * x

                y_normal = y_true_mean + rng.normal(loc=0, scale=0.5 + 0.5 * x, size=x.shape[0])
                a = 5
                y_pareto = y_true_mean + 10 * (rng.pareto(a, size=x.shape[0]) - 1 / (a - 1))

                solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

                from sklearn.linear_model import QuantileRegressor

                quantiles = [0.2, 0.5, 0.8]
                predictions = {}
                out_bounds_predictions = np.zeros_like(y_true_mean, dtype=np.bool_)
                for quantile in quantiles:
                    qr = QuantileRegressor(quantile=quantile, alpha=0, solver=solver)
                    y_pred = qr.fit(X, y_normal).predict(X)
                    predictions[quantile] = y_pred

                    if quantile == min(quantiles):
                        out_bounds_predictions = np.logical_or(
                            out_bounds_predictions, y_pred >= y_normal
                        )
                    elif quantile == max(quantiles):
                        out_bounds_predictions = np.logical_or(
                            out_bounds_predictions, y_pred <= y_normal
                        )

                for quantile, y_pred in predictions.items():   
                    self.df[f'Quantile: {quantile}'] = y_pred
                    #st.dataframe(self.df)
                self.df['X'] = X
                self.fig = px.line(self.df, x='X', y=['Quantile: 0.2', 'Quantile: 0.5', 'Quantile: 0.8'])



            st.plotly_chart(self.fig)


    def correlation(self):
        df_corr = self.df.corr()
        fig_corr = px.imshow(df_corr)
        st.plotly_chart(fig_corr)

    def show_table_info(self):
        df = self.df
        df_cagr = pd.DataFrame(cagr(df.values[-1], df.values[0], self.stock_date), columns=['CAGR'])
        df_cagr['Return'] = stock_return(df.values[-1], df.values[0])
        
        df_cagr.index = df.T.index
        st.dataframe(df_cagr)


