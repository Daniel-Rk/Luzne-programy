from msilib.schema import Class
from tkinter.messagebox import YES
from numpy import *
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats


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



class finance_data():
    def __init__(self, name):
        df = pd.read_html('https://www.biznesradar.pl/raporty-finansowe-bilans/'+name+',Q')[-3].drop([0]).set_index('Unnamed: 0')
        shares = pd.read_html('https://www.biznesradar.pl/wskazniki-wartosci-rynkowej/'+name)[0].drop([2,3,4,5,6,7,8,9
                                                                                                    ,10,11,12,13,14,15]).set_index('Unnamed: 0')
        ZiS = pd.read_html('https://www.biznesradar.pl/raporty-finansowe-rachunek-zyskow-i-strat/'+name+',Q')[-3].drop([0]).set_index(
                                                                                                                                    'Unnamed: 0')
        flow = pd.read_html('https://www.biznesradar.pl/raporty-finansowe-przeplywy-pieniezne/'+name+',Q')[0].drop([0]).set_index(
                                                                                                                                    'Unnamed: 0')  

        # ---------- Delete the last blank column
        del df["Unnamed: "+str(len(df.columns))]  
        del shares["Unnamed: "+str(len(shares.columns))] 
        del ZiS["Unnamed: "+str(len(ZiS.columns))]
        del flow["Unnamed: "+str(len(flow.columns))]


        # ---------- Delete the last string elements in columns name
        df = df.drop(df.columns[0:4], axis=1)
        shares = shares.drop(shares.columns[0:(len(shares.columns)-len(df.columns))], axis=1)
        ZiS = ZiS.drop(ZiS.columns[0:(len(ZiS.columns)-len(df.columns))], axis=1)
        flow = flow.drop(flow.columns[0:(len(flow.columns)-len(df.columns))], axis=1)
        df.columns = df.columns.str.slice(start=2, stop=7)
        flow.columns = ZiS.columns = shares.columns = df.columns


        # ---------- Delete an unnecessary string (like (k/k), etc.)
        for i in range(len(df.columns)):
            df[str(df.columns[i])] = df[str(df.columns[i])].str.replace('(k/k).*','')
            df[str(df.columns[i])] = df[str(df.columns[i])].str.replace('(r/r).*','')
            ZiS[str(ZiS.columns[i])] = ZiS[str(ZiS.columns[i])].str.replace('(k/k).*','')
            ZiS[str(ZiS.columns[i])] = ZiS[str(ZiS.columns[i])].str.replace('(r/r).*','')
            flow[str(flow.columns[i])] = flow[str(flow.columns[i])].str.replace('(k/k).*','')
            flow[str(flow.columns[i])] = flow[str(flow.columns[i])].str.replace('(r/r).*','')

            df[str(df.columns[i])] = df[str(df.columns[i])].str.replace(' ','')
            shares[str(shares.columns[i])] = shares[str(shares.columns[i])].str.replace(' ','')
            ZiS[str(ZiS.columns[i])] = ZiS[str(ZiS.columns[i])].str.replace(' ','')
            flow[str(flow.columns[i])] = flow[str(flow.columns[i])].str.replace(' ','')

        self.df = df.apply(pd.to_numeric, errors='ignore')
        self.shares = shares.apply(pd.to_numeric, errors='ignore')
        self.ZiS = ZiS.apply(pd.to_numeric, errors='ignore')
        self.flow = flow.apply(pd.to_numeric, errors='ignore')


class Balance(finance_data):
    def __init__(self, ticker):
        zis = finance_data(ticker).ZiS
        shares = finance_data(ticker).shares
        df = finance_data(ticker).df
        flow = finance_data(ticker).flow

        self.time = df.columns
        self.shares = shares.T['Liczba akcji']
        self.year = [self.time[i] for i in range(3,len(self.time),4)]
        if len(self.year)%4 == 0:
            pass
        else:
            self.year = np.append(self.year, '21/Q4' )

        df = df.T
        self.revenue = zis.T['Przychody ze sprzedaży']
        self.Production_cost = zis.T['Techniczny koszt wytworzenia produkcji sprzedanej']
        self.Sell_profit = zis.T['Zysk ze sprzedaży']
        self.Gross_profit = zis.T['Zysk z działalności gospodarczej']
        self.net_profit = zis.T['Zysk netto']
        self.net_profit_year = [ np.sum(self.net_profit[i:i+4]) for i in range(0,len(self.net_profit),4) ]
        self.net_profit_share = self.net_profit/self.shares*1000
        self.net_profit_share_year = [ np.sum(self.net_profit_share[i:i+4]) for i in range(0,len(self.net_profit_share),4) ]
        self.Last4_profit = []
        self.Last4_revenue = []
        self.Revenue_year = [ np.sum(self.revenue[i:i+4]) for i in range(0,len(self.shares),4) ]
        self.Profit_year = [ np.sum(self.net_profit[i:i+4]) for i in range(0,len(self.shares),4) ]
        self.Last4_profit = [ np.sum(self.net_profit[i-4:i]) for i in range(4,len(self.shares)+1) ]
        self.Last4_revenue = [ np.sum(self.revenue[i-4:i]) for i in range(4,len(self.shares)+1) ]

        self.Noncurrent_assets = df['Aktywa trwałe']
        self.Nontangible = df['Wartości niematerialne i prawne']
        self.Tangible = df['Rzeczowe składniki majątku trwałego']
        self.Current_assets = df['Aktywa obrotowe']
        self.Supplies = df['Zapasy']
        self.cash = df['Środki pieniężne i inne aktywa pieniężne']
        self.Assets_total = df['Aktywa razem']
        self.Short_debt = df['Zobowiązania krótkoterminowe']
        self.Loans = df.T.iloc[23] + df.T.iloc[29]

        self.operating_flow = flow.T['Przepływy pieniężne z działalności operacyjnej']
        self.amortization = flow.T['Amortyzacja']
        self.capex = flow.T['CAPEX (niematerialne i rzeczowe)']
        self.Free_cash_flow = flow.T['Free Cash Flow']

        self.Assets_year_mean = [ np.mean(self.Assets_total[i:i+4]) for i in range(0,len(self.shares),4) ]
        self.Supplies_year_mean = [ np.mean(self.Supplies[i:i+4]) for i in range(0,len(self.shares),4) ]
        self.Equity = df['Kapitał własny akcjonariuszy jednostki dominującej']
        self.beta = self.Equity[3:len(self.Equity)]/self.Last4_revenue
        self.ROE = self.Last4_profit/self.Equity[3:len(self.Equity)]
        self.Assets_productivity = [ self.Revenue_year[i]/self.Assets_year_mean[i]*100 for i in range(0,len(self.Revenue_year))]
        self.Supply_productivity = [ self.Profit_year[i]/self.Supplies_year_mean[i]*100 for i in range(0,len(self.Revenue_year))]
        self.Current_ratio = self.Current_assets/self.Short_debt
        self.Quick_ratio = (self.Current_assets - self.Supplies)/self.Short_debt
        self.Cash_cover = self.cash/self.Short_debt
        self.SupplyCover = self.Supplies/self.Short_debt
        self.ROI = (self.cash+self.Supplies)/self.Loans






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
            self.dataframe[self.index_array[i]] = self.dataframe[self.index_array[i]]/self.dataframe[self.index_array[i]][0]
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
        
