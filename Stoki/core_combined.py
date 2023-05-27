import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Balance():
    def __init__(self, df, shares, zis, flow):
        self.time = df.columns
        self.shares = shares.T['Liczba akcji']
        self.year = [self.time[i] for i in range(3,len(self.time),4)]
        if len(self.year)%4 != 0:
            pass
        else:
            self.year = np.append(self.year, '23/Q4' )

        df = df.T
        df = df*1000
        self.revenue = zis.T['Przychody ze sprzedaży']
        self.Production_cost = zis.T['Techniczny koszt wytworzenia produkcji sprzedanej']
        self.sell_ex = zis.T['Koszty sprzedaży']
        self.Sell_profit = zis.T['Zysk ze sprzedaży']
        self.Gross_profit = zis.T['Zysk z działalności gospodarczej']
        self.net_profit = zis.T['Zysk netto']
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
        self.Receivable = df['Należności krótkoterminowe']
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
        self.Receivable_year_mean = [ np.mean(self.Receivable[i:i+4]) for i in range(0,len(self.shares),4) ]
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

        self.A = (self.Current_assets - self.Short_debt)/self.Assets_total          # Working Capital/Total assets 
        self.B = self.net_profit/self.Assets_total                                  # Earnings/Total assets
        self.C = self.Gross_profit/self.Assets_total                                # EBIT/Total assets
        self.D = self.Equity/(self.Assets_total - self.Equity)                      # Equity/Total liabilities 
        self.E = self.Sell_profit/self.Assets_total
        
        self.Z_score = 6.56*self.A + 3.26*self.B + 6.76*self.C + 1.05*self.D 


    def equity(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        ax = plt.subplot(311)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        plt.bar(self.time, self.Equity/self.shares*1000, 
                label='equity/share ='"$"+str(round(self.Equity[len(self.shares)-1]/self.shares[len(self.shares)-1]*1000,2))+"$")
        plt.plot(self.time, self.cash/self.shares*1000 , color='green', linewidth = 3, 
                 label='cash/share ='"$"+str(round(self.cash[len(self.shares)-1]/self.shares[len(self.shares)-1]*1000,2))+"$")
        plt.xticks(rotation = 50 , fontsize = 12)
        plt.legend(loc='best' , fontsize = 12 )

        self.ax = plt.subplot(312)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)

        plt.bar(self.time[3:len(self.shares)], self.ROE*100, color='darkgreen')
        self.ax.set_title('ROE')
        self.ax.set_ylabel('[%]')
        plt.xticks(rotation = 50 , fontsize = 12)
        plt.tight_layout(pad=3)

        self.ax = plt.subplot(313)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        
        plt.plot(self.time[3:len(self.shares)], self.beta[3:len(self.shares)]*self.ROE*100 )
        self.ax.set_title('ROE*BETA')
        self.ax.set_ylabel('[%]')
        plt.xticks(rotation = 50 , fontsize = 12)
        plt.tight_layout(pad=3)



    def Assets(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_title('Assets' , fontsize = 20)
        plt.bar(self.time, self.Assets_total/1000000, label='Total Assets')
        plt.plot(self.time, self.Current_assets/1000000, color='firebrick',label='Current Assets')
        plt.plot(self.time, self.cash/1000000, color = 'lime', label='Cash')
        self.ax.set_ylabel('w mld')
        plt.xticks(rotation=50)

        self.ax = plt.subplot(212)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.plot(self.year , self.Assets_productivity, marker = '.', label='Assets productivity = revenue/assets')
        plt.plot(self.year , self.Supply_productivity, marker = '.', label = 'Supplies productivity = profit/supplies')
        plt.legend(loc='best')

        self.fig = plt.figure(figsize = (20,15) , dpi=80)

        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.plot(self.time , (self.Current_assets-self.Short_debt)/self.Assets_total*100, marker = '.', 
                                                                        label='(Current assets-Short debt)/Total Assets')
        plt.plot(self.time , self.Supplies/self.Current_assets*100, label='Supplies/Current assets')
        plt.legend(loc='best' , fontsize = 12 )
        plt.xticks(rotation=50)


    def Cover(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_title('Liquidity' , fontsize=20)
        plt.plot(self.time, self.Current_ratio, label='Current ratio')
        plt.plot(self.time, self.Quick_ratio, label = 'Quick ratio')
        plt.plot(self.time, self.SupplyCover, label='Supply cover')
        plt.plot(self.time, self.Cash_cover, label='Cash cover')
        plt.legend(loc='best')
        plt.xticks(rotation = 50)
                    
        self.ax = plt.subplot(212)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.bar(self.time, self.Supplies/self.Production_cost, label= 'Supply/Production Costs')
        plt.plot(self.time, self.ROI, label='ROI= (Cash+supplies)/Loans')
        #plt.plot(self.time, self.Loans)
        plt.xticks(rotation = 50)
        plt.legend(loc='best')


    def Flow(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.plot(self.time, self.operating_flow, label = 'Operating flow')
        plt.plot(self.time, self.revenue - self.operating_flow)
        plt.xticks(rotation = 50 , fontsize = 12)
        plt.legend(loc = 'best')

        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.plot((self.net_profit + self.amortization + (self.Current_assets - self.Short_debt) - self.capex)/self.shares*1000, 
                                                                                                                label = 'FCF/share')  

        #plt.plot((self.net_profit + self.amortization + (self.Current_assets - self.Short_debt) - self.capex)/self.shares*1000, 
        #                                                                                                        label = 'FCF/share') 
        # Zamiast net profit ----> zysk operacyjny skorygowany stopa podatkowa

        plt.plot(self.net_profit/self.shares*1000, label = 'net profit/ shares')
        plt.xticks(rotation = 50 , fontsize = 12)
        plt.legend(loc = 'best')




"""
-------------------------------------------------- ZiS
"""





class zis():
    def __init__(self , df, shares):  
        self.time = df.columns 
        self.shares = shares.T['Liczba akcji']
        self.year = [ self.time[i] for i in range(3,len(self.time),4)]
        if len(self.year)%4 != 0:
            pass
        else:
            self.year = np.append(self.year, '23/Q4' )
    
        df = df.T
        df = df*1000
        self.sell_revenue = df['Przychody ze sprzedaży']
        self.Revenue_year = [np.sum(self.sell_revenue[i:i+4]) for i in range(0,len(self.sell_revenue),4)] 
        self.production_costs = df['Techniczny koszt wytworzenia produkcji sprzedanej']
        self.sell_costs = df['Koszty sprzedaży']
        self.board_costs = df['Koszty ogólnego zarządu']
        self.sell_profit = df['Zysk ze sprzedaży']
        self.gross_profit = df['Zysk z działalności gospodarczej']
        self.op_profit = df['Zysk operacyjny (EBIT)']
        self.op_profit_year = [ np.sum(self.op_profit[i:i+4]) for i in range(0,len(self.op_profit),4) ]


        self.net_profit = df['Zysk netto']
        self.net_profit_year = [ np.sum(self.net_profit[i:i+4]) for i in range(0,len(self.net_profit),4) ]
        self.net_profit_share = self.net_profit/self.shares
        self.net_profit_share_year = [ np.sum(self.net_profit_share[i:i+4]) for i in range(0,len(self.net_profit_share),4) ]

        self.Revenue_1Q = [ self.sell_revenue[i] for i in range(0,len(self.sell_revenue),4)]
        self.Revenue_2Q = [ self.sell_revenue[i] for i in range(1,len(self.sell_revenue),4)]
        self.Revenue_3Q = [ self.sell_revenue[i] for i in range(2,len(self.sell_revenue),4)]
        self.Revenue_4Q = [ self.sell_revenue[i] for i in range(3,len(self.sell_revenue),4)]
        
        self.gross_profit_1Q = [ self.gross_profit[i] for i in range(0,len(self.sell_revenue),4)]
        self.gross_profit_2Q = [ self.gross_profit[i] for i in range(1,len(self.sell_revenue),4)]
        self.gross_profit_3Q = [ self.gross_profit[i] for i in range(2,len(self.sell_revenue),4)]
        self.gross_profit_4Q = [ self.gross_profit[i] for i in range(3,len(self.sell_revenue),4)]




    def Revenue(self):
        self.fig = plt.figure(figsize = (20,8) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)  
        
        plt.bar(self.time , self.sell_revenue/self.shares*1000 , label = 'sell_revenue' )
        plt.plot(self.time , self.production_costs/self.shares*1000 , label = 'production_costs', color ='saddlebrown')
        plt.plot(self.time , self.sell_costs/self.shares*1000 , label = 'sell_costs', color='crimson')
        plt.plot(self.time , self.board_costs/self.shares*1000 , label = 'board_costs', color='limegreen')
        plt.xticks(rotation = 50)
        plt.yticks(fontsize = 12)
        self.ax.legend(loc = 'best' , fontsize = 12)
        self.ax = plt.subplot(212)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_ylabel('Sell revenue/production costs', fontsize=15)
        plt.bar(self.time, self.sell_revenue/self.production_costs)
        plt.xticks(rotation = 50)
        
        self.fig = plt.figure(figsize = (15,8) , dpi=80)
        self.ax = plt.subplot(212)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_ylabel('mld zł')
        plt.bar(self.year, self.Revenue_year)
        
        self.fig = plt.figure(figsize = (15,8) , dpi=80)
        self.ax = plt.subplot(221)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax2 = self.ax.twinx()
        self.ax.plot([self.time[i] for i in range(0,len(self.sell_revenue),4)] , self.Revenue_1Q)
        self.ax.plot([self.time[i] for i in range(0,len(self.sell_revenue),4)] , self.gross_profit_1Q)
        
        self.ax2.plot([self.time[i] for i in range(0,len(self.sell_revenue),4)] 
                      ,[ self.gross_profit_1Q[i]/self.Revenue_1Q[i]*100 for i in range(0,len(self.Revenue_1Q))],linestyle='--')         
        self.ax = plt.subplot(222)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax2 = self.ax.twinx()
        self.ax.plot([self.time[i] for i in range(1,len(self.sell_revenue),4)] , self.Revenue_2Q)
        self.ax.plot([self.time[i] for i in range(1,len(self.sell_revenue),4)] , self.gross_profit_2Q)
        
        self.ax2.plot([self.time[i] for i in range(1,len(self.sell_revenue),4)] 
                      , [ self.gross_profit_2Q[i]/self.Revenue_2Q[i]*100 for i in range(0,len(self.Revenue_2Q))],linestyle='--')
        self.ax = plt.subplot(223)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax2 = self.ax.twinx()
        self.ax.plot([self.time[i] for i in range(2,len(self.sell_revenue),4)] , self.Revenue_3Q)
        self.ax.plot([self.time[i] for i in range(2,len(self.sell_revenue),4)] , self.gross_profit_3Q)
        
        self.ax2.plot([self.time[i] for i in range(2,len(self.sell_revenue),4)] 
                      , [ self.gross_profit_3Q[i]/self.Revenue_3Q[i]*100 for i in range(0,len(self.Revenue_3Q))],linestyle='--')
        self.ax = plt.subplot(224)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax2 = self.ax.twinx()
        self.ax.plot([self.time[i] for i in range(3,len(self.sell_revenue),4)] , self.Revenue_4Q)
        self.ax.plot([self.time[i] for i in range(3,len(self.sell_revenue),4)] , self.gross_profit_4Q)
        
        self.ax2.plot([self.time[i] for i in range(3,len(self.sell_revenue),4)] 
                      , [ self.gross_profit_4Q[i]/self.Revenue_4Q[i]*100 for i in range(0,len(self.Revenue_4Q))],linestyle='--')
        self.fig.tight_layout(pad=3)



    def margins(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)

        plt.plot(self.time , self.sell_profit/self.sell_revenue*100, label='sell margin', color='darkviolet',marker='o')
        plt.plot(self.time , self.net_profit/self.sell_revenue*100, label='net margin', color='forestgreen',marker='o')
        plt.xticks(rotation =50)
        plt.legend(loc='best', fontsize=14)
        self.ax = plt.subplot(212)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)

        plt.plot(self.time , (self.sell_profit-self.net_profit)/self.sell_revenue*100)
        plt.xticks(rotation =50)
        plt.tight_layout(pad = 4 )




    def earnings(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_title('Earnings per share [zł]')
   
        self.polynomial_coeff = np.polyfit([i for i in range(0,len(self.time))], self.net_profit_share, 1)
        self.curve = np.polyval(self.polynomial_coeff, [i for i in range(0,len(self.time))])
       
        self.net_profit_mean = [ np.sum(self.net_profit_share[0:i])/i
                   for i in range(1,len(self.net_profit)+1) ] 
        self.variance = [np.sum(self.net_profit_share[0:i]-self.net_profit_mean[i])**2/(i+1) for i in range(0,len(self.net_profit_share)) ]
        self.sd = np.sqrt(self.variance)

        plt.vlines(x=self.time , ymin =0 ,ymax = self.net_profit_share , color='lime', linewidth=2) 
        plt.scatter(self.time , self.net_profit_share , s =100 , color = 'lime')
        plt.plot(self.time , self.net_profit_mean , color= 'darkviolet')
        plt.fill_between(self.time , self.net_profit_mean -self.sd , self.net_profit_mean+self.sd , alpha=0.3)
        plt.plot(self.time, self.curve)

        plt.xticks(rotation = 50)
        plt.yticks(fontsize = 12)
        
        self.ax=plt.subplot(212)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        
        j = 3
        while(j<=len(self.shares)):
            self.coefficients = np.polyfit([j-3,j-2,j-1], self.net_profit_share[j-3:j], 1)
            self.coefficients_mean = np.polyfit([j-3,j-2,j-1], self.net_profit_mean[j-3:j], 1)
            self.curve = np.polyval(self.coefficients,[j-3,j-2,j-1])
            self.curve_mean = np.polyval(self.coefficients_mean,[j-3,j-2,j-1])
            if self.coefficients[0]<=0:
                self.color = 'red'
            else:
                self.color = 'green'
            plt.plot(self.time[j-3:j], self.curve, color=self.color)
            plt.plot(self.time[j-3:j], self.curve_mean, color='black')
            j = j+1
        plt.xticks(rotation=50)


    def earnings_year(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_title('Earnings per share [zł]') 
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylim([-100,100])
        self.net_profit_year_diff = [0]
        self.net_profit_year_diff = np.append(self.net_profit_year_diff,
                            np.diff(self.net_profit_share_year)/self.net_profit_share_year[0:len(self.year)-1]*100)
        
        self.ax.vlines(x=self.year , ymin =0 ,ymax = self.net_profit_share_year , color='lime',alpha=0.5, linewidth=4) 
        self.ax.scatter(self.year , self.net_profit_share_year , s=65 , color = 'lime')

        self.net_profit_mean_year = [ np.sum(self.net_profit_share_year[0:i])/i
                   for i in range(1,len(self.net_profit_share_year)+1) ] 
        self.variance_year = [np.sum(self.net_profit_share_year[0:i]-self.net_profit_mean_year[i])**2/(i+1) 
                              for i in range(0,len(self.net_profit_share_year)) ]
        self.sd_year = np.sqrt(self.variance_year)
        
        self.ax.plot(self.year , self.net_profit_mean_year,color='navy')
        self.ax.fill_between(self.year , self.net_profit_mean_year -self.sd_year , self.net_profit_mean_year+self.sd_year , alpha=0.3 )
        self.ax2.plot(self.year, self.net_profit_year_diff, linestyle='--')
        plt.xticks(rotation = 50)
        plt.yticks(fontsize = 12)

    def tax(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_ylabel('w Milionach', fontsize=15)
        plt.plot(self.time, self.sell_revenue/1000, label = 'Revenue', color = 'navy')
        plt.plot(self.time, self.net_profit/1000, label = 'Net profit', color = 'darkgreen')
        #plt.plot(self.time, self.sell_profit/1000, label = 'Sell profit', color = 'darkviolet')
        plt.legend(loc='best', fontsize=14)
        plt.xticks(rotation=50)


        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_ylim([-100,100])
        
        plt.plot(self.time , (100 - self.net_profit/self.gross_profit*100) )
        plt.axhline( np.mean((100 - self.net_profit/self.gross_profit*100)) , linestyle ='--')
        plt.xticks(rotation=50)

"""
-------------------------------------------------- Flow
"""
class Flow():
    def __init__(self, df, shares, zis, flow):
        self.time = df.columns
        self.shares = shares.T['Liczba akcji']
        self.year = [self.time[i] for i in range(3,len(self.time),4)]
        self.year = np.append(self.year, '22/Q4' )

        self.operating_flow = flow.T['Przepływy pieniężne z działalności operacyjnej']*1000
        self.investment_flow = flow.T['Przepływy pieniężne z działalności inwestycyjnej']*1000
        self.financial_flow = flow.T['Przepływy pieniężne z działalności finansowej']*1000
        self.capex = flow.T['CAPEX (niematerialne i rzeczowe)']*1000

        self.sell_revenue = zis.T['Przychody ze sprzedaży']







"""
-------------------------------------------------- Banki ZiS
"""

class Bank_zis:
    def __init__(self , df, shares):
        self.time = df.columns 
        self.shares = shares.T['Liczba akcji']
        self.year = [ self.time[i] for i in range(3,len(self.time),4)]
        if len(self.year)%4 == 0:
            pass
        else:
            self.year = np.append(self.year, '21/Q4' )

        df = df.T
        self.prowizje_revenue = df['Przychody prowizyjne']
        self.prowizje_costs = df['Koszty prowizyjne']
        self.prowizje_score = df['Wynik z tytułu prowizji']
        self.odsetki_revenue = df['Przychody odsetkowe']
        self.odsetki_costs = df['Koszty odsetkowe']
        self.odsetki_score = df['Wynik z tytułu odsetek']
        self.credit_reserves = df['Odpisy netto z tytułu utraty wartości kredytów']
        self.Admistrative_costs = df['Ogólne koszty administracyjne']
        self.Gross_profit = df['Zysk przed opodatkowaniem']
        self.net_profit = df['Zysk netto']

        self.sell_revenue = self.prowizje_revenue + self.odsetki_revenue 
        self.net_profit_share = self.net_profit/self.shares*1000
        self.net_profit_share_year = [ np.sum(self.net_profit_share[i:i+4]) for i in range(0,len(self.net_profit_share),4) ]
        self.Revenue_year = [np.sum(self.sell_revenue[i:i+4]) for i in range(0,len(self.net_profit_share),4)] 
        self.Admistrative_costs_year = [-np.sum(self.Admistrative_costs[i:i+4]) for i in range(0,len(self.net_profit_share),4)]  

        self.Revenue_1Q = [ self.sell_revenue[i] for i in range(0,len(self.shares),4)]
        self.Revenue_2Q = [ self.sell_revenue[i] for i in range(1,len(self.shares),4)]
        self.Revenue_3Q = [ self.sell_revenue[i] for i in range(2,len(self.shares),4)]
        self.Revenue_4Q = [ self.sell_revenue[i] for i in range(3,len(self.shares),4)]
        
        self.net_profit_year_diff = [0]
        self.net_profit_year_diff = np.append(self.net_profit_year_diff, np.diff(self.net_profit_share_year)/1)
                                              #self.net_profit_share_year[0:len(self.year)-1]*100)
        
        self.gross_profit_1Q = [ self.net_profit[i] for i in range(0,len(self.shares),4)]
        self.gross_profit_2Q = [ self.net_profit[i] for i in range(1,len(self.shares),4)]
        self.gross_profit_3Q = [ self.net_profit[i] for i in range(2,len(self.shares),4)]
        self.gross_profit_4Q = [ self.net_profit[i] for i in range(3,len(self.shares),4)]

    def Revenues(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(411)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.bar(self.time, self.prowizje_revenue/self.shares*1000)
        plt.plot(self.time, self.prowizje_score/self.shares*1000, color='firebrick', marker='o')
        self.ax.set_title('odsetki')
        plt.xticks(rotation = 50)
        self.ax = plt.subplot(412)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.plot(self.time, self.prowizje_score/self.prowizje_revenue*100, linestyle='--',marker='o')
        plt.xticks(rotation = 50)
        self.ax = plt.subplot(413)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.bar(self.time, self.odsetki_revenue/self.shares*1000)
        plt.plot(self.time, self.odsetki_score/self.shares*1000, color='firebrick', marker='o')
        plt.xticks(rotation = 50)
        self.ax.set_title('prowizje')
        self.ax = plt.subplot(414)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.plot(self.time, self.odsetki_score/self.odsetki_revenue*100, linestyle='--',marker='o')
        plt.tight_layout(pad=2)
        plt.xticks(rotation = 50)
        self.fig = plt.figure(figsize = (15,8) , dpi=80)
        self.ax = plt.subplot(212)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_ylabel('mld zł')
        plt.bar(self.year, self.Revenue_year)
        
        self.fig = plt.figure(figsize = (15,8) , dpi=80)
        self.ax = plt.subplot(221)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.xticks(rotation = 50)
        self.ax2 = self.ax.twinx()
        self.ax.plot([self.time[i] for i in range(0,len(self.shares),4)] , self.Revenue_1Q)
        self.ax.plot([self.time[i] for i in range(0,len(self.shares),4)] , self.gross_profit_1Q)
        
        self.ax2.plot([self.time[i] for i in range(0,len(self.shares),4)] 
                      ,[ self.gross_profit_1Q[i]/self.Revenue_1Q[i]*100 for i in range(0,len(self.Revenue_1Q))],linestyle='--')
        
        self.ax = plt.subplot(222)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.xticks(rotation = 50)
        self.ax2 = self.ax.twinx()
        self.ax.plot([self.time[i] for i in range(1,len(self.shares),4)] , self.Revenue_2Q)
        self.ax.plot([self.time[i] for i in range(1,len(self.shares),4)] , self.gross_profit_2Q)
        
        self.ax2.plot([self.time[i] for i in range(1,len(self.shares),4)] 
                      , [ self.gross_profit_2Q[i]/self.Revenue_2Q[i]*100 for i in range(0,len(self.Revenue_2Q))],linestyle='--')
        
        self.ax = plt.subplot(223)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.xticks(rotation = 50)
        self.ax2 = self.ax.twinx()
        self.ax.plot([self.time[i] for i in range(2,len(self.shares),4)] , self.Revenue_3Q)
        self.ax.plot([self.time[i] for i in range(2,len(self.shares),4)] , self.gross_profit_3Q)
    
        self.ax2.plot([self.time[i] for i in range(2,len(self.shares),4)] 
                      , [ self.gross_profit_3Q[i]/self.Revenue_3Q[i]*100 for i in range(0,len(self.Revenue_3Q))],linestyle='--')

        
        self.ax = plt.subplot(224)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.xticks(rotation = 50)
        self.ax2 = self.ax.twinx()
        self.ax.plot([self.time[i] for i in range(3,len(self.shares),4)] , self.Revenue_4Q)
        self.ax.plot([self.time[i] for i in range(3,len(self.shares),4)] , self.gross_profit_4Q)
        
        self.ax2.plot([self.time[i] for i in range(3,len(self.shares),4)] 
                      , [ self.gross_profit_4Q[i]/self.Revenue_4Q[i]*100 for i in range(0,len(self.Revenue_4Q))],linestyle='--')
        self.fig.tight_layout(pad=3)
        
    def credits(self):
        self.fig = plt.figure(figsize = (20,18) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_title('Total revenue / Credits reserves')
        plt.plot(self.time, self.sell_revenue/np.absolute(self.credit_reserves))
        plt.xticks(rotation = 50)

    def earnings(self):
        self.fig = plt.figure(figsize = (20,18) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_title('Earnings per share [zł]')
        self.polynomial_coeff = np.polyfit([i for i in range(0,len(self.time))], self.net_profit_share, 1)
        self.curve = np.polyval(self.polynomial_coeff, [i for i in range(0,len(self.time))])
       
        self.net_profit_mean = [ np.sum(self.net_profit_share[0:i])/i for i in range(1,len(self.net_profit)+1) ] 
        self.variance = [np.sum(self.net_profit_share[0:i]-self.net_profit_mean[i])**2/(i+1) for i in range(0,len(self.net_profit_share)) ]
        self.sd = np.sqrt(self.variance)

        plt.vlines(x=self.time , ymin =0 ,ymax = self.net_profit_share , color='lime', linewidth=2) 
        plt.scatter(self.time , self.net_profit_share , s =100 , color = 'lime')
        plt.plot(self.time , self.net_profit_mean , color= 'darkviolet')
        plt.fill_between(self.time , self.net_profit_mean -self.sd , self.net_profit_mean+self.sd , alpha=0.3)
        plt.plot(self.time, self.curve)
    
        plt.xticks(rotation = 50)
        plt.yticks(fontsize = 12)
        self.ax=plt.subplot(212)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        
        j = 4
        while(j<=len(self.shares)):
            self.coefficients = np.polyfit([j-4,j-3,j-2,j-1], self.net_profit_share[j-4:j], 1)
            self.coefficients_mean = np.polyfit([j-4,j-3,j-2,j-1], self.net_profit_mean[j-4:j], 1)
            self.curve = np.polyval(self.coefficients,[j-4,j-3,j-2,j-1])
            self.curve_mean = np.polyval(self.coefficients_mean,[j-4,j-3,j-2,j-1])
            if self.coefficients[0]<=0:
                self.color = 'red'
            else:
                self.color = 'green'
            plt.plot(self.time[j-4:j], self.curve, color=self.color)
            plt.plot(self.time[j-4:j], self.curve_mean, color='black')
            j = j+1
        plt.xticks(rotation = 50)
        
        
        
    def earnings_year(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.set_title('Earnings per share [zł]')
        self.ax2 = self.ax.twinx()
        self.ax.vlines(x=self.year , ymin =0 ,ymax = self.net_profit_share_year , color='lime',alpha=0.5, linewidth=4) 
        self.ax.scatter(self.year , self.net_profit_share_year , s =65 , color = 'lime')

        self.net_profit_mean_year = [ np.sum(self.net_profit_share_year[0:i])/i for i in range(1,len(self.net_profit_share_year)+1) ] 
        self.variance_year = [np.sum(self.net_profit_share_year[0:i]-self.net_profit_mean_year[i])**2/(i+1) 
                              for i in range(0,len(self.net_profit_share_year)) ]
        self.sd_year = np.sqrt(self.variance_year)
        
        self.ax.plot(self.year , self.net_profit_mean_year,color='navy')
        self.ax.fill_between(self.year , self.net_profit_mean_year -self.sd_year , self.net_profit_mean_year+self.sd_year , alpha=0.3 )
        self.ax2.plot(self.year, self.net_profit_year_diff, linestyle='--')
        plt.xticks(rotation = 50)
        plt.yticks(fontsize = 12)
        
    def tax(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        
        plt.plot(self.time , (100 - self.net_profit/self.Gross_profit*100) )
        plt.axhline( np.mean((100 - self.net_profit/self.Gross_profit*100)) , linestyle ='--')
        plt.xticks(rotation = 50)

"""
-------------------------------------------------- Banki Balance
"""

class Bank_balance:
    def __init__(self , df, zis, shares, flow):
        self.time = df.columns
        self.shares = shares.T['Liczba akcji']
        self.year = [ self.time[i] for i in range(3,len(self.time),4)]
        #self.year = np.append(self.year , '21/Q4')

        zis = zis.T
        self.prowizje_revenue = zis['Przychody prowizyjne']
        self.prowizje_costs = zis['Koszty prowizyjne']
        self.prowizje_score = zis['Wynik z tytułu prowizji']
        self.odsetki_revenue = zis['Przychody odsetkowe']
        self.odsetki_costs = zis['Koszty odsetkowe']
        self.odsetki_score = zis['Wynik z tytułu odsetek']
        self.credit_reserves = zis['Odpisy netto z tytułu utraty wartości kredytów']
        self.Admistrative_costs = zis['Ogólne koszty administracyjne']
        self.Gross_profit = zis['Zysk przed opodatkowaniem']
        self.net_profit = zis['Zysk netto']

        self.revenue = self.prowizje_revenue + self.odsetki_revenue
        self.Last4_revenue = [ np.sum(self.revenue[i-4:i]) for i in range(4,len(self.shares)+1) ]
        self.Last4_profit = [ np.sum(self.net_profit[i-4:i]) for i in range(4,len(self.shares)+1) ]
        self.Revenue_year = [ np.sum(self.revenue[i:i+4]) for i in range(0,len(self.shares),4) ]

        df = df.T
        self.client_receivables = df['Należności od klientów']
        self.Current_finance = df['Aktywa finansowe przeznaczone do obrotu'] 

        self.client_debt = df['Zobowiązania wobec klientów']
        self.Current_finance_debt = df['Zobowiązania finansowe przeznaczone do obrotu']

        self.operating_flow = flow.T['Przepływy pieniężne z działalności operacyjnej']
        self.amortization = flow.T['Amortyzacja']
        self.capex = flow.T['CAPEX (niematerialne i rzeczowe)']
        self.Free_cash_flow = flow.T['Free Cash Flow']

        self.Equity = df['Kapitały razem']
        self.cash = df['Gotówka i operacje z bankami centralnymi'] + df['Należności od banków']
        self.Assets_total = df['Aktywa razem']
        self.Assets_year_mean = [ np.mean(self.Assets_total[i:i+4]) for i in range(0,len(self.shares),4) ]
        
        self.ROE = self.Last4_profit/self.Equity[3:len(self.Equity)] 
        self.beta = self.Equity[3:len(self.Equity)]/self.Last4_revenue
        self.Assets_productivity = [ self.Revenue_year[i]/self.Assets_year_mean[i]*100 for i in range(0,len(self.Revenue_year))]


    def equity(self):
        self.fig = plt.figure(figsize = (20,15) , dpi=80)
        self.ax = plt.subplot(311)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        
        plt.bar(self.time, self.Equity/self.shares*1000, 
                label='equity/share ='"$"+str(round(self.Equity[len(self.shares)-1]/self.shares[len(self.shares)-1]*1000,2))+"$")
        plt.plot(self.time, self.cash/self.shares*1000 , color='green', linewidth = 3, 
                 label='cash/share ='"$"+str(round(self.cash[len(self.shares)-1]/self.shares[len(self.shares)-1]*1000,2))+"$")
        
        plt.xticks(rotation = 50 , fontsize = 12)
        plt.legend(loc='best' , fontsize = 12 )
        self.ax = plt.subplot(312)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        
        plt.bar(self.time[3:len(self.shares)], self.ROE*100, color='darkgreen')
        self.ax.set_title('ROE')
        self.ax.set_ylabel('[%]')
        plt.xticks(rotation = 50 , fontsize = 12)
        plt.tight_layout(pad=3)
        self.ax = plt.subplot(313)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        
        plt.plot(self.time[3:len(self.shares)], self.beta[3:len(self.shares)]*self.ROE*100 )
        self.ax.set_title('ROE*BETA')
        self.ax.set_ylabel('[%]')
        plt.xticks(rotation = 50 , fontsize = 12)
        plt.tight_layout(pad=3)
        
        
    def cover(self):
        self.fig = plt.figure(figsize = (20,18) , dpi=80)
        self.ax = plt.subplot(211)
        self.ax.spines["top"].set_visible(False)
        self.ax.yaxis.grid(True)
        self.ax.set_axisbelow(True)
        plt.bar(self.time, self.client_receivables/self.client_debt, label='clients receives/debt')
        
        # plt.plot(self.time, self.cash/np.absolute(self.credit_reserves), label = 'cash / credit reserves')
        # plt.plot(self.time, self.client_receivables/np.absolute(self.credit_reserves), label= ' receives / credit reserves')
        plt.xticks(rotation = 50)        
        self.ax.legend(loc='best')
