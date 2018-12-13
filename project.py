from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def Data(path,b_w,s_w,l_w,bollinger):
    data = pd.read_csv(path,index_col='Date',usecols=['Date', 'Price'],parse_dates=True)
    data['return']=np.log(data['Price']).diff()

    data['s_ma']=pd.rolling_mean(data['Price'],window=s_w)
    data['l_ma']=pd.rolling_mean(data['Price'],window=l_w)
    
    mstd=pd.rolling_std(data['Price'],window=b_w)
    b_ma=pd.rolling_mean(data['Price'],window=b_w)
    
    data['upper_band']=b_ma+bollinger*mstd
    data['lower_band']=b_ma-bollinger*mstd
    
    return data

def Bollinger_band_strategy_filter(data,stop_loss,BO_spread=0,delay=0):
    data.dropna(inplace=True)
    data['signal']=np.nan
    stop_loss_point=[]

    
    for row in range(2,len(data)):
        if(data['Price'].iloc[row-1]>data['upper_band'].iloc[row-1] and data['Price'].iloc[row-2]<data['upper_band'].iloc[row-2] and data['s_ma'].iloc[row-1]<=data['l_ma'].iloc[row-1]):
            data['signal'].iloc[row]=-1
        elif(data['Price'].iloc[row-1]>data['lower_band'].iloc[row-1] and data['Price'].iloc[row-2]<data['lower_band'].iloc[row-2] and data['s_ma'].iloc[row-1]>data['l_ma'].iloc[row-1]):
            data['signal'].iloc[row]=1
    
    data['holding']=data['signal']
    data['signal'].fillna(value=0, inplace=True)
    data['holding'].iloc[0]=0
    data['trade_cum_return']=np.nan
    data['trade_cum_return'].iloc[0]=0
    
    for row in range(1,len(data)):
        if data['trade_cum_return'].iloc[row-1]<stop_loss:
            data['holding'].iloc[row]=0
            stop_loss_point.append(data.index[row])## Stop Loss point
            data['return'].iloc[row-1]=np.log((data['Price'].iloc[row-1]+delay/60.*(data['Price'].iloc[row]-data['Price'].iloc[row-1]))/data['Price'].iloc[row-2])
            data['return'].iloc[row-1]=data['return'].iloc[row-1] + np.log(1 - data['holding'].iloc[row-1]*BO_spread)
        elif abs(data['holding'].iloc[row-1] + data['signal'].iloc[row])>1:
            data['holding'].iloc[row]=data['holding'].iloc[row-1]
        elif data['holding'].iloc[row-1]==0:
            data['holding'].iloc[row]=data['holding'].iloc[row-1]+data['signal'].iloc[row]
            try:
                data['return'].iloc[row]=np.log((data['Price'].iloc[row]+delay/60.*(data['Price'].iloc[row+1]-data['Price'].iloc[row]))/data['Price'].iloc[row-1])
            except IndexError:
                pass
            data['return'].iloc[row]=data['return'].iloc[row]-np.log(1 + data['signal'].iloc[row]*BO_spread)
        else:
            data['holding'].iloc[row]=data['holding'].iloc[row-1]+data['signal'].iloc[row]
            try:
                data['return'].iloc[row]=np.log((data['Price'].iloc[row]+delay/60.*(data['Price'].iloc[row+1]-data['Price'].iloc[row]))/data['Price'].iloc[row-1])
            except IndexError:
                pass
            data['return'].iloc[row]=data['return'].iloc[row] + np.log(1 + data['signal'].iloc[row]*BO_spread)
        if data['holding'].iloc[row]==0:
            data['trade_cum_return'].iloc[row]=0
        else:
            data['trade_cum_return'].iloc[row]=data['trade_cum_return'].iloc[row-1]+data['return'].iloc[row]*data['holding'].iloc[row-1]##same

    data['str_return']=data['return']*data['holding']
    data['str_cum_return']=data['str_return'].cumsum()
    data['str_cum_simple_return']=np.exp(data['str_cum_return'])-1
    
    i = np.argmax(np.maximum.accumulate(data['str_cum_simple_return'].dropna()) - data['str_cum_simple_return'].dropna()) # end of the period
    j = np.argmax(data['str_cum_simple_return'].dropna()[:i]) # start of period
    plt.figure(1,figsize=(16,8))
    plt.title('Cumulative Strategy Simple Return')
    plt.plot(data['str_cum_simple_return'].dropna(),label='Cumulative Return')
    plt.plot([i, j], [data['str_cum_simple_return'].dropna()[i], data['str_cum_simple_return'].dropna()[j]], 'o', color='Red', markersize=10,label='Maximum Drawdown')
    plt.plot(stop_loss_point, data['str_cum_simple_return'].dropna()[stop_loss_point], 'o', color='Green', markersize=10,label='Stop Loss Trigger')
    plt.grid(ls=':')
    plt.legend()
    plt.show()
    
    print (data['str_cum_return'].iloc[-1]/(len(data)-2)*24*252.)/(np.std(data['str_return'])*np.sqrt(24*252))
    
    return data[['str_return','str_cum_return']]
    
def Performance(data):
    data['Drawdown']=data['str_cum_return']-np.maximum.accumulate(data['str_cum_return'])
  
    Annualized_return=data['str_cum_return'].iloc[-1]/(len(data)-2)*24*252.
    Annualized_vol=np.std(data['str_return'])*np.sqrt(24*252)
    IR=Annualized_return/Annualized_vol
    SR=(Annualized_return-0.04)/Annualized_vol
    Top5=data['Drawdown'].sort_values(inplace=False).head(5)
    GR=Annualized_return/Top5.iloc[0]
    
    VaR_95=Annualized_vol*norm.ppf(0.05)
    
    print(Annualized_return)
    print(Annualized_vol)
    print(IR)
    print(SR)
    print(Top5)
    print(GR)
    print(VaR_95)
# main function
def project():
    path1="C:\Users\Kai Wang\OneDrive\classes\Algo\project\\NYU_XAU_USD_20142015.csv"
    path2="C:\Users\Kai Wang\OneDrive\classes\Algo\project\\NYU_USD_JPY_2016.csv"
    
    XAU=Data(path1,55,10,40,2)
    Performance(Bollinger_band_strategy_filter(XAU,-0.02))
#    for i in (1,3,5):
#        print(i)
#        Bollinger_band_strategy_filter(XAU,-0.02,0.0001*i)
#        Bollinger_band_strategy_filter(XAU,-0.02,0,i)
    
    JPY=Data(path2,40,40,80,2)
    Performance(Bollinger_band_strategy_filter(JPY,-0.005))
#    for i in (1,3,5):
#        print(i)
#        Bollinger_band_strategy_filter(JPY,-0.005,0.0001*i)
#        Bollinger_band_strategy_filter(JPY,-0.005,0,i)


if __name__ == '__main__':
    project()