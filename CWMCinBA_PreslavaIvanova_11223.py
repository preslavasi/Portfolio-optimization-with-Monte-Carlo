# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:19:28 2019

@author: Preslava Ivanova, 11223
"""

# PART 1 - Importing the financial data
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# Importing data from Yahoo Finance 
stocks = ['CSCO', 'DEO', 'HLT', 'JNJ', 'JPM', 'KO', 'MAR', 'NFLX', 'TTM', 'V']
data = web.DataReader(stocks,data_source="yahoo",start='01/01/2015',end='05/24/2019')['Adj Close']

# Computing stock returns in percentage format
log_returns = np.log(1 + data.pct_change())
log_returns.tail()
data.plot(figsize=(10, 6)) # stocks prices
log_returns.plot(figsize = (10, 6)) # stocks log returns - stable mean, normally distributed
plt.legend(log_returns,loc='upper left')

# Stationarity test
summary=log_returns.describe()
X = log_returns
split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print(mean1, mean2)
print(var1, var2)

# Calculating mean returns and covariances for all of the stocks
mean_log_returns = log_returns.mean()
print (log_returns.round(4)*100)
cov_matrix = log_returns.cov()
print (mean_log_returns)
print (cov_matrix)

# Removing stocks with negative mean_log_returns
stock=['CSCO', 'DEO', 'HLT', 'JNJ', 'JPM', 'KO', 'MAR', 'NFLX', 'V']
data_new=data.drop(['TTM'],axis=1)
data_new.columns=stock

# Computing new optimization with improved portfolio
log_returns_new = np.log(1 + data_new.pct_change())
log_returns_new.tail()
data.plot(figsize=(10, 6)) # stocks prices
log_returns_new.plot(figsize = (10, 6)) # stocks log returns - stable mean, normally distributed

# Checking mean returns and covariances for all of the stocks from the subportfolio
mean_log_returns_new = log_returns_new.mean()
print (log_returns_new.round(4)*100)
cov_matrix_new = log_returns_new.cov()
print (mean_log_returns_new) # all of the values are positive so we could go further
print (cov_matrix_new)

#Set the number of iterations to 10000 and define an array to hold the simulation results; initially set to all zeros
num_iterations = 12000
simulation_res = np.zeros((4+len(stock)-1,num_iterations))
for i in range(num_iterations):
#Select random weights and normalize to set the sum to 1
        weights = np.array(np.random.random(9))
        weights /= np.sum(weights)
#Calculate the return and standard deviation for every step
        portfolio_return = np.sum(mean_log_returns_new * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix_new, weights)))
#Store all the results in a defined array
        simulation_res[0,i] = portfolio_return
        simulation_res[1,i] = portfolio_std_dev
#Calculate Sharpe ratio and store it in the array
        simulation_res[2,i] = simulation_res[0,i] / simulation_res[1,i]
#Save the weights in the array
        for j in range(len(weights)):
                simulation_res[j+3,i] = weights[j]
sim_frame = pd.DataFrame(simulation_res.T,columns=['ret','stdev','sharpe','CSCO', 'DEO', 'HLT', 'JNJ', 'JPM', 'KO', 'MAR', 'NFLX', 'V'])
print sim_frame.head (5)
print sim_frame.tail (5)

# Position of the portfolio with highest Sharpe Ratio
max_sharpe = sim_frame.iloc[sim_frame['sharpe'].idxmax()]
print ("The portfolio for max Sharpe Ratio:\n", max_sharpe) # 
# Position of the portfolio with minimum Standard Deviation
min_std = sim_frame.iloc[sim_frame['stdev'].idxmin()]
print ("The portfolio for min risk:\n", min_std)

# Plotting the results with standart deviation on the x-axis and returns on the y-axis
plt.scatter(sim_frame.stdev,sim_frame.ret,c=sim_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.ylim(0.0003,0.0012)
plt.xlim(0.007200,0.016)
# Red star for the position with highest Sharpe Ratio
plt.scatter(max_sharpe[1],max_sharpe[0],marker=(5,1,0),color='r',s=400)
# Blue star for the position with minimum variance
plt.scatter(min_std[1],min_std[0],marker=(5,1,0),color='b',s=400)
plt.show()  

# PART 2 - Generating a Random Walk simulation

days = len(data_new)
cagr = ((((data_new.iloc[-1106]) / data_new.iloc[-1])) ** (365.0/days)) - 1
print ('CAGR =',portfolio_return.round(4)*100) # 0.1
mu = cagr # level of return for the assets
portfolio_return=max_sharpe[0]
stdev=max_sharpe[1]
vol=stdev*math.sqrt(252)
print ("Annual Volatility =",str(round(vol,4)*100)+"%") # 19.38%

# Defining the variables
S0 = data_new.iloc[-1106]
S = S0 # defining a starting point
T = 252 # number of trading days
mu = 0.0009574825007296051 # value of portfolio_return
vol = 0.19376265700655645 # value of volatility

# Creating list of daily returns using random normal distribution
daily_returns=np.random.normal((mu/T),vol/math.sqrt(T),T)+1
# Setting starting price and creating price series
price_list = [S]
for x in daily_returns:
    price_list.append(price_list[-1]*x)
# Generating price series plot and histogram of daily returns
plt.figure(figsize=(10,6))
plt.plot(price_list)
plt.ylabel('Stock Price, $')
plt.xlabel('Day')
plt.legend(data_new,loc='upper right')
plt.show()
plt.figure(figsize=(10,6))  
plt.hist(daily_returns-1, 100) # we run the line plot and histogram separately
plt.ylabel('Returns,%')
plt.xlabel('Day') 
plt.show()

# PART 3 - Brownian motion - r=drift+stdev*exp^r

# Calculating mean and variance
u = mean_log_returns_new.values
var = log_returns_new.var()

# Computing drift component 
drift = u - (0.5 * var)
weights_new=max_sharpe[3:,]
stdev=max_sharpe[1]
norm.ppf(0.95)
type(drift)
type(stdev)

# Calculating daily returns
t_intervals = 252 # future stock prices for 1 trade year
iterations = 9 # 9 series of future stock predictions, because we have 9 assets
daily_returns_BM = np.exp(np.array(drift) + (np.array(stdev) * norm.ppf(np.random.rand(t_intervals, iterations))))
daily_returns_BM

# Creating a price list
S0 = data_new.iloc[-1106]
price_list = np.zeros_like(daily_returns_BM)
price_list[0] = S0
print ('Price list', price_list)

# Generating values 
for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_returns_BM[t]
price_list_BM=pd.DataFrame(price_list)
price_list_BM.columns=stock
returns_BM=price_list_BM.pct_change()

# Plotting results
price_list_BM.plot(figsize=(10,6))
plt.ylabel('Stock Price, $')
plt.xlabel('Day')
plt.title('Brownian Motion prices simulation')
plt.figure(figsize=(10,6))
plt.plot(returns_BM.round(4)*100)
plt.xlabel('Day')
plt.ylabel('Portfolio return, %')
plt.legend(returns_BM,loc='upper left')
plt.title("Brownian motion for " + str(t_intervals) + " Days")
plt.show()

# Creating a histogramm
plt.hist(returns_BM)
plt.hist(price_list_BM)
plt.hist(daily_returns_BM)
returns_BM.mean()
returns_BM.var()
# Calculating Value at Risk
last_price = S0
price_array = price_list[-1, :]
price_array = sorted(price_array, key=int)  
var =  np.percentile(price_array, 1)
value_at_risk = last_price - var
print ("Value at Risk: ", value_at_risk)
value_at_risk.plot(figsize=(10,6))
plt.title('Value at Risk')
plt.ylabel('Stock price')
plt.xlabel('Day')
mean = np.mean(log_returns_new)
std = np.std(log_returns_new)
Z_99 = norm.ppf(1-0.99)
print(mean, std, Z_99, price_array)
price_array=np.asarray(price_array)
ParamVAR = price_array*Z_99*std
HistVAR = price_array*np.percentile(log_returns_new.dropna(), 1)
print('Parametric VAR is {0:.3f} and Historical VAR is {1:.3f}'
     (ParamVAR, HistVAR))

def port_var(y):
    #Calculate returns
    port_rets = daily_returns
  
    var =  np.percentile(port_rets, .01)
    var1 =  np.percentile(port_rets, 1)
    var2 =  np.percentile(port_rets, 5)
    
    #Output histogram
    plt.hist(port_rets,normed=True)
    plt.xlabel('Portfolio Returns')
    plt.ylabel('Frequency')
    plt.title(r'Histogram of Portfolio', fontsize=18, fontweight='bold')
    plt.axvline(x=var2, color='r', linestyle='--', label='Price at Confidence Interval: ' + str(round(var2, 2)))
    plt.legend(loc='upper right', fontsize = 'x-small')
    plt.show() 
    
    #VaR stats
    print "99.99% Confident the actual loss will not exceed: " + str(round(var, 2))
    print "99% Confident the actual loss will not exceed: " + str(round(var1, 2))
    print "95% Confident the actual loss will not exceed: " + str(round(var2, 2))
    
    print "Losses expected to exceed " + "{0:.2f}".format(var2) + " " + str(.05*len(port_rets)) + " out of " + str(len(port_rets)) + " days"      
varres=port_var(daily_returns)
# PART 4 - Geometric Brownian Motion
# Defining function
def GBM(S0, drift, stdev, W, T, N):    
    t = np.linspace(0.,1.,N+1)
    S = []
    S.append(S0)
    for i in xrange(1,int(N+1)):
        drift = (u - 0.5 * stdev**2) * t[i]
        diffusion = stdev* W[i-1]
        S_temp = S0*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t
# Defining parameters
W = price_list # the prices of the assets in the subportfolio
T = 1.
N = 252 # 252 trading days
solution = GBM(S0, mu, stdev, W, T, N)[0]    # Exact solution
t = GBM(S0, mu, stdev, W, T, N)[1]       # time increments for  plotting
predicted_prices_GBM=pd.DataFrame(solution)
predicted_prices_GBM.columns=stock
return_GBM=predicted_prices_GBM.pct_change()
# Plotting the results
plt.figure(figsize=(10,6)) 
plt.plot(t, solution)
plt.ylabel('Stock Price, $')
plt.title('Geometric Brownian Motion')
plt.legend(return_GBM,loc='upper left')
plt.show()
