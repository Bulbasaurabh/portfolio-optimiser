from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import math
import sys
import matplotlib
from  MyMatrixLibrary.MatrixModule import MatrixMultiplication as MM
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#initialising global variables
rf=0.02
yesterday=datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
one_year_ago=datetime.strftime(datetime.now() - timedelta(365), '%Y-%m-%d')

class InvdPerformanceCalc:
    def __init__(self, ticker, log_returns):
      self.ticker = ticker
      self.log_returns = log_returns

    def calculate_annualised_mean(self):
        daily_mean = self.log_returns[self.ticker].mean()
        return math.exp(daily_mean * 252)-1

    def calculate_annualised_std(self):
        daily_std = self.log_returns[self.ticker].std()
        return daily_std * math.sqrt(252)



def opt_pf(user_input, user_constraint=0.1):
  user_input='APPL,      AMZ'
  user_input_clean = user_input.replace(" ", "")

  #Data from yfinance
  data = yf.download(tickers, start=one_year_ago, end=yesterday)
  adjusted_close = (data["Close"])

  #Calculating Daily Log Returns using Pandas
  log_returns = np.log(adjusted_close/adjusted_close.shift(1))

  #Removing any empty data
  log_returns = log_returns.dropna()

  performance_dict = {} #Dictionary format is '{ticker}': [ticker_mean, ticker_std]

  for ticker in tickers:
    ticker_performance = InvdPerformanceCalc(ticker, log_returns)
    ticker_mean = ticker_performance.calculate_annualised_mean() # Calculate Annual expected return of asset
    ticker_std = ticker_performance.calculate_annualised_std() # Calculate Annual standard deviation of asset
    print(f"The mean of {ticker} is: {ticker_mean}.\nThe standard deviation of {ticker} is: {ticker_std}.")

    performance_dict[ticker] = [float(ticker_mean),float(ticker_std)] # Store calculations into a dictionary

  print(performance_dict)

  # 1. Correlation Matrix
  correlation_matrix = log_returns.corr()
  print("\nCorrelation Matrix:""\n", correlation_matrix)

  # 2. Covariance Matrix
  covariance_matrix = log_returns.cov()
  print("\nCovariance Matrix:\n", covariance_matrix)

  no_assets=len(tickers)

  mu_transpose=np.array([[x[0] for x in performance_dict.values()]])

  e_transpose=np.array([[1]*no_assets])

  e=np.array([[1] for x in range(no_assets)])

  mu_star=float(user_constraint)

  cov_matrix=np.array(covariance_matrix) 

  cov_matrix_inverse=np.linalg.inv(cov_matrix)

  mu=np.array([[mu_transpose[0][x]] for x in range(no_assets)])

  a = MM(MM(e_transpose, cov_matrix_inverse),e)[0][0]

  b = MM(MM(mu_transpose, cov_matrix_inverse),e)[0][0]

  c = MM(MM(mu_transpose, cov_matrix_inverse),mu)[0][0]

  const1=(c-b*mu_star) / (a*c - b**2)

  const2 = (a*mu_star-b) / (a*c - b**2)

  calc1= MM(cov_matrix_inverse,e)

  calc2= MM(cov_matrix_inverse,mu)

  calc3=np.array(calc1)*const1

  calc4=np.array(calc2)*const2

  w_star=list(map(lambda x, y: x + y, calc3, calc4))

  print("portfolio weights with negative weights:", w_star, sum(w_star))
  def fill_final_output_dict_efficient(w_star):
    final_output_dict_efficient={"final_weights": {}, "statistics": {"expected return":0, "standard deviation":0, "sharpe ratio":0}}
    for i in range(len(tickers)):
      final_output_dict_efficient["final_weights"]={**final_output_dict_efficient["final_weights"], tickers[i]: w_star[i].item()}
    final_output_dict_efficient["statistics"]["expected return"] = MM(B=w_star, A=mu_transpose)[0][0].item()
    final_output_dict_efficient["statistics"]["standard deviation"] = np.sqrt(np.dot(np.array(w_star).T, np.dot(covariance_matrix, w_star)))[0][0].item()
    sharpe= (final_output_dict_efficient["statistics"]["expected return"] - rf)/ np.sqrt(final_output_dict_efficient["statistics"]["standard deviation"])
    final_output_dict_efficient["statistics"]["sharpe ratio"] = sharpe.item()
    return final_output_dict_efficient
  final_output_dict_efficient=fill_final_output_dict_efficient(w_star)
  print(final_output_dict_efficient)

  w_star_1 = np.maximum(w_star, 0)

  new_sum = np.sum(w_star_1)

  if new_sum > 0:
    w_star_1 = w_star_1 * (1/ new_sum)

  print("portfolio weights without negative weights:", list(w_star_1), sum(w_star_1))

  def fill_final_output_dict_efficient_neg(w_star):
    final_output_dict_efficient_neg={"final_weights": {}, "statistics": {"expected return":0, "standard deviation":0, "sharpe ratio":0}}
    for i in range(len(tickers)):
      final_output_dict_efficient_neg["final_weights"]={**final_output_dict_efficient_neg["final_weights"], tickers[i]: w_star[i].item()}
    final_output_dict_efficient_neg["statistics"]["expected return"] = MM(B=w_star, A=mu_transpose)[0][0].item()
    final_output_dict_efficient_neg["statistics"]["standard deviation"] = np.sqrt(np.dot(np.array(w_star).T, np.dot(covariance_matrix, w_star)))[0][0].item()
    sharpe= (final_output_dict_efficient_neg["statistics"]["expected return"] - rf)/ np.sqrt(final_output_dict_efficient_neg["statistics"]["standard deviation"])
    final_output_dict_efficient_neg["statistics"]["sharpe ratio"] = sharpe.item()
    return final_output_dict_efficient_neg
  final_output_dict_efficient_neg=fill_final_output_dict_efficient_neg(w_star_1)
  print(final_output_dict_efficient_neg)

  #######min_var

  num= MM(cov_matrix_inverse, e)
  den=MM(MM(e_transpose, cov_matrix_inverse), e)

  final=num/den[0][0]

  def fill_final_output_dict_min_var(w_star):
    final_output_dict_min_var={"final_weights": {}, "statistics": {"expected return":0, "standard deviation":0, "sharpe ratio":0}}
    for i in range(len(tickers)):
      final_output_dict_min_var["final_weights"]={**final_output_dict_min_var["final_weights"], tickers[i]: w_star[i].item()}
    final_output_dict_min_var["statistics"]["expected return"] = MM(B=w_star, A=mu_transpose)[0][0].item()
    final_output_dict_min_var["statistics"]["standard deviation"] = np.sqrt(np.dot(np.array(w_star).T, np.dot(covariance_matrix, w_star)))[0][0].item()
    sharpe= (final_output_dict_min_var["statistics"]["expected return"] - rf)/ np.sqrt(final_output_dict_min_var["statistics"]["standard deviation"])
    final_output_dict_min_var["statistics"]["sharpe ratio"] = sharpe.item()
    return final_output_dict_min_var

  final_output_dict_min_var=fill_final_output_dict_min_var(final)
  print(final_output_dict_min_var)

  final_1 = np.maximum(final, 0)

  new_sum = np.sum(final_1)

  if new_sum > 0:
    final_1 = final_1 * (1/ new_sum)

  def fill_final_output_dict_min_var_neg(w_star):
    final_output_dict_min_var_neg={"final_weights": {}, "statistics": {"expected return":0, "standard deviation":0, "sharpe ratio":0}}
    for i in range(len(tickers)):
      final_output_dict_min_var_neg["final_weights"]={**final_output_dict_min_var_neg["final_weights"], tickers[i]: w_star[i].item()}
    final_output_dict_min_var_neg["statistics"]["expected return"] = MM(B=w_star, A=mu_transpose)[0][0].item()
    final_output_dict_min_var_neg["statistics"]["standard deviation"] = np.sqrt(np.dot(np.array(w_star).T, np.dot(covariance_matrix, w_star)))[0][0].item()
    sharpe= (final_output_dict_min_var_neg["statistics"]["expected return"] - rf)/ np.sqrt(final_output_dict_min_var_neg["statistics"]["standard deviation"])
    final_output_dict_min_var_neg["statistics"]["sharpe ratio"] = sharpe.item()
    return final_output_dict_min_var_neg

  final_output_dict_min_var_neg=fill_final_output_dict_min_var_neg(final_1)
  print(final_output_dict_min_var_neg)


  #### tangency


  a = MM(MM(e_transpose, cov_matrix_inverse),e)[0][0]
  b = MM(MM(mu_transpose, cov_matrix_inverse),e)[0][0]
  c = MM(MM(mu_transpose, cov_matrix_inverse),mu)[0][0]

  num=((c-b*rf)/(b-a*rf))-rf

  den=a*rf**2 - 2 * b * rf + c

  lamb=num/den

  var_min= den / (b-a*rf)**2

  print("minimised variance is:", var_min)
  e_transpose_new=e_transpose*rf

  calc7= mu_transpose-e_transpose_new

  calc7_t=np.array([[calc7[0][x]] for x in range(no_assets)])

  calc8=np.array(MM(cov_matrix_inverse,calc7_t))

  w=calc8*lamb

  print("optimised weights are:", w, sum(w))

  def fill_final_output_dict_tangency(w_star):
    final_output_dict_tangency={"final_weights": {}, "statistics": {"expected return":0, "standard deviation":0, "sharpe ratio":0}}
    for i in range(len(tickers)):
      final_output_dict_tangency["final_weights"]={**final_output_dict_tangency["final_weights"], tickers[i]: w_star[i].item()}
    final_output_dict_tangency["statistics"]["expected return"] = MM(B=w_star, A=mu_transpose)[0][0].item()
    final_output_dict_tangency["statistics"]["standard deviation"] = np.sqrt(np.dot(np.array(w_star).T, np.dot(covariance_matrix, w_star)))[0][0].item()
    sharpe= (final_output_dict_tangency["statistics"]["expected return"] - rf)/ np.sqrt(final_output_dict_tangency["statistics"]["standard deviation"])
    final_output_dict_tangency["statistics"]["sharpe ratio"] = sharpe.item()
    return final_output_dict_tangency
  final_output_dict_tangency=fill_final_output_dict_tangency(w)
  print(final_output_dict_tangency)

  w_1 = np.maximum(w, 0)

  new_sum = np.sum(w_1)

  if new_sum > 0:
    w_1 = w_1 * (1/ new_sum)

  print("portfolio weights without negative weights:", list(w_1), sum(w_1))

  def fill_final_output_dict_tangency_neg(w_star):
    final_output_dict_tangency_neg={"final_weights": {}, "statistics": {"expected return":0, "standard deviation":0, "sharpe ratio":0}}
    for i in range(len(tickers)):
      final_output_dict_tangency_neg["final_weights"]={**final_output_dict_tangency_neg["final_weights"], tickers[i]: w_star[i].item()}
    final_output_dict_tangency_neg["statistics"]["expected return"] = MM(B=w_star, A=mu_transpose)[0][0].item()
    final_output_dict_tangency_neg["statistics"]["standard deviation"] = np.sqrt(np.dot(np.array(w_star).T, np.dot(covariance_matrix, w_star)))[0][0].item()
    sharpe= (final_output_dict_tangency_neg["statistics"]["expected return"] - rf)/ np.sqrt(final_output_dict_tangency_neg["statistics"]["standard deviation"])
    final_output_dict_tangency_neg["statistics"]["sharpe ratio"] = sharpe.item()
    return final_output_dict_tangency_neg
  final_output_dict_tangency_neg=fill_final_output_dict_tangency_neg(w_1)
  print(final_output_dict_tangency_neg)

  return final_output_dict_efficient, final_output_dict_efficient_neg, final_output_dict_min_var, final_output_dict_min_var_neg, final_output_dict_tangency, final_output_dict_tangency_neg