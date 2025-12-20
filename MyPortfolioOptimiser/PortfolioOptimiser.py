from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from datetime import datetime, timedelta
from collections.abc import Iterable
import yfinance as yf
import numpy as np
import math
import sys
import matplotlib
from  MyMatrixLibrary.MatrixModule import MatrixMultiplication as MM
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')

#initialising global variables
rf=0.02 #rf is the risk-free rate
yesterday=datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
one_year_ago=datetime.strftime(datetime.now() - timedelta(365), '%Y-%m-%d')

#class of Individual Stock Performance
class InvdPerformanceCalc:
    default_ticker='AAPL'
    default_log_returns= 0.045
    def __init__(self, ticker, log_returns):
      self.ticker = ticker
      self.log_returns = log_returns

    def calculate_annualised_mean(self):
        daily_mean = self.log_returns[self.ticker].mean()
        return math.exp(daily_mean * 2_52)-1

    def calculate_annualised_std(self):
        daily_std = self.log_returns[self.ticker].std()
        return daily_std * math.sqrt(252)

def opt_pf(user_input, user_constraint=1e-01):
  user_input_clean = user_input.upper().replace(" ", "")
  tickers=list(set(user_input_clean.split(',')))

  #Download data from yfinance
  data = yf.download(tickers, start=one_year_ago, end=yesterday)
  adjusted_close = (data["Close"])

  #Calculating Daily Log Returns using Pandas
  log_returns = np.log(adjusted_close/adjusted_close.shift(1))

  #Removing any empty data
  log_returns = log_returns.dropna()

  performance_dict = {} #Dictionary format is '{ticker}': [ticker_mean, ticker_std]

  for ticker in tickers:
    print(ticker, type(ticker))
    ticker_performance = InvdPerformanceCalc(ticker, log_returns)
    ticker_mean = ticker_performance.calculate_annualised_mean() # Calculate Annual expected return of asset
    ticker_std = ticker_performance.calculate_annualised_std() # Calculate Annual standard deviation of asset
    print(f"The mean of {ticker} is: {ticker_mean}.\nThe standard deviation of {ticker} is: {ticker_std}.")
    performance_dict[ticker] = (float(ticker_mean),float(ticker_std)) # Store calculations into a dictionary

  # Covariance Matrix
  covariance_matrix = log_returns.cov()

  number_of_assets=len(tickers)

  mu_transpose=np.array([list(list(zip(*performance_dict.values()))[0])])

  e_transpose=np.array([[1]*number_of_assets])

  e=np.array([[1] for x in range(number_of_assets)])

  mu_star=float(user_constraint)

  cov_matrix=np.array(covariance_matrix) 

  cov_matrix_inverse=np.linalg.inv(cov_matrix)

  mu=np.array([[mu_transpose[0][x]] for x in range(number_of_assets)])
  
  print(isinstance(mu, Iterable))

  a = MM(MM(e_transpose, cov_matrix_inverse),e)[0][0]

  b = MM(MM(mu_transpose, cov_matrix_inverse),e)[0][0]

  c = MM(MM(mu_transpose, cov_matrix_inverse),mu)[0][0]

  lamb1=(c-b*mu_star) / (a*c - b**2)

  lamb2 = (a*mu_star-b) / (a*c - b**2)

  matrix1= MM(cov_matrix_inverse,e)

  matrix2= MM(cov_matrix_inverse,mu)

  scaled_matrix_1=np.array(matrix1)*lamb1

  scaled_matrix_2=np.array(matrix2)*lamb2

  w_star_efficient=list(map(lambda x, y: x + y, scaled_matrix_1, scaled_matrix_2))

  constrained_w_star_efficient = np.maximum(w_star_efficient, 0)

  new_sum = sum(constrained_w_star_efficient)

  if new_sum != 1:
    constrained_w_star_efficient=constrained_w_star_efficient * (1/ new_sum)

  #######min_var

  numerator= MM(cov_matrix_inverse, e)
  denominator=MM(MM(e_transpose, cov_matrix_inverse), e)
  print(numerator, denominator)
  w_star_min_var=numerator/(denominator[0][0])

  constrained_w_star_min_var = np.maximum(w_star_min_var, 0)

  new_sum = np.sum(constrained_w_star_min_var)

  if new_sum != 1:
    constrained_w_star_min_var = constrained_w_star_min_var * (1/ new_sum)

  #### tangency

  tangency_numerator=((c-b*rf)/\
                      (b-a*rf))-rf

  tangency_denominator=(a*rf**2 -
                         2 * b * rf + c)

  lamb=tangency_numerator/tangency_denominator

  scaled_e_transpose=e_transpose*rf

  difference_matrix= mu_transpose-scaled_e_transpose

  difference_matrix_transpose=\
    np.array([[difference_matrix[0][x]] for x in range(number_of_assets)])

  calculation=np.array(MM(cov_matrix_inverse,difference_matrix_transpose))

  w_star_tangency=calculation*lamb

  constrained_w_star_tangency = np.maximum(w_star_tangency, 0)

  new_sum = np.sum(constrained_w_star_tangency)

  if new_sum != 1:
    constrained_w_star_tangency = constrained_w_star_tangency * (1/ new_sum)

  def output_portfolio_dictionary(w_star): #replaced by fill_final_output_dict_efficient
    portfolio_dictionary={"final_weights": {}, "statistics": {"expected return":0, "standard deviation":0, "sharpe ratio":0}}
    for i in range(len(tickers)):
      portfolio_dictionary["final_weights"]={**portfolio_dictionary["final_weights"], tickers[i]: round(w_star[i].item(),4)}
    mu=MM(B=w_star, A=mu_transpose)[0][0].item()
    var=np.sqrt(np.dot(np.array(w_star).T, np.dot(covariance_matrix, w_star)))[0][0].item()
    portfolio_dictionary["statistics"]["expected return"] = round(mu,6)
    portfolio_dictionary["statistics"]["standard deviation"] = round(var,6)
    sharpe= (mu - rf)/ np.sqrt(var)
    portfolio_dictionary["statistics"]["sharpe ratio"] = round(sharpe.item(),6)
    return portfolio_dictionary
  
  efficient_portfolio_dictionary=output_portfolio_dictionary(w_star_efficient)

  negative_efficient_portfolio_dictionary=output_portfolio_dictionary(constrained_w_star_efficient)

  min_var_portfolio_dictionary=output_portfolio_dictionary(w_star_min_var)

  negative_min_var_portfolio_dictionary=output_portfolio_dictionary(constrained_w_star_min_var)

  tangency_portfolio_dictionary=output_portfolio_dictionary(w_star_tangency)

  negative_tangency_portfolio_dictionary=output_portfolio_dictionary(constrained_w_star_tangency)

  return efficient_portfolio_dictionary, negative_efficient_portfolio_dictionary, min_var_portfolio_dictionary, negative_min_var_portfolio_dictionary, tangency_portfolio_dictionary, negative_tangency_portfolio_dictionary

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class Ui_MainWindow(object):
    def plot_data(self, weights_dict):
        tickers = list(weights_dict.keys())
        weights = list(weights_dict.values())
        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.bar(tickers, weights, color=['blue', 'green', 'red'] + ['purple'])
        self.plot_canvas.axes.set_title('Portfolio Weights')
        self.plot_canvas.axes.set_xlabel('Tickers')
        self.plot_canvas.axes.set_ylabel('Weights')
        self.plot_canvas.axes.axhline(0, color='black', linewidth=0.8, linestyle='--')
        self.plot_canvas.draw()

    def optimize_portfolio_efficient(self):
        print(3*"Optimising Portfolio...\n")
        try:
            user_input = self.ticker_line_edit.text()
            if not self.validate_tickers(user_input):
                raise ValueError("Invalid ticker symbols")
            user_constraints = self.desired_return_line_edit.text()
            if self.allow_negative_weights_checkbox.isChecked():
                final=opt_pf(user_input, user_constraints)[0]
            else:
                final=opt_pf(user_input, user_constraints)[1]
            self.value_weights_label.setText(str(final['final_weights'])[1:-1])
            self.plot_data(final['final_weights'])
            self.value_exp_return_label.setText(str(final['statistics']['expected return']))
            self.value_std_dev_label.setText(str(final['statistics']['standard deviation']))
            self.value_sharpe_label.setText(str(final['statistics']['sharpe ratio']))

        except ValueError as e:
            self.showErrorDialog(str(e))

    def optimize_portfolio_min_var(self):  
        try:
            user_input = self.ticker_line_edit.text()
            if not self.validate_tickers(user_input):
                raise ValueError("Invalid ticker symbols")
            final=opt_pf(user_input)[2] if self.allow_negative_weights_checkbox.isChecked() else opt_pf(user_input)[3]
            self.value_weights_label.setText(str(final['final_weights'])[1:-1])
            self.plot_data(final['final_weights'])
            self.value_exp_return_label.setText(str(final['statistics']['expected return']))
            self.value_std_dev_label.setText(str(final['statistics']['standard deviation']))
            self.value_sharpe_label.setText(str(final['statistics']['sharpe ratio']))
        except ValueError as e:
            self.showErrorDialog(str(e))

    def optimize_portfolio_tangency(self):
        try:
            user_input = self.ticker_line_edit.text()
            if not self.validate_tickers(user_input):
                raise ValueError("Invalid ticker symbols")
            if self.allow_negative_weights_checkbox.isChecked():
                final=opt_pf(user_input)[4]
            else:
                final=opt_pf(user_input)[5]
            self.value_weights_label.setText('' + str(final['final_weights'])[1:-1])
            self.plot_data(final['final_weights'])
            self.value_exp_return_label.setText(str(final['statistics']['expected return']))
            self.value_std_dev_label.setText(str(final['statistics']['standard deviation']))
            self.value_sharpe_label.setText(str(final['statistics']['sharpe ratio']))
        except ValueError as e:
            self.showErrorDialog(str(e))

    def validate_tickers(self, tickers):
        tickers_list = tickers.split(',')
        i=0
        while i in range(len(tickers_list)):
            ticker=tickers_list[i]
            ticker = ticker.strip()
            if not ticker.isalpha():
                return False
            i=i+1
        return True
    
    def showErrorDialog(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText("Error")
        error_dialog.setInformativeText(message)
        error_dialog.setWindowTitle("Error")
        error_dialog.exec_()
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1203, 906)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Graph widget
        self.graph_widget = QtWidgets.QWidget(self.centralwidget)
        self.graph_widget.setAutoFillBackground(True)
        self.graph_widget.setObjectName("graph_widget")
        self.graph_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Layout widget
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setObjectName("layoutWidget")
        self.upper_vlayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.upper_vlayout.setContentsMargins(0, 0, 0, 0)
        self.upper_vlayout.setObjectName("upper_vlayout")

        # Title instructions frame
        self.title_instructions_frame = QtWidgets.QFrame(self.layoutWidget)
        self.title_instructions_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.title_instructions_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.title_instructions_frame.setObjectName("title_instructions_frame")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.title_instructions_frame)
        self.verticalLayout_4.setObjectName("verticalLayout_4")

        # Title layout
        self.title_hlayout = QtWidgets.QHBoxLayout()
        self.title_hlayout.setObjectName("title_hlayout")
        spacerItem = QtWidgets.QSpacerItem(208, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.title_hlayout.addItem(spacerItem)
        self.title_label = QtWidgets.QLabel(self.title_instructions_frame)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.title_label.setFont(font)
        self.title_label.setObjectName("title_label")
        self.title_hlayout.addWidget(self.title_label)
        spacerItem1 = QtWidgets.QSpacerItem(218, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.title_hlayout.addItem(spacerItem1)
        self.verticalLayout_4.addLayout(self.title_hlayout)

        # Instructions layout
        self.instruct_vlayout = QtWidgets.QVBoxLayout()
        self.instruct_vlayout.setObjectName("instruct_vlayout")
        self.instruction_header_label = QtWidgets.QLabel(self.title_instructions_frame)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.instruction_header_label.setFont(font)
        self.instruction_header_label.setObjectName("instruction_header_label")
        self.instruct_vlayout.addWidget(self.instruction_header_label)
        self.instruct1_label = QtWidgets.QLabel(self.title_instructions_frame)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.instruct1_label.setFont(font)
        self.instruct1_label.setObjectName("instruct1_label")
        self.instruct_vlayout.addWidget(self.instruct1_label)
        self.instruct2_label = QtWidgets.QLabel(self.title_instructions_frame)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.instruct2_label.setFont(font)
        self.instruct2_label.setObjectName("instruct2_label")
        self.instruct_vlayout.addWidget(self.instruct2_label)
        self.instruct3_label = QtWidgets.QLabel(self.title_instructions_frame)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.instruct3_label.setFont(font)
        self.instruct3_label.setObjectName("instruct3_label")
        self.instruct_vlayout.addWidget(self.instruct3_label)
        self.verticalLayout_4.addLayout(self.instruct_vlayout)
        self.upper_vlayout.addWidget(self.title_instructions_frame)

        # Input frame
        self.input_frame = QtWidgets.QFrame(self.layoutWidget)
        self.input_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.input_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.input_frame.setObjectName("input_frame")
        self.input_frame_hlayout = QtWidgets.QHBoxLayout(self.input_frame)
        self.input_frame_hlayout.setObjectName("input_frame_hlayout")

        # Ticker line edit
        self.ticker_line_edit = QtWidgets.QLineEdit(self.input_frame)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setItalic(True)
        self.ticker_line_edit.setFont(font)
        self.ticker_line_edit.setObjectName("ticker_line_edit")
        self.ticker_line_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.input_frame_hlayout.addWidget(self.ticker_line_edit)

        # Desired return line edit
        self.desired_return_line_edit = QtWidgets.QLineEdit(self.input_frame)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setItalic(True)
        self.desired_return_line_edit.setFont(font)
        self.desired_return_line_edit.setObjectName("desired_return_line_edit")
        self.desired_return_line_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.input_frame_hlayout.addWidget(self.desired_return_line_edit)
        self.input_frame_hlayout.addWidget(self.ticker_line_edit, stretch=3) 
        self.input_frame_hlayout.addWidget(self.desired_return_line_edit, stretch=1)
        self.upper_vlayout.addWidget(self.input_frame)

        # Buttons frame
        self.buttons_frame = QtWidgets.QFrame(self.layoutWidget)
        self.buttons_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.buttons_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.buttons_frame.setObjectName("buttons_frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.buttons_frame)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        # Buttons
        self.min_var_button = QtWidgets.QPushButton(self.buttons_frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.min_var_button.setFont(font)
        self.min_var_button.setObjectName("min_var_button")
        self.min_var_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.horizontalLayout_2.addWidget(self.min_var_button)
        self.min_var_button.clicked.connect(self.optimize_portfolio_min_var)
        self.min_var_button_2 = QtWidgets.QPushButton(self.buttons_frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.min_var_button_2.setFont(font)
        self.min_var_button_2.setObjectName("min_var_button_2")
        self.min_var_button_2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.horizontalLayout_2.addWidget(self.min_var_button_2)
        self.min_var_button_2.clicked.connect(self.optimize_portfolio_efficient)
        self.min_var_button_3 = QtWidgets.QPushButton(self.buttons_frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.min_var_button_3.setFont(font)
        self.min_var_button_3.setObjectName("min_var_button_3")
        self.min_var_button_3.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.horizontalLayout_2.addWidget(self.min_var_button_3)
        self.upper_vlayout.addWidget(self.buttons_frame)
        self.min_var_button_3.clicked.connect(self.optimize_portfolio_tangency)    

        # Results layout
        self.results_layout = QtWidgets.QVBoxLayout()
        self.results_layout.setObjectName("results_layout")

        # Results header label
        self.results_header_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.results_header_label.setFont(font)
        self.results_header_label.setObjectName("results_header_label")
        self.results_layout.addWidget(self.results_header_label)
        self.allow_negative_weights_checkbox = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(12)
        self.allow_negative_weights_checkbox.setFont(font)
        self.allow_negative_weights_checkbox.setObjectName("allow_negative_weights_checkbox")
        self.allow_negative_weights_checkbox.setText("Allow Negative Weights")
        self.results_layout.addWidget(self.allow_negative_weights_checkbox)
        
        # Result weights header label
        self.result_weights_header_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.result_weights_header_label.setFont(font)
        self.result_weights_header_label.setObjectName("result_weights_header_label")
        self.results_layout.addWidget(self.result_weights_header_label)

        # Value weights label
        self.value_weights_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(12)
        self.value_weights_label.setFont(font)
        self.value_weights_label.setObjectName("value_weights_label")
        self.results_layout.addWidget(self.value_weights_label)

        # Stats header label
        self.stats_header_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.stats_header_label.setFont(font)
        self.stats_header_label.setObjectName("stats_header_label")
        self.results_layout.addWidget(self.stats_header_label)

        # Splitter layout
        self.splitter_layout = QtWidgets.QHBoxLayout()
        self.splitter_layout.setObjectName("splitter_layout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.widget = QtWidgets.QWidget(self.splitter)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.exp_return_label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.exp_return_label.setFont(font)
        self.exp_return_label.setObjectName("exp_return_label")
        self.verticalLayout_2.addWidget(self.exp_return_label)
        self.std_dev_label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.std_dev_label.setFont(font)
        self.std_dev_label.setObjectName("std_dev_label")
        self.verticalLayout_2.addWidget(self.std_dev_label)
        self.sharpe_label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.sharpe_label.setFont(font)
        self.sharpe_label.setObjectName("sharpe_label")
        self.verticalLayout_2.addWidget(self.sharpe_label)
        self.widget1 = QtWidgets.QWidget(self.splitter)
        self.widget1.setObjectName("widget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.value_exp_return_label = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(11)
        self.value_exp_return_label.setFont(font)
        self.value_exp_return_label.setObjectName("value_exp_return_label")
        self.verticalLayout.addWidget(self.value_exp_return_label)
        self.value_std_dev_label = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(11)
        self.value_std_dev_label.setFont(font)
        self.value_std_dev_label.setObjectName("value_std_dev_label")
        self.verticalLayout.addWidget(self.value_std_dev_label)
        self.value_sharpe_label = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(11)
        self.value_sharpe_label.setFont(font)
        self.value_sharpe_label.setObjectName("value_sharpe_label")
        self.verticalLayout.addWidget(self.value_sharpe_label)
        self.splitter_layout.addWidget(self.splitter)
        self.results_layout.addLayout(self.splitter_layout)
        self.plot_canvas = MplCanvas(self.centralwidget, width=8, height=4, dpi=100)
        self.results_layout.addWidget(self.plot_canvas)

        # Add layouts to central widget
        self.central_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.central_layout.addWidget(self.layoutWidget)
        self.central_layout.addLayout(self.results_layout)
        self.central_layout.addWidget(self.graph_widget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1203, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title_label.setText(_translate("MainWindow", "Asset Portfolio Optimizer"))
        self.instruction_header_label.setText(_translate("MainWindow", "Instructions"))
        self.instruct1_label.setText(_translate("MainWindow", "Type in the stock tickers of the assets in your portfolio as a list (ie. 2 stock portfolio: BA, PFE)."))
        self.instruct2_label.setText(_translate("MainWindow", "If you want an efficient portfolio with constrained expected return, please input your desired return as a decimal."))
        self.instruct3_label.setText(_translate("MainWindow", "Click an optimization to perform and view the results!"))
        self.ticker_line_edit.setPlaceholderText("Insert stock tickers (ie. AAPL, AMZ)")
        self.ticker_line_edit.clear() 
        self.desired_return_line_edit.setPlaceholderText("Desired return")
        self.desired_return_line_edit.clear()  
        self.min_var_button.setText(_translate("MainWindow", "Minimize Variance"))
        self.min_var_button_2.setText(_translate("MainWindow", "Minimize Variance and Constraint"))
        self.min_var_button_3.setText(_translate("MainWindow", "Tangency Portfolio"))
        self.results_header_label.setText(_translate("MainWindow", "Results"))
        self.result_weights_header_label.setText(_translate("MainWindow", "Weights"))
        self.value_weights_label.setText(_translate("MainWindow", ""))
        self.stats_header_label.setText(_translate("MainWindow", "Statistics"))
        self.exp_return_label.setText(_translate("MainWindow", "Expected Return:"))
        self.std_dev_label.setText(_translate("MainWindow", "Standard Deviation:"))
        self.sharpe_label.setText(_translate("MainWindow", "Sharpe Ratio:"))
        self.value_exp_return_label.setText(_translate("MainWindow", ""))
        self.value_std_dev_label.setText(_translate("MainWindow", ""))
        self.value_sharpe_label.setText(_translate("MainWindow", ""))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())