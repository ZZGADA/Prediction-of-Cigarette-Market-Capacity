import math
import random
from math import e  # 引入自然数e
import numpy as np  # 科学计算库
import matplotlib.pyplot as plt  # 绘图库
import pandas as pd
import sklearn.datasets
from scipy import interpolate as itp
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
from scipy.optimize import leastsq, curve_fit
#科学计算库 机器学习标准库
import sklearn.preprocessing as preproc
import sympy as sp
from sklearn.model_selection import train_test_split  #训练集和测试集分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
# import sys
# sys.path.append(r"/DataAnalysis")  #返回上级目录
from DataAnalysis import Projected_evaluation_indicators as pei
# import plot_Bass_Model as pbm
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False