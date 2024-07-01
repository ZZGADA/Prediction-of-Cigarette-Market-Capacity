import os

import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import sys
from sklearn.svm import SVR as svr
from sklearn.metrics import get_scorer_names as scoreName
from sklearn import preprocessing as prep
sys.path.append(r"../")  #返回上级目录
import Config
from Dao import Data_Acquisition as DA
from Service import Bass_Model as BM
from Service import Model1_province_sku as Model1
from Service import Model2_sample_province as Model2
from Service import Characteristic_Mapping as CM
from DataAnalysis import Projected_evaluation_indicators as pei
from Dao import Data_Processing as Dpcs
from math import e
from sklearn.model_selection import GridSearchCV
import warnings
from copy import deepcopy
from Service import Model3_region as Model3

def template_plot_Bass_model_predict_sales_old():

    path_head_old = "../源文件/地市/销量预测误差统计_test/各个SKU的预测销量统计/老品/老品(curve_fit)/老品阈值处理(剔除五年平均销量小于20None)/"
    original_data_old_path = "../源文件/地市/销量预测误差统计_test/各个SKU的预测销量统计/老品/原始销量统计.csv"
    os.makedirs(path_head_old + "图片", exist_ok=True)
    target_sku_name = ["南京_利群(软红长嘴)","南京_利群(阳光)","宿迁_利群(新版)","南京_利群(新版)","常州_利群(长嘴)","泰州_利群(西子阳光)","常州_利群(新版)","常州_利群(硬)","淮安_利群(西子阳光)","淮安_利群(硬)"]


    for sku_name in target_sku_name:
        figure = plt.figure()
        original_data = pd.read_csv(original_data_old_path, index_col=0)[sku_name]
        original_data = original_data[original_data > 0]
        year_get = original_data.index.to_list()
        plt.plot(year_get, original_data, "o-", alpha=0.8, color="#053AC4", label="原始数据", linewidth=1)

        model_sales_data=pd.read_csv(path_head_old + "老品预测销量统计(XG).csv" , index_col=0)[sku_name].loc[year_get]

        plt.plot(year_get, model_sales_data, "o-", alpha=0.8, color="#BA1CAB",
                     label="Bass" , linewidth=1)

        plt.legend(loc="upper right")
        plt.title(sku_name)
        plt.xlabel("年份")
        plt.xticks(year_get)
        plt.ylabel("销量")
        plt.savefig(path_head_old + "图片" + "/%s" % sku_name, dpi=300)
def template_plot_each_model_predict_sales_new():
    path_head_new="../源文件/地市/销量预测误差统计_test/各个SKU的预测销量统计/新品/新品阈值处理(剔除五年平均销量小于20None)/"


    original_data_new_path="../源文件/地市/销量预测误差统计_test/各个SKU的预测销量统计/新品/原始销量统计.csv"


    os.makedirs(path_head_new+"图片",exist_ok=True)
    target_sku_name=["淮安_利群(楼外楼)","泰州_利群(休闲)","南通_利群(江南韵)","宿迁_利群(西子阳光)","宿迁_利群(西子阳光)","淮安_利群(休闲细支)","连云港_利群(楼外楼)","连云港_利群(硬)","南通_利群(阳光橙中支)","苏州_利群(休闲云端)"]
    # target_sku_name = [ "泰州_利群(休闲)"]
    model_name_all=["SVR","RT","KNN","XG"]
    model_name_change=dict(zip(model_name_all,["SVR","RF","KNN","XG"]))
    color=["#F9F871","#FF795B","#FE3581","#BA1CAB"]
    model_color=dict(zip(model_name_all,color))


    for sku_name in target_sku_name:
        figure=plt.figure()
        original_data=pd.read_csv(original_data_new_path,index_col=0)[sku_name]
        original_data=original_data[original_data > 0]
        year_get=original_data.index.to_list()
        plt.plot(year_get, original_data, "o-", alpha=0.8, color="#053AC4", label="原始数据", linewidth=1)
        for model_name in model_name_all:
            if model_name=="SVR":
                continue
            model_sales_data=pd.read_csv(path_head_new+"新品预测销量统计(%s).csv"%model_name,index_col=0)[sku_name].loc[year_get]

            plt.plot(year_get,model_sales_data,"o-", alpha=0.8, color=model_color[model_name], label="%s"%model_name_change[model_name], linewidth=1)

        plt.legend(loc="upper right")
        plt.title(sku_name)
        plt.xlabel("年份")
        plt.xticks(year_get)
        plt.ylabel("销量")
        plt.savefig(path_head_new+"图片"+"/%s"%sku_name,dpi=300)

if __name__=="__main__":
    template_plot_Bass_model_predict_sales_old()