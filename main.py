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

#科学计算库 机器学习标准库
import sklearn.preprocessing as preproc

from sklearn.model_selection import train_test_split  #训练集和测试集分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
# import sys
# sys.path.append(r"/DataAnalysis")  #返回上级目录
# import plot_Bass_Model as pbm
from Dao import Data_Acquisition as DA
from Service import Bass_Model as BM
from DataAnalysis import Projected_evaluation_indicators as pei
from Dao import Data_Processing as Dpcs
from math import e
from Service import Characteristic_Mapping as CM
from DataAnalysis import Each_Machine_Model_MAPE as EM
from Service import Model3_region as Model3
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':
    print("it is main function ")

    start_year = 2006
    Data: pd.DataFrame
    rolling_start=2017
    rolling_end=2022

    # return_path= "../源文件/清洗后的全sku数据.csv"

    def rolling():
        for end_year in range(rolling_start, rolling_end):  # 2017-2022
            count = end_year - 2016
            print("这是第%d轮滚动" % count)
            path = "../源文件/地市/特征结果"
            Data = DA.get_region_data(path)
            # feature_supplementary=DA.get_region_supplementary_feature(path,end_year)

            path_clean = "../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/相关数据"
            return_path = Dpcs.data_PreProcessing(Data, start_year, end_year, count, path_clean)
            Data = DA.get_Data(return_path)
            attributes_path = "../源文件/property_feature_all.csv"
            self_attribute = DA.get_feature_region(attributes_path, index_col=0).iloc[:, :13]  # 提取特征属性表
            sku_feature_all, sku_all = CM.Discrete_data_processing_region(self_attribute,
                                                                          Data,
                                                                          count)  # 获得添加虚拟变量后的特征表 ,过滤掉没有特征的sku 第一遍筛选数据

            # sku_feature_all=CM.supplementary_Feature_Matching(sku_feature_all,feature_supplementary)
            sku_all = Model3.sku_classification(sku_all, start_year, end_year)  # 对sku进行 分类
            params_all = Model3.get_Bass_params(sku_all, start_year, end_year)  # 获得分类后的 bass参数
            Model3.sku_sales_index_processing(sku_all)  # 因为之前的dataFrame表 没有设置index 现在设置属性表
            Model3.Self_attributing_features(count, params_all, sku_feature_all, sku_all, start_year, end_year)


    rolling()


    EM.get_Each_sku_Mape_all_new("/地市",rolling_start,rolling_end)
    EM.get_train_set_Mape_feature_mapping_roll_year("/地市", 2018, 2022, "老品")








