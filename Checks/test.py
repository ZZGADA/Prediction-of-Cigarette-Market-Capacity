import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.svm import SVR as svr
from sklearn.metrics import get_scorer_names as scoreName
from sklearn import preprocessing as prep
sys.path.append(r"../")  #返回上级目录
import Config
from Dao import static_file as sf
from Dao import Data_Acquisition as DA
from Service import Bass_Model as BM
from Service import Model1_province_sku as Model1
from Service import Model2_sample_province as Model2
from Service import Model3_region as Model3
from Service import Characteristic_Mapping as CM
from DataAnalysis import Projected_evaluation_indicators as pei
from Dao import Data_Processing as Dpcs
from DataAnalysis import Each_Machine_Model_MAPE as EM
import os
from math import e
from sklearn.model_selection import GridSearchCV
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')



#获取原始数据的测试函数
def get_Data_test():
    path="../源文件/销量统计总表.xlsx"
    Data=DA.get_Data(path)
    # print(Data)
def get_feature_province():
    path="../源文件/property_feature_all.csv"
    Data=DA.get_feature_province(path,index_col=0).iloc[:,:13]
#巴斯模型中n(t)的二阶导数 类似于一个sin的式子
def get_first_order_derivative_of_Bass():
    m,p,q,t=sp.symbols('m p q t')  #  求导test
    f_new=sp.diff(BM.func(t,m,p,q),t,1)
    f_new2= sp.diff(BM.func(t, m, p, q), t, 2)
    nums=20
    nums1=100
    T=np.linspace(1,25,nums)
    T1=np.linspace(1,25,nums1)
    Y_hat=[f_new.evalf(subs={t:i,m:173280.0695,p:0.003460318,q:0.604036119})for i in T1 ]
    y_hat=BM.func(T,173280.0695,0.003460318,0.604036119)
    y_means=[ (y_hat[i+1]-y_hat[i-1])/(T[i+1]-T[i-1]) for i in range(1,nums-1) ]
    y_hat2=[f_new2.evalf(subs={t:i,m:173280.0695,p:0.003460318,q:0.604036119})for i in T1 ]
    fig=plt.figure()
    plt.plot(T1[1:nums1-1],Y_hat[1:nums1-1],c='r')
    plt.plot(T[1:nums-1],y_means,c='b')
    plt.plot(T1[1:nums1-1],y_hat2[1:nums1-1],c='g')

    # plt.plot(T[1:nums-1],y_hat[1:nums-1],c='g')
    plt.show()
    '''
    发现规律记录以及解决方案：
    n(t)一阶导函数求得的两个斜率最大值t1 and t2 还有斜率为0的 t*外 在t1左侧和t2右侧存在 t11和t22两个
    均值斜率和实际一阶导函数中的数值（真实的斜率）是相等的 
    所以根据n(t)的二阶导的结果可以将n(t)划分为四个时段    具体样式见记录 
    根据斜率变化的情况趋势图 斜率的变化程度可以近似为多条直线
    所以第一步数据清洗
    方案1：将有问题的数据映射到 正常的数据(tk,tn,k>n)的 k-n+1 位点上面
    '''

#k-n+1位点做修订 异常值处理
def data_PreProcessing_test():
    path="../源文件/销量统计总表.xlsx"
    Data=DA.get_Data(path)
    return_path=Dpcs.data_PreProcessing(Data)
    return return_path

def rolling_forecast():
    start_year=2006
    Data:pd.DataFrame
    # print("it is rolling forecast")

    count=1
    for end_year in range(2018,2019):
        return_path = data_PreProcessing_test()
        # return_path = "../源文件/清洗后的全sku数据.csv"
        Data = DA.get_Data(return_path)
        sku_all=Model1.sku_classification(Data,start_year,end_year)
        params_all=Model1.get_Bass_params(sku_all,start_year,end_year)
        attributes_path = "../源文件/property_feature_all.csv"
        self_attribute = DA.get_feature_province(attributes_path,index_col=0).iloc[:, :13]
        Model1.sku_sales_index_processing(sku_all)
        sku_feature_all,params_all=CM.get_key_feature(self_attribute,params_all)
        Model1.Self_attributing_features(count,params_all,sku_feature_all,sku_all,start_year,end_year)
        count+=1

def get_key_feature_test(feature,params):
    # attributes_path = "D:\非常有用的大学资料\中国烟草\测试数据\property_feature_all.csv"
    # feature = DA.get_feature_province(attributes_path).iloc[:, :14]   #

    sku_feature_all,params_all=CM.get_key_feature(feature,params)

def question_test():
    a=pd.DataFrame(data={'a':[2020,2021]},index=['q','p'])
    b=pd.DataFrame(data={'b':[2021,2022]},index=['q','u'])
    # c=pd.concat([a,b],axis=1) 合并
    # c=pd.concat([a,b],axis=1,join="inner")   取交集

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True)
    a = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    b = [1, 2, 3, 4, 5,6,7,8,9,10]
    for i, j in kf.split(a, b):
        print(i,j)
def svr_test():
    # 构造数据集
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel()

    # 加入噪声
    y[::5] += 3 * (0.5 - np.random.rand(16))

    # 训练模型
    model = svr(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X, y)

    # 在图表上展示结果
    plt.scatter(X, y, label='data', color='red')
    plt.plot(X, model.predict(X), label='SVR', color='blue')
    plt.legend()
    plt.show()
def estimator():
    # 定义多项式函数
    def polynomial_func(x, a, b, c):
        return a * x ** 2 + b * x + c

    # 定义参数搜索范围
    param_grid = {'a': np.linspace(-1, 1, 10),
                  'b': np.linspace(-1, 1, 10),
                  'c': np.linspace(-1, 1, 10)}

    # 定义评估函数
    def evaluate_func(x, y, func, **params):
        popt, _ = curve_fit(func, x, y, **params)
        y_pred = func(x, *popt)
        return mean_squared_error(y, y_pred)

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(estimator=evaluate_func, param_grid=param_grid, cv=5)

    # 进行参数搜索
    grid_search.fit(x, y)

    # 输出最佳参数和最佳得分
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
def rolling_forecast_sample():
    start_year=2006
    Data:pd.DataFrame

    # return_path= "../源文件/清洗后的全sku数据.csv"
    def rolling():
        for end_year in range(2017,2022):    #2017-2022
            count=end_year-2016
            path = "../源文件/销量统计总表.xlsx"
            Data = DA.get_Data(path)
            path_clean="../源文件/销量预测误差统计_test/第%d轮滚动结果/相关数据"
            return_path = Dpcs.data_PreProcessing(Data,start_year,end_year,count,path_clean)
            Data = DA.get_Data(return_path)
            attributes_path = "../源文件/property_feature_all.csv"
            self_attribute = DA.get_feature_province(attributes_path, index_col=0).iloc[:, :13]  # 提取特征属性表
            sku_feature_all, sku_all = CM.Discrete_data_processing(self_attribute,
                                                                   Data)  # 获得添加虚拟变量后的特征表 ,过滤掉没有特征的sku 第一遍筛选数据

            sku_all=Model2.sku_classification(sku_all,start_year,end_year)  #对sku进行 分类
            params_all=Model2.get_Bass_params(sku_all,start_year,end_year)  #获得分类后的 bass参数
            Model2.sku_sales_index_processing(sku_all)  #因为之前的dataFrame表 没有设置index 现在设置属性表
            Model2.Self_attributing_features(count,params_all,sku_feature_all,sku_all,start_year,end_year)
    # rolling()
    # EM.get_Each_Machine_Model_MAPE_all_new_feature_mapping("")
    # EM.get_Each_Machine_Model_MAPE_semi_feature_mapping("")
    # EM.get_Each_Machine_Model_MAPE_semi_curve_fit("")
    # EM.get_Each_Machine_Model_MAPE_old_curve_fit("")
    EM.get_Each_sku_Mape_all_new("")

def rolling_forecast_region():
    start_year = 2006
    Data: pd.DataFrame

    # return_path= "../源文件/清洗后的全sku数据.csv"
    rolling_start=2017
    rolling_end=2022

    def rolling():
        for end_year in range(rolling_start, rolling_end):  # 2017-2022
            count = end_year - 2016
            print("这是第%d轮滚动"%count)
            path = "../源文件/地市/特征结果"
            Data = DA.get_region_data(path)
            # feature_supplementary=DA.get_region_supplementary_feature(path,end_year)

            path_clean="../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/相关数据"
            return_path = Dpcs.data_PreProcessing(Data, start_year, end_year, count,path_clean)
            Data = DA.get_Data(return_path)
            attributes_path = "../源文件/property_feature_all.csv"
            self_attribute = DA.get_feature_region(attributes_path, index_col=0).iloc[:, :13]  # 提取特征属性表
            sku_feature_all, sku_all = CM.Discrete_data_processing_region(self_attribute,
                                                                   Data,count,end_year)  # 获得添加虚拟变量后的特征表 ,过滤掉没有特征的sku 第一遍筛选数据

            # sku_feature_all=CM.supplementary_Feature_Matching(sku_feature_all,feature_supplementary)
            sku_all = Model3.sku_classification(sku_all, start_year, end_year)  # 对sku进行 分类
            params_all = Model3.get_Bass_params(sku_all, start_year, end_year)  # 获得分类后的 bass参数
            Model3.sku_sales_index_processing(sku_all)  # 因为之前的dataFrame表 没有设置index 现在设置属性表
            Model3.Self_attributing_features(count, params_all, sku_feature_all, sku_all, start_year, end_year)

    rolling()
    # EM.get_Each_Machine_Model_MAPE_all_new_feature_mapping("/地市")
    # EM.get_Each_Machine_Model_MAPE_semi_feature_mapping("/地市")
    # EM.get_Each_Machine_Model_MAPE_semi_curve_fit("/地市")
    # EM.get_Each_Machine_Model_MAPE_old_curve_fit("/地市")


    EM.get_Each_sku_Mape_all_new("/地市",rolling_start,rolling_end)
    # EM.get_train_set_Mape_feature_mapping_roll_year("/地市",rolling_start+1,rolling_end,"老品")
    # EM.get_train_set_Mape_feature_mapping_sku("/地市",2018,2022,"老品")

    EM.get_train_set_Error_feature_mapping_plus_curve_fit(rolling_start,rolling_end)
    EM.get_test_set_Error_feature_mapping_plus_curve_fit(rolling_start,rolling_end)





def test_get_sku(Data:pd.DataFrame):
    sku_all_region=Data.columns.to_list()[1:]
    sku_all_region_only_name=[i.split("_")[1] for i in sku_all_region]
    i:str
    sku_all_region_only_name=[i for i in sku_all_region_only_name if i.__contains__("利群")]
    set_final=list(set(sku_all_region_only_name))
    print(len(set_final))
    set_final.append("aaaa")
    t1=np.array(set_final,ndmin=2)
    a=t1.reshape(-1,4)
    aa=pd.DataFrame(a)
    aa.to_csv("../源文件/all_sku_name.csv")

def find_problem():
    path_1="..\源文件\地市\测试数据结果\第4轮\data_sku_old.csv"
    path_2="..\源文件\地市\测试数据结果\第5轮\data_sku_old.csv"
    data_1=pd.read_csv(path_1,index_col=0)
    data_2=pd.read_csv(path_2,index_col=0)
    columns_1=data_1.columns
    columns_2=data_2.columns
    res=[]
    for i in columns_1:
        if i not in columns_2:
            res.append(i)
    print(res)
    print(len(res))

def find_problem2():
    path1="D:\python\数据分析\巴斯模型新品预测\源文件\地市\销量预测误差统计_test\各个SKU的预测销量统计\新品\新品预测销量统计(XG).csv"
    path2="D:\python\数据分析\巴斯模型新品预测\源文件\地市\销量预测误差统计_test\各个SKU的预测销量统计\新品\新品测试集误差统计\XG\误差集合\XG_测试集每年各SKU_MAPE误差_feature_mapping.csv"

    data_test=pd.read_csv(path2,index_col=0)
    data_all_new=pd.read_csv(path1,index_col=0)
    data_all_new_columns=data_all_new.columns.to_list()
    brand_new_pass=[]

    for sku_name in data_all_new.columns:
        if sku_name not in data_test.index:
            brand_new_pass.extend(sku_name)
            data_all_new_columns.remove(sku_name)

    for sku_name in data_test.index:
        if sku_name not in data_all_new_columns:
            try:
                temp=pd.read_csv("../源文件/地市/特征结果/%s.csv"%(sku_name),index_col=0)["sale"].rename(sku_name)
                print(temp)
            except:
                pass

if __name__=="__main__":
    print("---test---")
    # rolling_forecast_sample()
    # find_problem2()

    rolling_forecast_region()










