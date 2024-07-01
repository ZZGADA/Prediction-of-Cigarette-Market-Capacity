#省份+sku
import math
import os
import random

import pylab as pl
from scipy.optimize import leastsq, curve_fit
import sympy as sp
import numpy as np  # 科学计算库
import matplotlib.pyplot as plt  # 绘图库
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from math import e
from numba import cuda,jit

from Dao import static_file as sf
from sklearn.model_selection import train_test_split  #训练集和测试集分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso as LS
from sklearn.linear_model import LassoCV as LScv
from sklearn.linear_model import ElasticNet as EN
from sklearn.linear_model import ElasticNetCV as ENcv
from sklearn.ensemble import RandomForestRegressor as Rfr
from sklearn.svm import SVR as svr
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn import preprocessing as prep
from deap import base, creator, tools, algorithms
import sys
sys.path.append(r"../")  #返回上级目录
from Service import Bass_Model as BM
from DataAnalysis import Projected_evaluation_indicators as pei
from copy import deepcopy
from sklearn import svm
from pylab import mpl
from Service import Characteristic_Mapping as CM
from Dao import Data_Acquisition as DA
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

now_year=2023
Bass_params = sf.static_class.Bass_params
Errors_columns_name=sf.static_class.Errors_columns_name
Machine_Model_name=sf.static_class.Machine_Model_name
train_test_dict_name=sf.static_class.train_test_dict_name
train_test_keys_name=sf.static_class.train_test_keys_name
Model_return_contend=sf.static_class.Model_return_contend
#获取Bass模型的三个参数
#Data是每一个品规的数据
#start_year,end_year 是起使时间和终止时间


def sku_classification(Data:pd.DataFrame,start_year:int,end_year:int):
    last_year=Data.shape[0]+start_year-1
    start_year_index = Data["year"][Data['year'] == start_year].index.to_list()[0]
    end_year_index = Data["year"][Data['year'] == end_year].index.to_list()[0]
    end_year_index_plus1=end_year_index+1
    data_sku_old=pd.DataFrame(data={"year":[i for i in range(start_year,last_year+1)]})  #要使用的老品规数据
    sku_brand_new=pd.DataFrame(data={"year":[i for i in range(end_year+1,last_year+1)]})  #下一年的完全新品数据
    semi_flag_year_index = end_year_index - 4  # 在销售新品的标志下标
    sku_semi_brand_new=pd.DataFrame(data={"year":[i for i in  range(end_year-4,last_year+1)]})  #非完全新品数据 在end年之前有连续的小于等于5年的销量
    Sales_path_head = "../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/相关数据/" % (end_year-2016)

    for sku in Data.columns.to_list()[1:]:

        data_temp:pd.Series
        data_sku=Data[sku][start_year_index:end_year_index_plus1]
        data_temp=np.trim_zeros(data_sku)

        if data_temp.size==0 :#如果首尾去0 之后没有数据则表明该sku的销量 在当前阶段没有意义和价值
            if Data[sku][end_year_index_plus1]!=0:  #下一年有数据 则表示该sku为下一年的完全新品
                sku_brand_new[sku]=Data[sku][end_year_index_plus1:].to_list()
            continue
        elif data_temp.index.to_list()[0]>semi_flag_year_index:  #则表示为在销新品数据  !!!
        # elif np.trim_zeros(data_sku[semi_flag_year_index:]).size()>1 or data_sku[end_year_index_plus1]>0:
            sku_semi_brand_new[sku]=Data[sku][semi_flag_year_index:].to_list()
            continue
        else :
            #细切割
            data_temp1:pd.Series
            data_temp2:pd.Series

            data_temp1=np.trim_zeros(data_temp.iloc[:semi_flag_year_index]).size
            data_temp2=np.trim_zeros(data_temp.iloc[semi_flag_year_index:end_year_index_plus1]).size
            data_temp_all=data_temp2+data_temp1
            #四分位点数据
            four_FenWeiDian=int((end_year-start_year+1)/4)
            if data_temp_all>4:
                data_sku_old[sku] = Data[sku].to_list()
            elif Data[sku][end_year_index_plus1]==0 and data_temp_all<3 :
                #小于巴斯模型参数个数 而且不属于完全新品 则直接跳过 该sku 没有价值
                #TODO:  有些在销售的新品就被Pass掉了
                continue
            elif data_temp1>data_temp2:  #在在销新品的划分位置比对 之前的销量数据大于在销新品的销量数据
                if data_temp1 <3:
                    if data_temp2==0 and Data[sku][end_year_index_plus1]!=0:
                        sku_brand_new[sku] = Data[sku][end_year_index_plus1:].to_list()
                        #TODO 这部分的品规确认为新品 需要检查
                        continue
                    sku_semi_brand_new[sku] = Data[sku][semi_flag_year_index:].to_list()
                    continue
                data_sku_old[sku] = Data[sku].to_list()
            else:
                sku_semi_brand_new[sku] = Data[sku][semi_flag_year_index:].to_list()

    # print(sku_brand_new)
    sku_all={"data_sku_old":data_sku_old,"sku_brand_new":sku_brand_new,"sku_semi_brand_new":sku_semi_brand_new}
    for i, j in sku_all.items():
        # print(i)
        # print(j)
        j.to_csv(Sales_path_head+i+"原始销量.csv")
    return sku_all
# data_sku_old:pd.DataFrame,sku_brand_new:pd.DataFrame,sku_semi_brand_new:pd.DataFrame
#每sku_all中每一个dataFrame 的index 都是从0开始的
def get_Bass_params(sku_all:dict,start_year:int,end_year:int):
    # print("it is getting Bass params")
    params_old=pd.DataFrame(columns=Bass_params)
    params_brand_new=pd.DataFrame(columns=Bass_params)
    params_semi_brand_new=pd.DataFrame(columns=Bass_params)
    params_all={"data_sku_old":params_old,"sku_brand_new":params_brand_new,"sku_semi_brand_new":params_semi_brand_new}
    Sales_path_head_old = "../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/相关数据/Bass_curve_fit图片/老品" % (end_year - 2016)
    Sales_path_head_new = "../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/相关数据/Bass_curve_fit图片/新品" % (end_year - 2016)
    os.makedirs(Sales_path_head_old,exist_ok=True)
    os.makedirs(Sales_path_head_new, exist_ok=True)

    sku_data:pd.DataFrame
    temp:pd.Series
    for sku_classification_name,sku_data in sku_all.items():  #遍历字典中的pd.DataFrame
        # print(sku_classification_name)
        # if sku_classification_name=="sku_brand_new":
        #     continue
        # t=np.subtract(sku_data["year"].to_list(),start_year-1.0)
        try:
            end_year_index=sku_data["year"][sku_data["year"]==end_year].index.to_list()[0]
            # semi_flag_year_index = end_year_index - 4
        except:
            pass

        for sku in sku_data.columns.to_list()[1:]:  #遍历每一个DataFrame的数据 但是不包括第一列的年份
            temp=np.trim_zeros(sku_data[sku][:end_year_index+1])  #获取连续时段内有销量的数据  首尾去0
            if temp.size<3 : #or temp.size<round((end_year-start_year)/4)
                # 如果样本数量小于3 表示该样本没有价值 无法进行拟合
                continue
            if sku_classification_name=="sku_brand_new":
                temp=np.trim_zeros(sku_data[sku])
            temp=temp[temp>0]

            year_get=temp.index.to_list()
            t=np.subtract(year_get,year_get[0]-1)  #获取时间序列


            try:
                #TODO:curve_fit的界定范围
                params_bounds=None
                if sku_classification_name=="data_sku_old":
                    params_bounds=([temp.sum(),0,0],[temp.sum()*5,1,1])
                else:
                    params_bounds = ([temp.sum()*1.2, 0, 0], [temp.sum() * 5, 1, 1])
                #p0=[temp.sum(),10,10]
                params,pcov=curve_fit(BM.func,t,temp,bounds=params_bounds,maxfev=1000) #样本数必须大于参数的个数 所以如果样本数 小于3 则该样本没有意义


                '''pcov 返回参数的协方差矩阵 
                参数的协方差矩阵（pcov）是一个二维数组，表示参数的不确定性。
                对角线上的元素是每个参数的方差，非对角线上的元素是参数之间的协方差。
                协方差矩阵可以用来估计参数的置信区间和相关性。
                '''
                '''
                
                reg=svr()
                kernel = ['poly', 'sigmoid','linear',  'rbf']
                search_space = {
                    'kernel': kernel,
                    'C': [1, 10, 100, 1000]
                }
                search=GridSearchCV(reg,search_space,cv=3,n_jobs=-1,scoring='r2')
                t=t.reshape(-1, 1)
                search.fit(t,temp)
                y_svr_hat=search.predict(t)
                t = t.reshape(1,-1)[0]'''
                params_all[sku_classification_name].loc[sku] = params

                y_hat=BM.func(t,*params)
                # print(sku)
                # print(y_hat)
                figure=plt.figure()
                plt.plot(t,y_hat,'ro-',label='bass拟合_curvefit',linewidth=1)
                plt.plot(t,temp,'bo-',label='原始值',linewidth=0.8)
                plt.title(sku)
                # a=[str(i) for i in np.add(t+2006,year_get[0]-1)]
                a=sku_data['year'][year_get]
                plt.xticks(t,a)
                plt.legend(loc='upper right')
                # print("sales_max_t:",BM.get_sales_max_t(params[1],params[2]))
                if sku_classification_name=="data_sku_old":
                    plt.savefig(Sales_path_head_old+"/%s"%sku)
                else:

                    plt.savefig(Sales_path_head_new + "/%s" % sku)
                # plt.show()

            except RuntimeError as RE:
                # 如果拟合超时那么该品规的数据也没有价值和意义  表示数据有问题
                # #雄狮(红老板) 被踢掉了 old_sku  重新调整一下

                continue


    return params_all #返回巴斯参数集合


# params_all={"data_sku_old":params_old,"sku_brand_new":params_brand_new,"sku_semi_brand_new":params_semi_brand_new}
# sku_all={"data_sku_old":data_sku_old,"sku_brand_new":sku_brand_new,"sku_semi_brand_new":sku_semi_brand_new}
#data_sku_old  sku_brand_new  sku_semi_brand_new

# 这里必须按照params_all 中每一个dataFrame的index来 因为在bass参数拟合时对异常值和无价值数据进行了筛选

# 自身属性部分和Bass参数的映射
# 有多个机器学习模型  回归类的
def Self_attributing_features(roll_num,params_all:dict,self_attributes_all,sku_sales_data_all:dict,start_year,end_year):
    # print("it is Self-attributing features")
    #end_year 为滚动中时间范围的终止点 end_year+1为新品预测的起始年份   end_year==2022 那么就完全没有新的销量数据了

    train_old:dict
    test_semi:dict
    test_brand:dict
    scaler: prep.StandardScaler
    x_train_f :pd.DataFrame  #老品的特征 训练集
    y_train_p :pd.DataFrame #老品的Bass参数 训练集
    x_train_t:pd.DataFrame #老品的生命周期记录 训练集
    y_train_s :pd.DataFrame #老品的销量 训练集
    x_test_f :pd.DataFrame #新品的特征 测试集
    y_test_s :pd.DataFrame #新品的销量 测试集
    x_test_t :pd.DataFrame #新品的生命周期 测试集
    y_test_p :pd.DataFrame #新品的Bass参数拟合值 测试集 每一个模型都不同
    y_train_p_hat :pd.DataFrame #老品在模型中拟合出来的新的Bass参数
    predict_year=end_year+1
    Model_return_contend_chines = ["老品_历史_销量误差(特征映射)", "新品_历史_销量误差(特征映射)",
                                   "老品_预测年份(%d)_销量误差(特征映射)"%(predict_year),
                                   "新品_预测年份(%d)_销量误差(特征映射)"%(predict_year),
                                   "老品_历史_销量误差(curve_fit)",
                                   "新品_历史_销量误差(curve_fit)",
                                   "老品_预测年份(%d)_销量误差(curve_fit)"%(predict_year),
                                   "新品_预测年份(%d)_销量误差(curve_fit)"%(predict_year),
                                   "新品_有销量的预测年份(%d)_销量误差(特征映射)"%(predict_year)]
    Model_return_contend_chines_dict=dict(zip(Model_return_contend,Model_return_contend_chines))
    machine_model = dict(zip(Machine_Model_name, [{} for i in range(len(Machine_Model_name))]))
    Sales_path_head = "../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/" % (roll_num)
    Bass_path_head = "../源文件/地市/Bass参数特征拟合结果_test/第%d轮滚动结果/" % (roll_num)

    train_test_data,scaler=get_train_test(params_all,self_attributes_all,sku_sales_data_all,start_year,end_year,roll_num)
    train_old,test_semi,test_brand=train_test_data.values()
    brand_new_sku=test_brand[train_test_keys_name[1]].columns.to_list()
    semi_brand_new_sku=test_semi[train_test_keys_name[1]].columns.to_list()
    new_sku=brand_new_sku+semi_brand_new_sku

    x_train_f = train_old[train_test_keys_name[0]]  # feature
    y_train_p = train_old[train_test_keys_name[3]]  # params
    x_train_t = train_old[train_test_keys_name[2]]  # t
    y_train_s = train_old[train_test_keys_name[1]]  # sales
    # print(y_train_s)

    x_test_f = pd.concat([test_semi[train_test_keys_name[0]], test_brand[train_test_keys_name[0]]], axis=0)  # feature
    y_test_p = test_semi[train_test_keys_name[3]] #params
    y_test_s = pd.concat([test_semi[train_test_keys_name[1]], test_brand[train_test_keys_name[1]]], axis=1)  # sales
    x_test_t = pd.concat([test_semi[train_test_keys_name[2]], test_brand[train_test_keys_name[2]]], axis=1)  # t

    semi_can_use_index=y_test_p.index

    # x_train_f=pd.concat([x_train_f,x_test_f.loc[semi_can_use_index]],axis=0)
    # y_train_p =pd.concat([y_train_p,y_test_p],axis=0)
    # x_train_t=pd.concat([x_train_t,x_test_t.loc[:,semi_can_use_index]],axis=1)
    # y_train_s=pd.concat([y_train_s,y_test_s.loc[:,semi_can_use_index]],axis=1).fillna(0)

    x_train_f.to_csv(Sales_path_head+"/相关数据/特征_训练集老品.csv")
    x_test_f.to_csv(Sales_path_head+"/相关数据/特征_训练集新品.csv")

    # print(x_test_f)
    # input("去检查特征表")


    def machine_training(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p,model_name,model,search_space=None,ifInverse=False):
        print("it is "+model_name)
        # 返回Bass 的三个参数 m,p,q  做记录


        x_train_f_new=x_train_f
        x_test_f_new=x_test_f
        if ifInverse:
            x_train_f_new = feature_inverse_transform(deepcopy(x_train_f), scaler)
            x_test_f_new = feature_inverse_transform(deepcopy(x_test_f), scaler)
  
        y_test_p_hat = pd.DataFrame(index=x_test_f.index, columns=Bass_params)  # param  新品的拟合Bass参数的记录DataFrame
        y_train_p_hat = pd.DataFrame(index=x_train_f.index, columns=Bass_params)  # 老品的特征值和属性值的映射
        bass_error = pd.DataFrame(index=Bass_params, columns=Errors_columns_name)  # 记录老品的三个bass参数误差
        bass_error_semi = pd.DataFrame(index=Bass_params, columns=Errors_columns_name)
        semi_brand_have_params = y_test_p.index.to_list()
        # 模型训练
        def func(x_train, x_test, y_train, param):
            if search_space is None:
                search=model[param]
            else:
                search=GridSearchCV(model[param],search_space,cv=4,n_jobs=-1)
            search.fit(x_train, y_train)
            y_hat: pd.Series
            y_test = search.predict(x_test)  # 获得新品的Bass参数
            y_train_hat = search.predict(x_train)  # 获得老品的新的Bass参数
            # print(search.best_params_)
            # y_test=pd.DataFrame(data=np.ones(x_test.shape[0]),index=x_test.index,columns=[param])
            # print(y_test.loc[semi_brand_have_params])
            # y_train_hat=np.ones(x_train.shape[0])

            y_train_p_hat[param] = y_train_hat
            y_test_p_hat[param] = y_test
            bass_error.loc[param] = get_all_Error_indicator_new(y_train, y_train_hat)
            bass_error_semi.loc[param] = get_all_Error_indicator_new(y_test_p[param],
                                                                     y_test_p_hat.loc[semi_brand_have_params, param])
            print(bass_error)

        for param in Bass_params:
            func(x_train_f_new, x_test_f_new, y_train_p[param], param)
            # 老品的误差
        bass_error_head_temp = Bass_path_head + model_name
        os.makedirs(bass_error_head_temp, exist_ok=True)


        #获取各个bass误差和bass参数结果
        bass_error.to_csv(bass_error_head_temp + "/训练集(老品)Bass参数误差结果.csv")
        y_train_p_hat.to_csv(bass_error_head_temp + "/训练集(老品)Bass参数特征映射拟合结果.csv")
        y_train_p.to_csv(bass_error_head_temp + "/训练集(老品)Bass参数原始拟合结果.csv")
        y_test_p_hat.to_csv(bass_error_head_temp + "/测试集(新品)Bass参数特征映射拟合结果.csv")
        y_test_p.to_csv(bass_error_head_temp + "/测试集(在销新品)Bass参数原始拟合结果.csv")
        bass_error_semi.to_csv(bass_error_head_temp + "/测试集(在销新品)Bass参数误差结果.csv")
        # 在销新品中curve_fit出来的bass参数误差

        # 老品的特征映射结果
        sales_old_predict = sales_predict_func(x_train_t, y_train_p_hat)
        # 新品的特征映射结果
        sales_new_predict = sales_predict_func(x_test_t, y_test_p_hat)
        # 在销新品中 有curve_fit巴斯参数的品规的时间
        x_test_t_semi_has_params = x_test_t[semi_brand_have_params]

        # 原始Bass_curve_fit的拟合结果的误差 以及拟合销量
        sales_old_init_bass = sales_predict_func(x_train_t, y_train_p)
        sales_semi_init_bass = sales_predict_func(x_test_t_semi_has_params, y_test_p)
        sales_Bass_init = pd.concat([sales_old_init_bass, sales_semi_init_bass], axis=1).fillna(0)

        sales_old_init_bass.to_csv(Sales_path_head + "/%s/Bass函数(curve_fit)拟合销量结果(老品).csv" % model_name)
        sales_semi_init_bass.to_csv(Sales_path_head + "/%s/Bass函数(curve_fit)拟合销量结果(在销新品).csv" % model_name)

        # 画图所用的数据 分为历史销量数据 预测年的预测数据 全生命周期的年份
        sales_past = pd.concat([y_train_s, y_test_s], axis=1).fillna(0)
        sales_predict = pd.concat([sales_old_predict, sales_new_predict], axis=1).fillna(0)

        # 获得每个部分的销量误差 依次为["老品_历史_销量误差(特征映射)", "新品_历史_销量误差(特征映射)", "老品_预测年份(%d)_销量误差(特征映射)"%(predict_year),"新品_预测年份(%d)_销量误差(特征映射)"%(predict_year),"老品_历史_销量误差(curve_fit)", "新品_历史_销量误差(curve_fit)", "老品_预测年份(%d)_销量误差(curve_fit)"%(predict_year),"新品_预测年份(%d)_销量误差(curve_fit)"%(predict_year)]
        sales_old_past_error = get_sale_error(y_train_s, sales_old_predict,predict_year)
        sales_semi_past_error = get_sale_error(y_test_s[semi_brand_new_sku], sales_new_predict[semi_brand_new_sku]
                                               ,predict_year)
        sales_predict_error_old = get_sale_error(y_train_s.loc[predict_year], sales_old_predict.loc[predict_year],
                                                 predict_year)
        sales_predict_error_new = get_sale_error(y_test_s.loc[predict_year], sales_new_predict.loc[predict_year],
                                                 predict_year)

        sales_old_past_error_curve_fit = get_sale_error(y_train_s, sales_old_init_bass,predict_year)
        sales_semi_past_error_curve_fit = get_sale_error(y_test_s[semi_brand_have_params], sales_semi_init_bass,predict_year)
        sales_predict_error_curve_fit = get_sale_error(y_train_s.loc[predict_year],
                                                       sales_old_init_bass.loc[predict_year], predict_year)
        sales_predict_error_new_curve_fit = get_sale_error(y_test_s.loc[predict_year, semi_brand_have_params],
                                                           sales_semi_init_bass.loc[predict_year], predict_year)
        sales_predict_error_semi=get_sale_error(y_test_s.loc[predict_year,semi_brand_new_sku],sales_new_predict.loc[predict_year,semi_brand_new_sku],predict_year)
        res = [sales_old_past_error, sales_semi_past_error, sales_predict_error_old, sales_predict_error_new,
               sales_old_past_error_curve_fit, sales_semi_past_error_curve_fit, sales_predict_error_curve_fit,
               sales_predict_error_new_curve_fit,sales_predict_error_semi]
        RES = dict(zip(Model_return_contend, res))  # 将数据打包合并
        machine_model[model_name] = RES
         #TODO  sales_plot 里面老品的年份有问题记得修改
        sales_plot(sales_past, sales_predict, new_sku, roll_num, model_name, sales_Bass_init, end_year)


    def Lasso(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p):
        model_name="Lasso"

        search_space={
            'alpha':(1e-100, 1e-3, 'log-uniform'),
            'fit_intercept':(True,False),
            'positive':(True,False)
        }
        model = {'m': LS(), 'p': LS(), 'q': LS()}
        machine_training(x_train_f, y_train_p, x_train_t, y_train_s, x_test_f, y_test_s, x_test_t, y_test_p, model_name,
                         model, search_space)

    # Lasso(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p)

    def ElasticNet(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p):
        model_name="ElasticNet"
        L1Range = np.logspace(1e0 - 50, 0, 100, base=10)
        search_space = {
            'alpha': (1e-100, 1e-3, 'log-uniform'),
            'l1_ratio': L1Range,
            'fit_intercept': (True, False),
            'positive': (True, False)
        }
        model = {'m': EN(), 'p': EN(), 'q': EN()}
        # search=BayesSearchCV(reg,search_space, n_iter=50, cv=5, n_jobs=-1)
        machine_training(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p,model_name,model,search_space)
    # ElasticNet(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p)


    def SVR(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p):#支持向量回归模型
        model_name="SVR"
        model={'m':svr(),'p':svr(),'q':svr()}
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        search_space = {
            'kernel': kernel,
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        machine_training(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p,model_name,model,search_space)

    SVR(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p)

    def RT(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p):#支持向量回归模型
        model_name="RT"
        model={'m':Rfr(),'p':Rfr(),'q':Rfr()}
        n_estimators = [i for i in range(100, 351, 10)]
        search_space = {
            'n_estimators': n_estimators,
            'max_depth': [i for i in range(3, 7)],
            'bootstrap': (True, False),
            'min_samples_split': [1, 2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3]
        }
        machine_training(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p,model_name,model,search_space,ifInverse=True)

    RT(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p)

    def KNN(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p):#支持向量回归模型
        model_name="KNN"
        model = {'m': knn(), 'p': knn(), 'q': knn()}

        n_neighbors=[k for k in range(1,8)]
        weight=['uniform','distance']
        search_space={
            'n_neighbors':n_neighbors,
            'weights':weight,
            'p':[p for p in range(1,8)]
        }

        machine_training(x_train_f, y_train_p, x_train_t, y_train_s, x_test_f, y_test_s, x_test_t, y_test_p, model_name,
                         model, search_space)
    KNN(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p)

    def XG(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p):#支持向量回归模型
        model_name="XG"
        n_estimators = [i for i in range(100, 501, 5)]
        xrange=np.logspace(1e-10,0,20)
        search_space={
            'n_estimators':n_estimators,
            'learning_rate' : [0.01,0.05,0.1],
            'max_depth':[6,7,8,9,10],
            'verbosity':[0]
        }
        #max_depth=6, learning_rate=0.05, n_estimators=500, verbosity=0
        model = {'m': xgb.XGBRegressor(), 'p': xgb.XGBRegressor(), 'q': xgb.XGBRegressor()}
        machine_training(x_train_f, y_train_p, x_train_t, y_train_s, x_test_f, y_test_s, x_test_t, y_test_p, model_name,
                         model,search_space ,ifInverse=True)
        # search=GridSearchCV(reg,search_space,cv=5,n_jobs=-1,error_score='raise')

    XG(x_train_f,y_train_p,x_train_t,y_train_s,x_test_f ,y_test_s ,x_test_t ,y_test_p)



    #最终打印输出的位置
    def file_save():
        Sales_path_head = "../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/" % (roll_num)
        # Bass_params_path_head="../源文件/Bass参数特征拟合结果_test/第%d轮滚动结果/"% (roll_num)
        for model_name in Machine_Model_name:

            sales_error_head_temp=Sales_path_head+model_name
            os.makedirs(sales_error_head_temp,exist_ok=True)
            Sales_path_head_final=sales_error_head_temp+"/%s.csv"
            for res_name,res in machine_model[model_name].items():
                res.to_csv(Sales_path_head_final % (Model_return_contend_chines_dict[res_name]),index=True)
        print("----结束----")
    file_save()

def cold_start_mechanism(Feature,Data):
    print("it is cold-start mechanism")





def get_all_Error_indicator_new(y_test,y_hat)->pd.Series:
    Accuracy=None
    # if y_test.size==1:
    #     Accuracy= 1 if y_test==y_hat else 0
    # else :
    # Accuracy = accuracy_score(y_test.astype("int64"), y_hat.astype("int64"))
    # R2=r2_score(y_test, y_hat)
    MSE=pei.get_MSE(y_test, y_hat)
    MAE=pei.get_MAE(y_test, y_hat)
    RMSE=pei.get_RMSE(y_test, y_hat)
    MAPE=mean_absolute_percentage_error(y_test,y_hat)
    # MAPE=pei.get_MAPE(y_test,y_hat)
    SMAPE=pei.get_SMAPE(y_test,y_hat)
    indicator_list=np.array([MSE, MAE, RMSE,MAPE,SMAPE])
    res=pd.Series(data=indicator_list,index=Errors_columns_name)
    return res


#特征缩放  特征缩放需要区分训练集和测试集
#一个样本一行数据
def feature_scaling(train,test=None):
    train_scale: np.ndarray
    test_scale: np.ndarray
    scaler = prep.StandardScaler()

    if train.__class__==pd.DataFrame:
        columns=train.columns
        index=train.index
        train_scale = scaler.fit_transform(train)  # 纵轴方向数据拟合
        res=pd.DataFrame(data=train_scale,columns=columns,index=index)
        if test is None:
            return res,scaler
        else :
            test_columns=test.columns
            test_index=test.columns
            test_scale = scaler.transform(test)  # 缩放器已经拟合不能再 fit
            res_test=pd.DataFrame(data=test_scale,columns=test_columns,index=test_index)
            return res,res_test,scaler

#获取销量预测数据
def sales_predict_func(T:pd.DataFrame,Bass_sku:pd.DataFrame)->pd.DataFrame:
    #T是生命周期的时间 Bass_sku是sku的参数记录
    sales=pd.DataFrame(index=T.index, columns=T.columns)
    for sku_name in Bass_sku.index:
        t=T[sku_name][T[sku_name].notna() ]   #去掉 nan 值
        m,p,q=Bass_sku.loc[sku_name]
        res=BM.func(t,m,p,q)
        sales[sku_name]=res   #数值添加
        #TODO:这里考虑加入新的模型
    return sales


#销量数据表中的year 做为index
def sku_sales_index_processing(sku_sales_all:dict):
    sku_classify_sales:pd.DataFrame
    for sku_classify_name,sku_classify_sales in sku_sales_all.items():
        sku_classify_sales.set_index(["year"],inplace=True)

#提取历史销量数据 和下一年（预测年的销量数据）
def sku_sales_past_processing(sku_sales,start_year,end_year,sku_can_use=None,ifBrandNew=False,params_old=None):
    sku_classify_sales: pd.DataFrame
    predict_year=end_year+1
    data:pd.DataFrame
    bool:pd.Series
    index_year=[i for i in range(start_year,predict_year+1)]
    sku_sales_data_all_init=pd.DataFrame(index=index_year)
    sku_t=pd.DataFrame(index=index_year)
    drop_column = []
    #平均销量 以及平均销量增速
    columns_index_As_Asgr=sf.static_class.columns_index_As_Asgr
    #生命周期的前三年相关特征
    columns_index_other1_variance=sf.static_class.columns_index_other1_variance
    #生命周期内 最大销量增速 以及时间等
    columns_index_other2_max=sf.static_class.columns_index_other2_max
    columns_index_other2_max_final_test = ['total_sales']
    columns_index_all=columns_index_As_Asgr+columns_index_other1_variance+columns_index_other2_max_final_test
    self_attribute_new=pd.DataFrame(columns=columns_index_all)

    def get_t_temp(sku,bool):
        t_temp = pd.Series(name=sku, index=bool.index)
        count = 1
        for i in bool.index:
            if bool.loc[i] > 0:
                t_temp[i] = count
            else:
                t_temp[i] = np.nan  # 如果销量为0 那么时间就要被挑掉
            count += 1
        return t_temp
    def get_As_and_Asgr(sku,bool:pd.Series,t):
        #As :average sales
        #Asgr :average sales growth rate
        # bool=bool[bool>0]

        size=bool.size
        As=None
        Asgr=None
        feature_new:pd.Series
        other_feature=None #品规生命周期前三年的PFprice_cv等指标 详细见Data_Acquisition包
        feature_supplement_1=None  #再次补充的特征1
        feature_supplement_2=None  #再次补充特征2
        feature_As_Asgr=None


        if ifBrandNew:
            feature_supplement_1 = CM.get_region_supplementary_feature_other1(sku,
                                                                            [-1,-1],columns_index_other1_variance)
            feature_supplement_2=CM.get_region_supplementary_feature_other2(sku,columns_index_other2_max,bool,t,ifBrandNew)
            feature_As_Asgr=pd.Series(name=sku, index=columns_index_As_Asgr, data=[0, 0])

        elif size>=3:
            front_three_size=3
            # As=np.mean(bool.iloc[:3])
            # Asgr=(bool.iloc[2]-bool.iloc[0])/2
            year_len:list=bool.index.to_list()[-2:]
            year_len.reverse()
            feature_supplement_1=CM.get_region_supplementary_feature_other1(sku,year_len,columns_index_other1_variance)
            feature_supplement_2=CM.get_region_supplementary_feature_other2(sku,columns_index_other2_max,bool,t)
            # feature_As_Asgr = pd.Series(name=sku, index=columns_index_As_Asgr, data=[As, Asgr])

            # if params_old is None:
            #     feature_As_Asgr = pd.Series(name=sku, index=columns_index_As_Asgr, data=[As, Asgr])
            # else:
            feature_As_Asgr =CM.get_region_supplementary_feature_other3(sku,columns_index_As_Asgr,bool,t,front_three_size,params_old if params_old is not None else None)

        else:
            As=np.mean(bool)
            year_len=bool.index.to_list()
            year_len.reverse()
            for i in range(size,2,1):
                year_len.append(year_len[0]-i)   #-1表示当前新品的生命周期还只有1年 -1为单独处理
            feature_supplement_1 = CM.get_region_supplementary_feature_other1(sku, year_len,columns_index_other1_variance)
            feature_supplement_2 = CM.get_region_supplementary_feature_other2(sku, columns_index_other2_max,bool, t)
            feature_As_Asgr=CM.get_region_supplementary_feature_other3(sku,columns_index_As_Asgr,bool,t,size,params_old if params_old is not None else None)
            # if size==1:
            #     Asgr=bool.iloc[0]
            # else:
            #     Asgr=(bool.iloc[-1]-bool.iloc[0])/(size-1)
            # feature_As_Asgr = pd.Series(name=sku, index=columns_index_As_Asgr, data=[As, Asgr])
        feature_new=pd.concat([feature_As_Asgr,feature_supplement_1,feature_supplement_2])
        # print(feature_new)

        return feature_new

    if not sku_sales.empty :  #DataFrame 不为非空
        data=sku_sales.loc[start_year:predict_year,:]
        for sku in data.columns:

            bool=np.trim_zeros(data[sku])
            if bool.size==0:
                #size 为0 表示该品规没有销售
                # sku_can_use 表示当前阶段有拟合结果的sku  销量过少在bass拟合的过程中 是被剔除掉的
                continue
            else:
                if sku_can_use is None:
                    t_temp = get_t_temp(sku, bool)
                    sku_t = pd.concat([sku_t, t_temp], axis=1)
                    feature_new=get_As_and_Asgr(sku,bool.loc[:end_year],t_temp.loc[:end_year])
                    self_attribute_new.loc[sku]=feature_new
                elif sku not in sku_can_use:
                    drop_column.append(sku)
                else:
                    t_temp = get_t_temp(sku, bool)
                    sku_t = pd.concat([sku_t, t_temp], axis=1)
                    feature_new=get_As_and_Asgr(sku,bool.loc[:end_year],t_temp.loc[:end_year])
                    self_attribute_new.loc[sku]=feature_new

        sku_sales_data_all_init=pd.concat([sku_sales_data_all_init,data],axis=1)
    sku_sales_data_all_init=sku_sales_data_all_init.drop(drop_column,axis=1)

    # # 只返回predict_year的数据  返回对象变成了一个series 而不是dataFrame
    # res_sales=sku_sales_data_all_init.loc[predict_year]
    # res_t=sku_t.loc[predict_year]
    # res_sales=res_sales[res_sales>0]
    # res_t=res_t[res_t.notnull()]
    return sku_sales_data_all_init,sku_t,self_attribute_new


# 处理销量预测误差
def get_sale_error(sales_true:pd.DataFrame,sales_hat:pd.DataFrame,predict_year=None):
    sku_error=None

    if sales_true.__class__==pd.DataFrame:
        sku_error=pd.DataFrame(index=sales_true.columns,columns=Errors_columns_name)
        for sku_name in sales_true.columns:
            y_true = sales_true[sku_name][sales_true[sku_name] > 0]
            y_hat = sales_hat[sku_name][sales_hat[sku_name].notna()]
            y_true=y_true.loc[:predict_year-1]
            y_hat=y_hat.loc[:predict_year-1]
            sku_error.loc[sku_name] = get_all_Error_indicator_new(y_true, y_hat)
    elif sales_true.__class__==pd.Series:
        y_true=sales_true[sales_true>0 ]
        y_hat=sales_hat[sales_hat.notna()]
        sku_error_temp=get_all_Error_indicator_new(y_true, y_hat)
        sku_error=pd.DataFrame(columns=Errors_columns_name)
        sku_error.loc[predict_year]=sku_error_temp

    else:
        sku_error=pd.DataFrame()
    return sku_error


def get_train_test(params_all,self_attributes_all,sku_sales_data_all,start_year,end_year,roll_num):
    # 整合数据 提取老品数据
    params_old_init = params_all["data_sku_old"]
    params_semi_brand_init = params_all["sku_semi_brand_new"]
    params_brand_new_init = params_all["sku_brand_new"]
    sku_all_name = self_attributes_all.index

    # 有一些虽然在预测年份之后有销量但是在前面的bass参数拟合的过程中 由于数据量太少 导致无法拟合出bass 参数
    # 导致预测年份有销量的sku 没有对应的bass 参数
    sku_not_can_use = sf.static_class.sku_not_can_use
    sku_can_ues_old =[ i for i in params_old_init.index  if i not in sku_not_can_use]
    params_old_init=params_old_init.loc[sku_can_ues_old]


    # question=["利群(江南忆)","雄狮(红老版)","利群(钱塘)"]
    # 获取销量数据
    sku_old_sales_predict_init, sku_old_t ,new_feature_old= sku_sales_past_processing(sku_sales_data_all["data_sku_old"],
                                                                      start_year, end_year, sku_can_use=sku_can_ues_old,params_old=params_old_init)
    sku_semi_brand_sales_predict_init, sku_semi_brand_t ,new_feature_semi= sku_sales_past_processing(
                                                                        sku_sales_data_all["sku_semi_brand_new"],
                                                                        end_year - 4, end_year)
    sku_brand_new_sales_predict_init, sku_brand_t,new_feature_brand = sku_sales_past_processing(sku_sales_data_all["sku_brand_new"],
                                                                           end_year, end_year,ifBrandNew=True)

    new_feature_total=pd.concat([new_feature_old, new_feature_semi, new_feature_brand],axis=0)
    # new_feature_total=CM.get_region_supplementary_feature_other4(new_feature_old, new_feature_semi, new_feature_brand,self_attributes_all,new_feature_total,roll_num)
    # 获取各个分类后的sku的名称分为老品 在销新品 和完全新品
    self_attributes_old_init = sku_old_sales_predict_init.columns.to_list()
    self_attributes_semi_brand_init = sku_semi_brand_sales_predict_init.columns.to_list()
    self_attributes_brand_init = sku_brand_new_sales_predict_init.columns.to_list()

    # 存数据
    data_save_path="../源文件/地市/测试数据结果/第%d轮/"%(end_year-2016)
    os.makedirs(data_save_path,exist_ok=True)
    sku_old_sales_predict_init.to_csv(data_save_path+"data_sku_old.csv")
    sku_semi_brand_sales_predict_init.to_csv(data_save_path+"sku_semi_brand_new.csv")
    sku_brand_new_sales_predict_init.to_csv(data_save_path+"sku_brand_new.csv")
    self_attributes_total = self_attributes_old_init + self_attributes_semi_brand_init + self_attributes_brand_init

    feature_name: str
    feature_name_need_scaling = []
    feature_name_not_scaling = []

    # self_attributes_all=CM.special_processing(self_attributes_all,self_attributes_brand_init)

    for feature_name in self_attributes_all.columns:  # 遍历属性集合的columns 分类连续数值 和离散数值
        if feature_name.__contains__("type"):
            feature_name_not_scaling.append(feature_name)
        else:
            feature_name_need_scaling.append(feature_name)
    #将新的销量特征加入到特征表中
    new_feature_need_scaling=pd.concat([self_attributes_all.loc[self_attributes_total, feature_name_need_scaling],new_feature_total],axis=1)
    new_feature_need_scaling.to_csv("../源文件/地市/特征未缩放检查_%d.csv"%(end_year))
    # 特征缩放  离散值不缩放 连续数值缩放 标准化
    feature_scaling_done, scaler = feature_scaling(new_feature_need_scaling)
    #特征合并
    self_attributes_total = pd.concat(
        [self_attributes_all.loc[self_attributes_total, feature_name_not_scaling], feature_scaling_done], axis=1)
    # self_attributes_total.to_csv("../源文件/地市/temp.csv")


    #获得各个分类的完整属性
    self_attributes_old_init = self_attributes_total.loc[self_attributes_old_init]
    self_attributes_semi_brand_init = self_attributes_total.loc[self_attributes_semi_brand_init]
    self_attributes_brand_init = self_attributes_total.loc[self_attributes_brand_init]

    #划分训练集 和测试集 两者都包括 自身属性特征 销量 各自销量的对应年份 测试集含有其拟合出的Bass参数
    train_dict={"feature":self_attributes_old_init,"sales":sku_old_sales_predict_init,"t":sku_old_t,"params":params_old_init}
    test_semi_dict={"feature":self_attributes_semi_brand_init,"sales":sku_semi_brand_sales_predict_init,"t":sku_semi_brand_t,"params":params_semi_brand_init}
    test_brand_dict={"feature":self_attributes_brand_init,"sales":sku_brand_new_sales_predict_init,"t":sku_brand_t}

    train_test=dict(zip(train_test_dict_name,(train_dict,test_semi_dict,test_brand_dict)))

    return train_test,scaler

def sales_plot( sales_past:pd.DataFrame,sales_predict:pd.DataFrame,new_sku,roll_num,Model_name,sales_Bass_init:pd.DataFrame,end_year_init):
    print("it is feature plotting")
    return_path="../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/%s/%s销量图片"
    return_path_jpg=return_path+"/%s.jpg"
    return_path_data_predict = "../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/%s/%s拟合销量结果(预测数据).csv"
    return_path_data_past= "../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/%s/%s拟合销量结果(原始数据).csv"


    os.makedirs(return_path%(roll_num,Model_name,"老品"),exist_ok=True)
    os.makedirs(return_path % (roll_num, Model_name, "新品"),exist_ok=True)
    sales_predict.to_csv(return_path_data_predict % (roll_num, Model_name, Model_name), index=True)
    sales_past.to_csv(return_path_data_past % (roll_num, Model_name, Model_name), index=True)

    model_name_plot="RF" if Model_name=="RT" else Model_name
    for sku_name in sales_past.columns:
        sku_past_data=sales_past[sku_name][sales_past[sku_name]>0]
        year_get = sku_past_data.index.to_list()
        sku_predict_data=sales_predict[sku_name][year_get]
        end_year=year_get[-1]-1
        height= sales_past[sku_name][end_year]if sales_past[sku_name][end_year] > sales_predict[sku_name][end_year] else sales_predict[sku_name][end_year]

        figure=plt.figure()
        plt.plot(year_get,sku_past_data,"ro-",alpha=0.8,color="#053AC4",label="原始数据",linewidth=1)

        plt.plot(year_get,sku_predict_data,"r*-",alpha=0.8,color="#4ABEA1",label=model_name_plot+"特征映射后的拟合数据",linewidth=1)
        if sku_name in sales_Bass_init.columns:
            sku_bass_data=sales_Bass_init[sku_name][sales_Bass_init[sku_name]>0]
            year_get_bass=sku_bass_data.index.to_list()
            plt.plot(year_get_bass,sku_bass_data,"r^-",alpha=0.8,color="#B87C4C",label="curve_fit的Bass拟合数据",linewidth=1)
        if year_get[-1]-1==end_year_init:
            plt.vlines(x=year_get[-1]-1,ymin=0,ymax=height,linestyles="--",color="#F21F3B")
        plt.legend(loc="upper left")
        plt.title(sku_name)
        plt.xlabel("年份")
        plt.xticks(year_get)
        plt.ylabel("销量")

        if sku_name not in new_sku:
            plt.savefig( return_path_jpg % (roll_num,Model_name,"老品",sku_name),dpi=300)
        else:
            plt.savefig(return_path_jpg % (roll_num,Model_name,"新品",sku_name),dpi=300)
    # print(return_path_data_predict % (roll_num,Model_name,Model_name))

    print("feature plotting done")




#特征值的还原
def feature_inverse_transform(feature_all:pd.DataFrame,scaler: prep.StandardScaler):
    i:str
    feature_need_inverse=[i for i in feature_all.columns if not i.__contains__("type")]
    feature_dont_inverse=[i for i in feature_all.columns if  i.__contains__("type")]
    sku_all=feature_all.index


    feature_need_inverse_done=scaler.inverse_transform(feature_all[feature_need_inverse])
    feature_need_inverse_done=pd.DataFrame(data=feature_need_inverse_done,columns=feature_need_inverse,index=sku_all)
    feature_all_done=pd.concat([feature_all[feature_dont_inverse],feature_need_inverse_done],axis=1)
    #  反转回去的特征值存在-0e-13的情况是否需要完全变成0
    return feature_all_done


