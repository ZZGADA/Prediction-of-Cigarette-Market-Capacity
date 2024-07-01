import os

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
from Dao import static_file as sf


#获得所有新品的预测年份的销量误差  特征映射出来的
def get_Each_Machine_Model_MAPE_all_new_feature_mapping(path):
    Machine_Model_name = ['SVR','RT', 'KNN', 'XG']
    all_machine_model_MAPE=pd.DataFrame(columns=Machine_Model_name,index=[2018,2019,2020,2021,2022,"mean_MAPE"])
    Mape_head_path="../源文件%s/销量预测误差统计_test/第%d轮滚动结果/%s/新品_预测年份(%d)_销量误差(特征映射).csv"
    for end_year in range(2017, 2022):
        roll_num=end_year-2016
        predict_year=end_year+1
        for model_name in Machine_Model_name:
            Mape_head_path_model=Mape_head_path%(path,roll_num,model_name,predict_year)
            sales_error=pd.read_csv(Mape_head_path_model,index_col=0)
            sales_error_Mape=sales_error["MAPE"].iloc[0]
            sales_error_year=sales_error.index.to_list()[0]
            all_machine_model_MAPE.loc[sales_error_year,model_name]=sales_error_Mape
    for model_name in all_machine_model_MAPE.columns:
        all_machine_model_MAPE.loc["mean_MAPE",model_name]=all_machine_model_MAPE.loc[2018:2022,model_name].mean()
    all_machine_model_MAPE.to_csv("../源文件%s/销量预测误差统计_test/所有新品预测结果_MAPE.csv"%path)
# 新品_有销量的预测年份(%d)_销量误差(特征映射)

#获得有销量的新品的销量误差 特征映射出来的
def get_Each_Machine_Model_MAPE_semi_feature_mapping(path):
    Machine_Model_name = ['SVR','RT', 'KNN', 'XG']
    all_machine_model_MAPE=pd.DataFrame(columns=Machine_Model_name,index=[2018,2019,2020,2021,2022,"mean_MAPE"])
    Mape_head_path="../源文件%s/销量预测误差统计_test/第%d轮滚动结果/%s/新品_有销量的预测年份(%d)_销量误差(特征映射).csv"
    for end_year in range(2017, 2022):
        roll_num=end_year-2016
        predict_year=end_year+1
        for model_name in Machine_Model_name:
            Mape_head_path_model=Mape_head_path%(path,roll_num,model_name,predict_year)
            sales_error=pd.read_csv(Mape_head_path_model,index_col=0)
            sales_error=sales_error.set_index(pd.Index([predict_year]))
            sales_error_Mape=sales_error["MAPE"].iloc[0]
            sales_error_year=predict_year
            all_machine_model_MAPE.loc[sales_error_year,model_name]=sales_error_Mape
    for model_name in all_machine_model_MAPE.columns:
        all_machine_model_MAPE.loc["mean_MAPE",model_name]=all_machine_model_MAPE.loc[2018:2022,model_name].mean()
    all_machine_model_MAPE.to_csv("../源文件%s/销量预测误差统计_test/新品中有销量的预测结果_MAPE.csv"%path)

#获得新品中可以使用curve_fit出来的销量误差
def get_Each_Machine_Model_MAPE_semi_curve_fit(path):
    Machine_Model_name = ['SVR','RT', 'KNN', 'XG']
    all_machine_model_MAPE=pd.DataFrame(columns=["curve_fit"],index=[2018,2019,2020,2021,2022,"mean_MAPE"])
    Mape_head_path="../源文件%s/销量预测误差统计_test/第%d轮滚动结果/%s/新品_预测年份(%d)_销量误差(curve_fit).csv"
    for end_year in range(2017, 2022):
        roll_num=end_year-2016
        predict_year=end_year+1
        for model_name in Machine_Model_name:
            Mape_head_path_model=Mape_head_path%(path,roll_num,model_name,predict_year)
            sales_error=pd.read_csv(Mape_head_path_model,index_col=0)
            sales_error_Mape=sales_error["MAPE"].iloc[0]
            sales_error_year=sales_error.index.to_list()[0]
            all_machine_model_MAPE.loc[sales_error_year,"curve_fit"]=sales_error_Mape
    for model_name in all_machine_model_MAPE.columns:
        all_machine_model_MAPE.loc["mean_MAPE",model_name]=all_machine_model_MAPE.loc[2018:2022,model_name].mean()
    all_machine_model_MAPE.to_csv("../源文件%s/销量预测误差统计_test/新品中可以使用curve_fit的预测结果_MAPE.csv"%path)

def get_Each_Machine_Model_MAPE_old_curve_fit(path):
    Machine_Model_name = ['SVR','RT', 'KNN', 'XG']
    all_machine_model_MAPE=pd.DataFrame(columns=["curve_fit"],index=[2018,2019,2020,2021,2022,"mean_MAPE"])
    Mape_head_path="../源文件%s/销量预测误差统计_test/第%d轮滚动结果/%s/老品_预测年份(%d)_销量误差(curve_fit).csv"
    for end_year in range(2017, 2022):
        roll_num=end_year-2016
        predict_year=end_year+1
        for model_name in Machine_Model_name:
            Mape_head_path_model=Mape_head_path%(path,roll_num,model_name,predict_year)
            sales_error=pd.read_csv(Mape_head_path_model,index_col=0)
            sales_error_Mape=sales_error["MAPE"].iloc[0]
            sales_error_year=sales_error.index.to_list()[0]
            all_machine_model_MAPE.loc[sales_error_year,"curve_fit"]=sales_error_Mape
    for model_name in all_machine_model_MAPE.columns:
        all_machine_model_MAPE.loc["mean_MAPE",model_name]=all_machine_model_MAPE.loc[2018:2022,model_name].mean()
    all_machine_model_MAPE.to_csv("../源文件%s/销量预测误差统计_test/老品curve_fit的预测结果_MAPE.csv"%path)
'''
以上函数都是按每年的维度来统筹误差的 暂时不考虑使用 但是可以用来观察每年的误差情况
'''

def get_Each_sku_Mape_all_new(path,rolling_start,rolling_end):
    Machine_Model_name = ['SVR', 'RT', 'KNN', 'XG']
    original_brand_new_sales_path_head="../源文件/%s/测试数据结果/第%d轮/sku_brand_new.csv"  #完全新品原始销量
    original_semi_sales_path_head="../源文件/%s/测试数据结果/第%d轮/sku_semi_brand_new.csv"  #有历史销量的新品
    original_old_sales_path_head="../源文件/%s/测试数据结果/第%d轮/data_sku_old.csv"
    predict_sales_each_model_path_head="../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/%s拟合销量结果(预测数据).csv"
    revise_old_sku_name_path_head="../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/XG/XG拟合销量结果(原始数据).csv"
    save_path="../源文件/%s/销量预测误差统计_test/各个SKU的预测销量统计"%(path)
    os.makedirs(save_path,exist_ok=True)
    roll_year=[i for i in range(2018,2023)]
    all_year_index=[i for i in range(2006,rolling_end+1)]
    feature_mapping_columns = ["%d年及以前_特征映射" % i for i in range(rolling_start, rolling_end)]
    curve_fit_columns = ["%d年及以前_curve_fit" % i for i in range(rolling_start, rolling_end)]
    original_new_sales_sku_name_total=[]
    original_old_sales_sku_name_total=[]

    original_semi_sales_data:pd.Series
    original_brand_new_sales_data: pd.Series



    def get_rolling_original():
        for end_year in range(2017,2022):
            roll_num=end_year-2016
            predict_year=end_year+1
            original_brand_new_sales_path =original_brand_new_sales_path_head%(path,roll_num)
            original_semi_sales_path = original_semi_sales_path_head%(path,roll_num)
            original_old_sales_path = original_old_sales_path_head%(path,roll_num)
            revise_old_sku_name_path=revise_old_sku_name_path_head%(path,roll_num)


            original_brand_new_sales_data=pd.read_csv(original_brand_new_sales_path,index_col=0).loc[predict_year]   #提取预测年的销量 完全新品
            original_semi_sales_data=pd.read_csv(original_semi_sales_path,index_col=0).loc[predict_year]   #提取预测年的销量 在销新品  历史销量不需要看 历史只看误差
            original_old_sales_data=pd.read_csv(original_old_sales_path,index_col=0).loc[predict_year]   #提取预测年的销量  老品
            revise_old_sku_name_data=pd.read_csv(revise_old_sku_name_path,index_col=0).loc[predict_year]   #老品修订的名字

            original_brand_new_sales_data =original_brand_new_sales_data[original_brand_new_sales_data!=0]  #去掉0
            original_semi_sales_data = original_semi_sales_data[original_semi_sales_data!=0]  #去掉0
            original_old_sales_data= original_old_sales_data[original_old_sales_data!=0]    #去掉0
            revise_old_sku_name=revise_old_sku_name_data[revise_old_sku_name_data!=0].index.to_list()  #去掉0



            original_brand_new_sales_sku=original_brand_new_sales_data.index.to_list()
            original_semi_sales_sku=original_semi_sales_data.index.to_list()
            original_old_sales_sku =original_old_sales_data.index.to_list()

            original_old_sales_sku=[sku for sku in original_old_sales_sku if sku in revise_old_sku_name]



            original_new_sales_all_sku_name=original_brand_new_sales_sku+original_semi_sales_sku#获取新品在该预测年份有销量的sku名字
            original_new_sales_sku_name_total.extend(original_new_sales_all_sku_name)   #滚动中新品的记录
            original_old_sales_sku_name_total.extend(original_old_sales_sku)
            if end_year==2021:
                rolling_new_sku_total_original_name=set(original_new_sales_sku_name_total)  #滚动完成后 所有的记录的新品名字  去重 brand_new 会变成semi
                rolling_old_sku_total_original_name=set(original_old_sales_sku_name_total)  #滚动后 记录所有老品的sku名称 去重
                return rolling_new_sku_total_original_name,rolling_old_sku_total_original_name

    rolling_new_sku_total_original_name,rolling_old_sku_total_original_name=get_rolling_original()

    # input()
    data_true = pd.DataFrame(columns=rolling_new_sku_total_original_name,index=roll_year,data=0)
    data_old_true=pd.DataFrame(columns=rolling_old_sku_total_original_name,index=roll_year,data=0)

    Model_sales=dict(zip(Machine_Model_name,[pd.DataFrame(columns=rolling_new_sku_total_original_name,
                                                          index=roll_year,data=0)  for i in range(4)]))
    Model_old_sales=dict(zip(Machine_Model_name,[pd.DataFrame(columns=rolling_old_sku_total_original_name,
                                                              index=roll_year,data=0) for i in range(4)]))
    Model_sale_past_feature_mapping_curve_fit=dict(zip(Machine_Model_name,
                                                       dict(zip(rolling_new_sku_total_original_name,
                                                                [ pd.DataFrame(index=all_year_index,columns=feature_mapping_columns+curve_fit_columns)
                                                                    for i in range(len(rolling_new_sku_total_original_name))]))))

    for end_year in range(2017, 2022):
        roll_num = end_year - 2016
        predict_year = end_year + 1
        original_brand_new_sales_path = original_brand_new_sales_path_head % (path, roll_num)
        original_semi_sales_path = original_semi_sales_path_head % (path, roll_num)
        original_old_sales_path = original_old_sales_path_head % (path, roll_num)
        revise_old_sku_name_path = revise_old_sku_name_path_head % (path, roll_num)

        original_brand_new_sales_data = pd.read_csv(original_brand_new_sales_path, index_col=0).loc[
            predict_year]  # 提取预测年的销量 完全新品
        original_semi_sales_data = pd.read_csv(original_semi_sales_path, index_col=0).loc[predict_year]  # 提取预测年的销量 在销新品  历史销量不需要看 历史只看误差
        original_old_sales_data = pd.read_csv(original_old_sales_path, index_col=0).loc[predict_year]  # 提取预测年的销量  老品
        revise_old_sku_name_data = pd.read_csv(revise_old_sku_name_path, index_col=0).loc[predict_year]  # 老品修订的名字

        original_brand_new_sales_data = original_brand_new_sales_data[original_brand_new_sales_data != 0]  # 去掉0
        original_semi_sales_data = original_semi_sales_data[original_semi_sales_data != 0]  # 去掉0
        original_old_sales_data = original_old_sales_data[original_old_sales_data != 0]  # 去掉0


        original_brand_new_sales_sku = original_brand_new_sales_data.index.to_list()
        original_semi_sales_sku = original_semi_sales_data.index.to_list()
        original_old_sales_sku = original_old_sales_data.index.to_list()
        revise_old_sku_name = revise_old_sku_name_data[revise_old_sku_name_data != 0].index.to_list()  # 去掉0
        original_old_sales_sku = [sku for sku in original_old_sales_sku if sku in revise_old_sku_name]

        old_sku=[sku for sku in original_old_sales_sku if sku in   rolling_new_sku_total_original_name]  #如果当前年份的分类中 老品中包含了新品的sku名字 则表示 之前的新品已经被划分为老品了 当然也存在之前的新品现在不再销售量 在分类的时候已经被过滤掉了的情况
        old_original_sku=[sku for sku in original_old_sales_sku if sku in   rolling_old_sku_total_original_name]   #当前滚动分类中 提取原始老品的品规
        new_semi_sku=[sku for sku in original_semi_sales_sku if sku in   rolling_old_sku_total_original_name]
        get_old(predict_sales_each_model_path_head, path,
                roll_num, predict_year, old_sku , Model_sales,data_true,
                Model_sale_past_feature_mapping_curve_fit)
        get_brand_new(predict_sales_each_model_path_head, path,
                      roll_num, predict_year, original_brand_new_sales_sku , Model_sales,data_true)
        get_semi(predict_sales_each_model_path_head, path,
                 roll_num, predict_year, original_semi_sales_sku, Model_sales,data_true,
                 Model_sale_past_feature_mapping_curve_fit)

        #获得老品品规的预测销量统计
        # get_semi(predict_sales_each_model_path_head, path, roll_num, predict_year, new_semi_sku, Model_old_sales,data_old_true)
        # get_old(predict_sales_each_model_path_head, path, roll_num, predict_year, old_original_sku , Model_old_sales,data_old_true,ifCurve_fit=True)


    calculate_Mape(Model_sales,data_true,save_path)
    # calculate_Mape(Model_old_sales,data_old_true,save_path,True) #老品误差计算


def get_brand_new(predict_sales_each_model_path_head,path,roll_num,predict_year,brand_new_sku_name,Model_sales,data_true):
    original_data_path_head="../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/%s拟合销量结果(原始数据).csv"
    Machine_Model_name = ['SVR', 'RT', 'KNN', 'XG']
    for model_name in Machine_Model_name:
        original_data_path=original_data_path_head%(path,roll_num,model_name,model_name)
        predict_sales_each_model_path=predict_sales_each_model_path_head%(path,roll_num,model_name,model_name)
        predict_sales=pd.read_csv(predict_sales_each_model_path,index_col=0).loc[predict_year,brand_new_sku_name]
        original_data=pd.read_csv( original_data_path,index_col=0).loc[predict_year,brand_new_sku_name]

        Model_sales[model_name].loc[predict_year,brand_new_sku_name]=predict_sales  #将预测数据放入dataFrame中
        data_true.loc[predict_year,brand_new_sku_name]=original_data

        #原始数据一个表 每一个模型一个数据表  columns为年份 index 为sku的名字

def get_semi(predict_sales_each_model_path_head,path,roll_num,predict_year,
             semi_sku_name,Model_sales,data_true,
             Model_sale_past_feature_mapping_curve_fit):
    error_path_curve_fit_head="../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/新品_历史_销量误差(curve_fit).csv"  #curve_fit出来的新品历史误差
    error_path_feature_mapping_head="../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/新品_历史_销量误差(特征映射).csv" #特征映射出来的新品历史误差  这个数量大于curve_fit的
    predict_sales_each_model_path_head_curve_fit="../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/Bass函数(curve_fit)拟合销量结果(在销新品).csv"   #curve_fit的预测销量结果
    original_data_path_head = "../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/%s拟合销量结果(原始数据).csv"
    '''predict_sales_each_model_path_head=
    "../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/%s拟合销量结果(预测数据).csv"
    '''
    for model_name in sf.static_class.Machine_Model_name_used:
            original_data_path = original_data_path_head % (path, roll_num, model_name, model_name)
            predict_sales_each_model_path=predict_sales_each_model_path_head%(path,roll_num,model_name,model_name)   #这个是特征映射出来的销量
            feature_mapping_sale=pd.read_csv(predict_sales_each_model_path,index_col=0)
            predict_sales_feature_mapping=feature_mapping_sale.loc[predict_year,semi_sku_name] #去掉0了的   #这个是特征映射出来的预测销量

            # feature_mapping_sale_past=feature_mapping_sale.loc[]

            predict_sales_each_model_path_curve_fit=predict_sales_each_model_path_head_curve_fit%(path,roll_num,model_name)
            predict_sales_curve_fit=pd.read_csv(predict_sales_each_model_path_curve_fit,index_col=0).loc[predict_year]#curve_fit 出来的预测销量 没有去0
            predict_sales_curve_fit=predict_sales_curve_fit[predict_sales_curve_fit>0]  #去掉0

            error_path_curve_fit=error_path_curve_fit_head%(path,roll_num,model_name)
            error_path_feature_mapping=error_path_feature_mapping_head%(path,roll_num,model_name)

            error_curve_fit=pd.read_csv(error_path_curve_fit,index_col=0)["MAPE"].rename("MAPE_curve_fit")
            error_feature_mapping=pd.read_csv(error_path_feature_mapping,index_col=0)["MAPE"].rename("MAPE_feature_mapping").loc[semi_sku_name]   #只要predict_year有销量的
            error_compare=pd.concat([error_feature_mapping,error_curve_fit,pd.DataFrame(columns=[predict_year])],axis=1).fillna(np.PINF)   #设置为无限大
            original_data = pd.read_csv(original_data_path, index_col=0).loc[predict_year, semi_sku_name]   #原始数据


            drop_column=[index for index in error_compare.index if error_compare["MAPE_feature_mapping"][index]==np.PINF]  #因为concat是求并集那么存在可以curve_fit的品规但是没有销量了 但是fillna的过程中我们将特征映射的残缺之设置为inf 会出现问题 所有需要剔除掉
            error_compare=error_compare.drop(drop_column,axis=0)
            for sku in error_compare.index:
                #如果curve_fit的新品历史销量的误差小于特征映射的 那么我们就选curve_fit预测出来的销量 否则选择特征映射出来的销量 再者如果说curve_fit没有该品规的预测销量 我们设置为inf然后选择特征映射出来的预测销量
                #如新品销量小于3年的 那么没有办法使用curve_fit
                error_compare.loc[sku,predict_year]= predict_sales_feature_mapping[sku] if error_compare.loc[sku,"MAPE_feature_mapping"]<error_compare.loc[sku,"MAPE_curve_fit"] else predict_sales_curve_fit[sku]
                # error_compare.loc[sku, predict_year] = predict_sales_curve_fit[sku]
            Model_sales[model_name].loc[predict_year,semi_sku_name]=error_compare[predict_year]
            data_true.loc[predict_year, semi_sku_name] = original_data

def get_old(predict_sales_each_model_path_head,path,roll_num,
            predict_year,old_sku,Model_sales,data_true,
            Model_sale_past_feature_mapping_curve_fit,ifCurve_fit=False):
    print("it is get old")
    error_path_curve_fit_head = "../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/老品_历史_销量误差(curve_fit).csv"  # curve_fit出来的新品历史误差
    error_path_feature_mapping_head = "../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/老品_历史_销量误差(特征映射).csv"  # 特征映射出来的新品历史误差  这个数量大于curve_fit的
    predict_sales_each_model_path_head_curve_fit = "../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/Bass函数(curve_fit)拟合销量结果(老品).csv"  # curve_fit的预测销量结果
    original_data_path_head = "../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/%s拟合销量结果(原始数据).csv"

    for model_name in sf.static_class.Machine_Model_name_used:
        original_data_path = original_data_path_head % (path, roll_num, model_name, model_name)
        predict_sales_each_model_path = predict_sales_each_model_path_head % (
        path, roll_num, model_name, model_name)  # 这个是特征映射出来的销量
        predict_sales_feature_mapping = pd.read_csv(predict_sales_each_model_path, index_col=0).loc[
            predict_year, old_sku]  # 去掉0了的   #这个是特征映射出来的预测销量
        predict_sales_each_model_path_curve_fit = predict_sales_each_model_path_head_curve_fit % (
        path, roll_num, model_name)
        predict_sales_curve_fit = pd.read_csv(predict_sales_each_model_path_curve_fit, index_col=0).loc[
            predict_year]  # curve_fit 出来的预测销量 没有去0
        predict_sales_curve_fit = predict_sales_curve_fit[predict_sales_curve_fit > 0]  # 去掉0

        error_path_curve_fit = error_path_curve_fit_head % (path, roll_num, model_name)
        error_path_feature_mapping = error_path_feature_mapping_head % (path, roll_num, model_name)

        error_curve_fit = pd.read_csv(error_path_curve_fit, index_col=0)["MAPE"].rename("MAPE_curve_fit")
        error_feature_mapping = \
        pd.read_csv(error_path_feature_mapping, index_col=0)["MAPE"].rename("MAPE_feature_mapping").loc[
            old_sku]  # 只要predict_year有销量的
        original_data = pd.read_csv(original_data_path, index_col=0).loc[predict_year, old_sku]  # 原始数据

        error_compare = pd.concat([error_feature_mapping, error_curve_fit, pd.DataFrame(columns=[predict_year])],
                                  axis=1).fillna(np.PINF)  # 设置为无限大

        drop_column = [index for index in error_compare.index if
                       error_compare["MAPE_feature_mapping"][index] == np.PINF]
        error_compare = error_compare.drop(drop_column, axis=0)
        for sku in error_compare.index:
            # 如果curve_fit的新品历史销量的误差小于特征映射的 那么我们就选curve_fit预测出来的销量 否则选择特征映射出来的销量 再者如果说curve_fit没有该品规的预测销量 我们设置为inf然后选择特征映射出来的预测销量
            # 如新品销量小于3年的 那么没有办法使用curve_fit
            if not ifCurve_fit:
                error_compare.loc[sku, predict_year] = predict_sales_feature_mapping[sku] if error_compare.loc[sku, "MAPE_feature_mapping"] < \
                                                                                             error_compare.loc[
                                                                                                 sku, "MAPE_curve_fit"] else \
                predict_sales_curve_fit[sku]
            else:
                error_compare.loc[sku,predict_year]=predict_sales_curve_fit[sku]  #老品curve_fit就用这个
        Model_sales[model_name].loc[predict_year,old_sku] = error_compare[predict_year]
        data_true.loc[predict_year, old_sku] = original_data


            # Model_sales[model_name]=pd.concat([Model_sales[model_name],predict_sales],axis=1)  #将预测数据放入dataFrame中

            #原始数据一个表 每一个模型一个数据表  columns为年份 index 为sku的名字


def calculate_Mape(Model_sales:dict,data_true,save_path,ifOld=False,ifCurve_fit=False):
    threshold_flag_all=[10,15,20]
    all_error=pd.DataFrame(columns=['MSE', 'MAE', 'RMSE','MAPE','SMAPE'],index=Model_sales.keys())
    for threshold_flag in threshold_flag_all:
        for year in [2019,2020,2021]:
            threshold_path = None
            year_reject=None
            if year==2020:
                year_reject="_2020"
            elif year==2021:
                year_reject="_2020_2021"
            for model_name,model_sales in Model_sales.items():
                model_sales: pd.DataFrame()
                model_sales_new=model_sales.T
                if ifOld:
                    model_sales.to_csv(save_path+"/老品/老品预测销量统计(%s).csv"%(model_name))
                else:
                    model_sales.to_csv(save_path+"/新品/新品预测销量统计(%s).csv"%(model_name))
                error_indicator=pd.DataFrame(columns=['MSE', 'MAE', 'RMSE','MAPE','SMAPE'])
                for sku in model_sales_new.index:
                    sku:str
                    sales_hat = model_sales_new.loc[sku][model_sales_new.loc[sku] != 0]
                    sales_true=data_true[sku][sales_hat.index]

                    # year_len=sales_hat.index.to_list()
                    # if sku=='南京_利群(夜西湖)':
                    #     print(year_len)
                    #     input()


                    if np.mean(sales_true)<threshold_flag:
                        continue
                    if sku.__contains__("雄狮"):
                        continue
                    # if len(year_len)==1 and year_len[0] in [2019,2020,2021]:
                    #     continue

                    if year>2019:
                        try:
                            s1=sales_true.loc[2020]
                            s2=sales_hat.loc[2020]
                            if abs((s2-s1)/s1)>1.5:
                                continue
                        except:
                            pass
                    if year>2020:
                        try:
                            s1 = sales_true.loc[2021]
                            s2 = sales_hat.loc[2021]
                            if abs((s2 - s1) / s1) > 1.5:
                                continue
                        except:
                            pass
                    res=Model3.get_all_Error_indicator_new(sales_true,sales_hat)
                    error_indicator.loc[sku]=res
                #创建目录
                if ifOld:
                    if ifCurve_fit:
                        threshold_path = "/老品/老品(curve_fit)/老品阈值处理(剔除五年平均销量小于%d%s)" % (threshold_flag, year_reject)
                        os.makedirs(save_path + '/老品/老品(curve_fit)', exist_ok=True)
                        os.makedirs(save_path + '/老品/老品(curve_fit)'+ "/所有老品预测结果", exist_ok=True)
                    else:
                        threshold_path = "/老品/老品阈值处理(剔除五年平均销量小于%d%s)" % (threshold_flag,year_reject)
                        os.makedirs(save_path + threshold_path, exist_ok=True)
                        os.makedirs(save_path+threshold_path+"/所有老品预测结果",exist_ok=True)
                else:
                    threshold_path="/新品/新品阈值处理(剔除五年平均销量小于%d%s)"%(threshold_flag,year_reject)
                    os.makedirs(save_path+threshold_path,exist_ok=True)
                    os.makedirs(save_path + threshold_path + "/所有新品预测结果", exist_ok=True)
                #提取平均误差
                error_indicator.loc["mean_Indicator"]=[np.nan,np.nan,np.nan,np.nan,np.nan]
                for indicator_name in error_indicator.columns:
                    error_indicator.loc["mean_Indicator"][indicator_name]=np.mean(error_indicator[indicator_name])
                all_error.loc[model_name]= error_indicator.loc["mean_Indicator"]
                if ifOld:
                    each_sku_sale_processing(save_path + threshold_path + "/所有老品预测结果", model_name, model_sales,
                                             error_indicator)
                else:
                    each_sku_sale_processing(save_path+threshold_path+"/所有新品预测结果",model_name,model_sales,error_indicator)
                select_columns = error_indicator.index.to_list()
                select_columns.remove("mean_Indicator")
                if ifOld:
                    model_sales=model_sales.loc[:,select_columns]
                    error_indicator.to_csv(save_path + threshold_path + "/老品预测销量误差(%s).csv" % (model_name))
                    model_sales.to_csv(save_path + threshold_path + "/老品预测销量统计(%s).csv" % (model_name))
                else:
                    model_sales = model_sales.loc[:, select_columns]
                    error_indicator.to_csv(save_path+threshold_path+"/新品预测销量误差(%s).csv"%(model_name))
                    model_sales.to_csv(save_path+threshold_path+"/新品预测销量统计(%s).csv"%(model_name))
                # sales_plot(data_true,model_sales,model_name)
            all_error.to_csv(save_path+threshold_path+"所有模型误差统计.csv")
    if ifOld:
        data_true.to_csv(save_path + "/老品/原始销量统计.csv")
    else:
        data_true.to_csv(save_path+"/新品/原始销量统计.csv")

def each_sku_sale_processing(path,model_name,model_sale:pd.DataFrame,error_all:pd.DataFrame):
    #每一个sku 都将会是一个表  一列原始数据 一列预测数据 旁边附上误差表
    original_sale_path="..\源文件\地市\特征结果\%s.csv"
    return_path=path+"/"+model_name
    os.makedirs(return_path,exist_ok=True)
    select_columns=error_all.index.to_list()
    select_columns.remove("mean_Indicator")
    model_sale=model_sale.loc[:,select_columns]
    for sku_name in model_sale.columns:
        original_sale=pd.read_csv(original_sale_path%(sku_name),index_col=0)['sale'].rename("original_sale")
        predict_sale=model_sale[sku_name].rename("predict_sale")
        sku_data=pd.concat([original_sale,predict_sale,pd.DataFrame(columns=["",""]+sf.static_class.Errors_columns_name)],axis=1)
        error_sku=error_all.loc[sku_name]
        first_index=sku_data.index.to_list()[0]
        sku_data.loc[first_index,sf.static_class.Errors_columns_name]=error_sku
        sku_data.to_csv(return_path+"\%s.csv"%(sku_name))




def sales_plot( sales_past:pd.DataFrame,sales_predict:pd.DataFrame,Model_name,ifOld=False):
    print("it is feature plotting")

    return_path="../源文件/地市/销量预测误差统计_test/各个SKU的预测销量统计/%s/%s"
    return_path_jpg=return_path+"/%s.jpg"

    if ifOld:
        os.makedirs(return_path%("老品",Model_name),exist_ok=True)
    else:
        os.makedirs(return_path % ("新品", Model_name), exist_ok=True)
    for sku_name in sales_past.columns:
        sku_past_data=sales_past[sku_name][sales_past[sku_name]>0]
        year_get = sku_past_data.index.to_list()
        sku_predict_data=sales_predict[sku_name][year_get]

        figure=plt.figure()
        plt.plot(year_get,sku_past_data,"ro-",alpha=0.8,color="#053AC4",label="原始数据",linewidth=1)
        plt.plot(year_get,sku_predict_data,"r*-",alpha=0.8,color="#4ABEA1",label="预测数据(%s)"%(Model_name),linewidth=1)

        plt.legend(loc="upper right")
        plt.title(sku_name)
        plt.xlabel("年份")
        plt.xticks(year_get)
        plt.ylabel("销量")

        if ifOld: #如果为老品
            plt.savefig( return_path_jpg % ("老品",Model_name,sku_name),dpi=300)
        else:
            plt.savefig(return_path_jpg % ("新品", Model_name, sku_name), dpi=300)


    print("feature plotting done")

def get_train_set_Mape_feature_mapping_roll_year(path,start_year,end_year,old_or_new):
    '''
        按照滚动年份来统计训练集的历史销量误差
        start_year表示滚动预测中 预测年份的起始年份 end_year表示滚动预测中 预测年份的终止年份
        old_or_new 表示是老品还是新品
    '''
    print("计算训练集误差_按年份")
    Machine_Model_name = ['SVR', 'RT', 'KNN', 'XG']
    error_path_feature_mapping_train_set_head = "../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/%s_历史_销量误差(特征映射).csv"  # 特征映射出来的新品历史误差  这个数量大于curve_fit的
    return_path_head="../源文件/%s/销量预测误差统计_test/各个SKU的预测销量统计/%s/%s训练集历史误差_按年份(%s).csv"

    for model_name in Machine_Model_name:
        all_roll_year_train_set_Mean_error = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE', 'SMAPE'])
        for roll_year in range(start_year, end_year + 1):
            roll_num = roll_year - start_year + 1
            error_path_feature_mapping_train_set=error_path_feature_mapping_train_set_head%(path,roll_num,model_name,old_or_new)
            roll_year_sku_error=pd.read_csv(error_path_feature_mapping_train_set,index_col=0)
            for error_indicator in all_roll_year_train_set_Mean_error.columns:
                all_roll_year_train_set_Mean_error.loc[roll_year,error_indicator]=np.mean(roll_year_sku_error[error_indicator])
        for error_indicator in all_roll_year_train_set_Mean_error.columns:
            all_roll_year_train_set_Mean_error.loc["mean_error", error_indicator] = np.mean(all_roll_year_train_set_Mean_error.loc[start_year:end_year,error_indicator])
        return_path=return_path_head%(path,old_or_new,old_or_new,model_name)
        all_roll_year_train_set_Mean_error.to_csv(return_path)

def get_train_set_Mape_feature_mapping_sku(path,start_year,end_year,old_or_new):
    '''
        按照sku来统计训练集的历史销量误差
        start_year表示滚动预测中 预测年份的起始年份 end_year表示滚动预测中 预测年份的终止年份
        old_or_new 表示是老品还是新品
    '''
    print("计算训练集误差_按sku")
    Machine_Model_name = ['SVR', 'RT', 'KNN', 'XG']
    error_path_feature_mapping_train_set_head = "../源文件/%s/销量预测误差统计_test/第%d轮滚动结果/%s/%s_历史_销量误差(特征映射).csv"  # 特征映射出来的新品历史误差  这个数量大于curve_fit的
    return_path_head="../源文件/%s/销量预测误差统计_test/各个SKU的预测销量统计/%s/%s训练集历史误差_按sku.csv"
    return_each_sku_error_head="../源文件/%s/销量预测误差统计_test/各个SKU的预测销量统计/%s/所有%s的误差集合"
    return_each_sku_error=return_each_sku_error_head%(path,old_or_new,old_or_new)
    os.makedirs(return_each_sku_error,exist_ok=True)
    error_columns=['MSE', 'MAE', 'RMSE','MAPE','SMAPE']
    all_roll_year_train_set_Mean_error=pd.DataFrame(columns=error_columns,index=Machine_Model_name)
    for model_name in Machine_Model_name:
        each_model_all_roll_year_train_set_Mean_error=dict()  #dict 键为sku的名称 值为sku的每年滚动计算出来的五个误差指标组成的DataFrame
        for roll_year in range(start_year,end_year+1):
            roll_num = roll_year - start_year + 1
            error_path_feature_mapping_train_set = error_path_feature_mapping_train_set_head % (path, roll_num, model_name, old_or_new)
            roll_year_sku_error = pd.read_csv(error_path_feature_mapping_train_set, index_col=0)
            data_temp: pd.DataFrame

            for sku in roll_year_sku_error.index:
                if sku not in each_model_all_roll_year_train_set_Mean_error.keys():   #如果字典中不存在那么创建dataFrame
                    each_model_all_roll_year_train_set_Mean_error[sku]=pd.DataFrame(columns=error_columns)
                    data_temp=each_model_all_roll_year_train_set_Mean_error[sku]  #提取dataFrame 这是一个引用
                    data_temp.loc[roll_year]=roll_year_sku_error.loc[sku]   #将当前数据放入字典
                else:
                    data_temp = each_model_all_roll_year_train_set_Mean_error[sku]  # 提取dataFrame 这是一个引用
                    data_temp.loc[roll_year]=roll_year_sku_error.loc[sku]  #将数据放入字典
        #一个机器学习模型走完之后 获得老品所有的滚动年份误差  然后开始计算每个品规的平均误差
        all_mean_error_sku=pd.DataFrame(columns=error_columns)
        for sku_name ,error_sku in each_model_all_roll_year_train_set_Mean_error.items():
            error_sku.to_csv(return_each_sku_error+"/%s.csv"%(sku_name))
            for error_Indicator in error_columns:  #计算每一个品规的平均误差
                all_mean_error_sku.loc[sku_name,error_Indicator]=np.mean(error_sku[error_Indicator])  #计算每一个品规的每一个误差指标的平均值 并放入all_mean_error_sku中
        for error_Indicator in error_columns:
            all_roll_year_train_set_Mean_error.loc[model_name,error_Indicator]=np.mean(all_mean_error_sku[error_Indicator])   #将该模型下 所有品规计算其平均误差
    return_path = return_path_head % (path, old_or_new,old_or_new)
    all_roll_year_train_set_Mean_error.to_csv(return_path)

def get_train_set_Error_feature_mapping_plus_curve_fit(rolling_start,rolling_end):
    print("获得训练集feature_mapping+curve_fit的所有结果")
    '''测试数据中的第五轮滚动结果将包含所有老品数据'''
    rolling_end_year=rolling_end-rolling_start
    all_train_set_sku_path="../源文件/地市/测试数据结果/第%d轮/data_sku_old.csv"%rolling_end_year
    curve_fit_path_head="..\源文件\地市\销量预测误差统计_test\第%d轮滚动结果\%s\Bass函数(curve_fit)拟合销量结果(老品).csv"
    feature_mapping_path_head="..\源文件\地市\销量预测误差统计_test\第%d轮滚动结果\%s\%s拟合销量结果(预测数据).csv"
    return_path_head="..\源文件\地市\销量预测误差统计_test\各个SKU的预测销量统计\老品"
    feature_mapping_columns=["%d年及以前_特征映射"%i for i in range(rolling_start,rolling_end)]
    curve_fit_columns=["%d年及以前_curve_fit"%i for i in range(rolling_start,rolling_end)]
    columns_all=['original_sale']+feature_mapping_columns+["",""]+curve_fit_columns   #得到当前需要的columns 的名称
    feature_mapping_plus_curve_fit_error_columns=["%d年及以前_训练集误差"%i for i in range(rolling_start,rolling_end)]
    all_train_set_data=pd.read_csv(all_train_set_sku_path,index_col=0 ) #读取原始销量数据


    for model_name in sf.static_class.Machine_Model_name_used:
        return_path=return_path_head+"\%s"%(model_name)
        return_path_sku=return_path
        os.makedirs(return_path,exist_ok=True)
        return_path=return_path+"\误差集合"
        os.makedirs(return_path,exist_ok=True)

        each_year_feature_mapping_error_set_path_head="..\源文件\地市\销量预测误差统计_test\第%d轮滚动结果\%s\老品_历史_销量误差(特征映射).csv"
        each_year_curve_fit_error_set_path_head="..\源文件\地市\销量预测误差统计_test\第%d轮滚动结果\%s\老品_历史_销量误差(curve_fit).csv"

        all_train_dict_data = {} #销量的历史记录
        all_train_dict_error={}  #销量的误差记录

        #特征映射误差 每一年记录
        all_train_error_all_roll_year_Mape=pd.DataFrame(index=all_train_set_data.columns,
                                                             columns=feature_mapping_columns)  #一个表记录一个模型 所有sku的mape误差
        all_train_error_all_roll_year_Smape = pd.DataFrame(index=all_train_set_data.columns,
                                                          columns=feature_mapping_columns)
        all_train_error_all_roll_year_Mae = pd.DataFrame(index=all_train_set_data.columns,
                                                          columns=feature_mapping_columns)
        all_train_error_all_roll_year_Rmse = pd.DataFrame(index=all_train_set_data.columns,
                                                         columns=feature_mapping_columns)

        #curve_fit 误差每一年记录
        all_train_error_all_roll_year_Mape_curve_fit = pd.DataFrame(index=all_train_set_data.columns,
                                                          columns=curve_fit_columns)  # 一个表记录一个模型 所有sku的mape误差
        all_train_error_all_roll_year_Smape_curve_fit = pd.DataFrame(index=all_train_set_data.columns,
                                                           columns=curve_fit_columns)
        all_train_error_all_roll_year_Mae_curve_fit = pd.DataFrame(index=all_train_set_data.columns,
                                                         columns=curve_fit_columns)
        all_train_error_all_roll_year_Rmse_curve_fit = pd.DataFrame(index=all_train_set_data.columns,
                                                         columns=curve_fit_columns)


        #训练集误差 每一年记录

        all_train_error_all_roll_year_Mape_train = pd.DataFrame(index=all_train_set_data.columns,
                                                          columns=feature_mapping_plus_curve_fit_error_columns)  # 一个表记录一个模型 所有sku的mape误差
        all_train_error_all_roll_year_Smape_train = pd.DataFrame(index=all_train_set_data.columns,
                                                           columns=feature_mapping_plus_curve_fit_error_columns)
        all_train_error_all_roll_year_Mae_train = pd.DataFrame(index=all_train_set_data.columns,
                                                         columns=feature_mapping_plus_curve_fit_error_columns)
        all_train_error_all_roll_year_Rmse_train = pd.DataFrame(index=all_train_set_data.columns,
                                                          columns=feature_mapping_plus_curve_fit_error_columns)

        #因为每个模型算出来的数据不同 所有要按每个模型来统计数据
        for sku_name in all_train_set_data.columns:
            sku_data = pd.DataFrame(data={"original_sale": all_train_set_data[sku_name]},
                                    index=all_train_set_data.index,
                                    columns=columns_all)
            all_train_dict_data[sku_name] = sku_data

            sku_error=pd.DataFrame(index=columns_all[1:]+["",""]+feature_mapping_plus_curve_fit_error_columns,
                                   columns=sf.static_class.Errors_columns_name)
            all_train_dict_error[sku_name]=sku_error


        #获得每年滚动的数据
        for roll_year in range(rolling_start,rolling_end):
            roll_num=roll_year-rolling_start+1
            curve_fit_path=curve_fit_path_head%(roll_num,model_name)
            feature_mapping_path=feature_mapping_path_head%(roll_num,model_name,model_name)

            #获取特征映射和curve_fit算出来的数据
            sale_curve_fit_frame=pd.read_csv(curve_fit_path,index_col=0)
            sale_feature_mapping_frame=pd.read_csv(feature_mapping_path,index_col=0)

            each_year_curve_fit_error_set_path=each_year_curve_fit_error_set_path_head%(roll_num,model_name)
            each_year_feature_mapping_error_set_path=each_year_feature_mapping_error_set_path_head%(roll_num,model_name)

            each_year_curve_fit_error_set=pd.read_csv(each_year_curve_fit_error_set_path,index_col=0)
            each_year_feature_mapping_error_set=pd.read_csv(each_year_feature_mapping_error_set_path,index_col=0)

            #遍历每一年的所有sku的误差然后求出均值
            for column in each_year_curve_fit_error_set.columns:
                curve_fit_error=np.mean(each_year_curve_fit_error_set[column])
                feature_mapping_error=np.mean(each_year_feature_mapping_error_set[column])
                each_year_curve_fit_error_set.loc["mean_Error",column]=curve_fit_error
                each_year_feature_mapping_error_set.loc["mean_Error",column]=feature_mapping_error

            each_year_curve_fit_error_set.to_csv(return_path+"\第%d年所有sku误差集合(curve_fit).csv"%roll_num)
            each_year_feature_mapping_error_set.to_csv(return_path+"\第%d年所有sku误差集合(feature_mapping).csv"%roll_num)

            for sku_name,sku_data in all_train_dict_data.items():
                #因为存在当前滚动年份的sku还并没有归为老品 所有使用try_except过滤
                try:

                    #获得对应的sku的数据
                    sale_curve_fit=sale_curve_fit_frame[sku_name].fillna(0)
                    sale_feature_mapping=sale_feature_mapping_frame[sku_name]

                    #将数据放入dataFrame中
                    flag=roll_num-1   #确定下标
                    sku_data[curve_fit_columns[flag]]=sale_curve_fit
                    sku_data[feature_mapping_columns[flag]]=sale_feature_mapping

                    # 获得原始销量
                    original_sale = sku_data['original_sale'].loc[:roll_year]
                    original_sale = original_sale[original_sale > 0]
                    #获得计算误差的数据
                    sale_curve_fit_temp=sale_curve_fit.loc[original_sale.index]
                    # sale_curve_fit_temp=sale_curve_fit_temp[sale_curve_fit_temp>0]
                    sale_feature_mapping_temp=sale_feature_mapping.loc[original_sale.index]
                    # sale_feature_mapping_temp=sale_feature_mapping_temp[sale_feature_mapping_temp>0]


                    error_curve_fit=Model3.get_all_Error_indicator_new(original_sale,sale_curve_fit_temp)
                    error_feature_mapping=Model3.get_all_Error_indicator_new(original_sale,sale_feature_mapping_temp)
                    error_train=Model3.get_all_Error_indicator_new(sale_curve_fit_temp,sale_feature_mapping_temp)

                    all_train_dict_error[sku_name].loc[curve_fit_columns[flag]]=error_curve_fit
                    all_train_dict_error[sku_name].loc[feature_mapping_columns[flag]] = error_feature_mapping
                    all_train_dict_error[sku_name].loc[feature_mapping_plus_curve_fit_error_columns[flag]]=error_train


                    #特征映射误差汇总
                    all_train_error_all_roll_year_Mape.loc[sku_name,feature_mapping_columns[flag]]=\
                        error_feature_mapping.loc["MAPE"]
                    all_train_error_all_roll_year_Smape.loc[sku_name, feature_mapping_columns[flag]] = \
                    error_feature_mapping.loc["SMAPE"]
                    all_train_error_all_roll_year_Mae.loc[sku_name, feature_mapping_columns[flag]] = \
                    error_feature_mapping.loc["MAE"]
                    all_train_error_all_roll_year_Rmse.loc[sku_name, feature_mapping_columns[flag]] = \
                        error_feature_mapping.loc["RMSE"]


                    #原始curve_fit误差汇总
                    all_train_error_all_roll_year_Mape_curve_fit.loc[sku_name, curve_fit_columns[flag]] = \
                        error_curve_fit.loc["MAPE"]
                    all_train_error_all_roll_year_Smape_curve_fit.loc[sku_name, curve_fit_columns[flag]] = \
                        error_curve_fit.loc["SMAPE"]
                    all_train_error_all_roll_year_Mae_curve_fit.loc[sku_name, curve_fit_columns[flag]] = \
                        error_curve_fit.loc["MAE"]
                    all_train_error_all_roll_year_Rmse_curve_fit.loc[sku_name, curve_fit_columns[flag]] = \
                        error_curve_fit.loc["RMSE"]

                    #训练集误差
                    all_train_error_all_roll_year_Mape_train.loc[sku_name,feature_mapping_plus_curve_fit_error_columns[flag]]=\
                        error_train.loc["MAPE"]
                    all_train_error_all_roll_year_Smape_train.loc[
                        sku_name, feature_mapping_plus_curve_fit_error_columns[flag]] = \
                        error_train.loc["SMAPE"]
                    all_train_error_all_roll_year_Mae_train.loc[
                        sku_name, feature_mapping_plus_curve_fit_error_columns[flag]] = \
                        error_train.loc["MAE"]
                    all_train_error_all_roll_year_Rmse_train.loc[
                        sku_name, feature_mapping_plus_curve_fit_error_columns[flag]] = \
                        error_train.loc["RMSE"]




                except:
                    pass

        Error_Mape_Smape_Mae_feature_mapping=dict(zip(["MAPE","SMAPE","MAE","RMSE"],
                                      [all_train_error_all_roll_year_Mape,
                                       all_train_error_all_roll_year_Smape,
                                       all_train_error_all_roll_year_Mae,
                                       all_train_error_all_roll_year_Rmse]))

        Error_Mape_Smape_Mae_curve_fit = dict(zip(["MAPE", "SMAPE", "MAE","RMSE"],
                                                        [all_train_error_all_roll_year_Mape_curve_fit,
                                                         all_train_error_all_roll_year_Smape_curve_fit,
                                                         all_train_error_all_roll_year_Mae_curve_fit,
                                                         all_train_error_all_roll_year_Rmse_curve_fit]))

        Error_Mape_Smape_Mae_train = dict(zip(["MAPE", "SMAPE", "MAE", "RMSE"],
                                                  [all_train_error_all_roll_year_Mape_train,
                                                   all_train_error_all_roll_year_Smape_train,
                                                   all_train_error_all_roll_year_Mae_train,
                                                   all_train_error_all_roll_year_Rmse_train]))

        #计算综合误差的汇总
        for error_indicator,error_all_roll_year in Error_Mape_Smape_Mae_feature_mapping.items():

            for roll_column in feature_mapping_columns:
                Error_all_sku=error_all_roll_year[roll_column]
                Error_all_sku=Error_all_sku[Error_all_sku.notna()]
                roll_res=np.mean(Error_all_sku)
                error_all_roll_year.loc["mean_Error",roll_column]=roll_res
            error_all_roll_year.to_csv(return_path+"\%s.csv"%(model_name+"_训练集每年各SKU_%s误差_feature_mapping"%(error_indicator)))

        for error_indicator, error_all_roll_year in Error_Mape_Smape_Mae_curve_fit.items():

            for roll_column in curve_fit_columns:
                Error_all_sku = error_all_roll_year[roll_column]
                Error_all_sku = Error_all_sku[Error_all_sku.notna()]
                roll_res = np.mean(Error_all_sku)
                error_all_roll_year.loc["mean_Error", roll_column] = roll_res
            error_all_roll_year.to_csv(
                return_path + "\%s.csv" % (model_name + "_训练集每年各SKU_%s误差_curve_fit" % (error_indicator)))

        for error_indicator, error_all_roll_year in Error_Mape_Smape_Mae_train.items():

            for roll_column in feature_mapping_plus_curve_fit_error_columns:
                Error_all_sku = error_all_roll_year[roll_column]
                Error_all_sku = Error_all_sku[Error_all_sku.notna()]
                roll_res = np.mean(Error_all_sku)
                error_all_roll_year.loc["mean_Error", roll_column] = roll_res
            error_all_roll_year.to_csv(
                return_path + "\%s.csv" % (model_name + "_训练集每年各SKU_%s误差_train" % (error_indicator)))

        for sku_name,sku_data in all_train_dict_data.items():
            error_temp=all_train_dict_error[sku_name].reset_index(drop=False)
            sku_data_index=sku_data.index
            sku_data_temp=sku_data.reset_index(drop=False)

            res=pd.concat([sku_data_temp,pd.DataFrame(columns=["",""]),error_temp],axis=1)
            #TODO: 这里的index 设置是有问题的 如果后期 拼接的dataFrame超过了的话 那么就可能数据显示不全
            # res=res.set_index(sku_data_index)
            # sku_data.to_csv(return_path+"\%s.csv"%(sku_name))

            res.to_csv(return_path_sku+"\%s.csv"%(sku_name),index=False)

def get_test_set_Error_feature_mapping_plus_curve_fit(rolling_start,rolling_end):
    print("获得测试集feature_mapping+curve_fit的所有结果")

    rolling_end_year=rolling_end-rolling_start
    feature_mapping_path_head = "..\源文件\地市\销量预测误差统计_test\第%d轮滚动结果\%s\%s拟合销量结果(预测数据).csv"
    curve_fit_path_head = "..\源文件\地市\销量预测误差统计_test\第%d轮滚动结果\%s\Bass函数(curve_fit)拟合销量结果(在销新品).csv"
    feature_mapping_roll_year_error_path="..\源文件\地市\销量预测误差统计_test\第%d轮滚动结果\%s\新品_历史_销量误差(特征映射).csv"
    each_year_curve_fit_error_set_path_head = "..\源文件\地市\销量预测误差统计_test\第%d轮滚动结果\%s\新品_历史_销量误差(curve_fit).csv"
    original_sale_path_head="..\源文件\地市\特征结果\%s.csv"
    original_sku_path_head="..\源文件\地市\测试数据结果\第%d轮\sku_semi_brand_new.csv"
    feature_mapping_columns = ["%d年及以前_特征映射" % i for i in range(rolling_start, rolling_end)]
    curve_fit_columns = ["%d年及以前_curve_fit" % i for i in range(rolling_start, rolling_end)]
    feature_mapping_plus_curve_fit_error_columns = ["%d年及以前_测试集误差" % i for i in
                                                    range(rolling_start, rolling_end)]

    all_columns=["original_sale"]+feature_mapping_columns+["",""]+curve_fit_columns
    return_path_head="..\源文件\地市\销量预测误差统计_test\各个SKU的预测销量统计\新品\新品测试集误差统计"
    os.makedirs(return_path_head,exist_ok=True)
    for model_name in sf.static_class.Machine_Model_name_used:
        all_test_dict_sale={}
        all_test_dict_error={}

        #特征映射误差记录
        error_Mape=pd.DataFrame(columns=feature_mapping_columns)
        error_Smape = pd.DataFrame(columns=feature_mapping_columns)
        error_Mae = pd.DataFrame(columns=feature_mapping_columns)
        error_Rmse = pd.DataFrame(columns=feature_mapping_columns)
        # curve_fit 误差每一年记录
        all_test_error_all_roll_year_Mape_curve_fit = pd.DataFrame(columns=curve_fit_columns)  # 一个表记录一个模型 所有sku的mape误差
        all_test_error_all_roll_year_Smape_curve_fit = pd.DataFrame(columns=curve_fit_columns)
        all_test_error_all_roll_year_Mae_curve_fit = pd.DataFrame(columns=curve_fit_columns)
        all_test_error_all_roll_year_Rmse_curve_fit = pd.DataFrame(columns=curve_fit_columns)

        # 训练集误差 每一年记录

        all_test_error_all_roll_year_Mape_test = pd.DataFrame( columns=feature_mapping_plus_curve_fit_error_columns)  # 一个表记录一个模型 所有sku的mape误差
        all_test_error_all_roll_year_Smape_test = pd.DataFrame(  columns=feature_mapping_plus_curve_fit_error_columns)
        all_test_error_all_roll_year_Mae_test = pd.DataFrame(columns=feature_mapping_plus_curve_fit_error_columns)
        all_test_error_all_roll_year_Rmse_test = pd.DataFrame( columns=feature_mapping_plus_curve_fit_error_columns)

        return_path = return_path_head + "\%s" % (model_name)
        return_path_error_set = return_path + "\误差集合"
        os.makedirs(return_path_error_set, exist_ok=True)

        for roll_year in range(rolling_start,rolling_end):  #17-21
            roll_num=roll_year-rolling_start+1  #获得滚动年份数据


            feature_mapping_path=feature_mapping_path_head%(roll_num,model_name,model_name)
            each_year_curve_fit_error_set_path=each_year_curve_fit_error_set_path_head%(roll_num,model_name)

            curve_fit_path=curve_fit_path_head%(roll_num,model_name,)
            #获得预测销量数据
            predict_sale=pd.read_csv(feature_mapping_path,index_col=0)  #特征映射的
            #当期滚动原始销量
            original_semi_sku=pd.read_csv(original_sku_path_head%(roll_num),index_col=0).columns.to_list()
            curve_fit_sale=pd.read_csv(curve_fit_path,index_col=0)
            # curve_fit_sale=pd.read_csv(curve_fit_path,index_col=0)   #curve_fit 的

            roll_year_error=pd.read_csv(feature_mapping_roll_year_error_path%(roll_num,model_name),index_col=0)
            roll_year_error_curve_fit=pd.read_csv(each_year_curve_fit_error_set_path,index_col=0)
            for column in sf.static_class.Errors_columns_name:
                roll_year_error.loc["mean_Error",column]=np.mean(roll_year_error[column])
                roll_year_error_curve_fit.loc["mean_Error",column]=np.mean(roll_year_error_curve_fit[column])

            roll_year_error.to_csv(return_path_error_set+"\第%d年所有sku误差集合(feature_mapping).csv"%roll_num)
            roll_year_error_curve_fit.to_csv(return_path_error_set+"\第%d年所有sku误差集合(curve_fit).csv"%roll_num)


            for sku_name in original_semi_sku:  #遍历每一个sku_name
                if sku_name not in all_test_dict_sale.keys():  #如果sku 不存在字典里面 那么创建该sku的dataFrame
                    original_sale=pd.read_csv(original_sale_path_head%(sku_name),
                                              index_col=0)["sale"].rename("original_sale")

                    feature_mapping_roll_sale=pd.DataFrame(data=original_sale,index=original_sale.index,columns=all_columns)
                    feature_mapping_roll_error=pd.DataFrame(index=all_columns[1:]+["",""]+feature_mapping_plus_curve_fit_error_columns,columns=sf.static_class.Errors_columns_name)
                    all_test_dict_sale[sku_name]=feature_mapping_roll_sale
                    all_test_dict_error[sku_name]=feature_mapping_roll_error

                feature_mapping_roll_sale=all_test_dict_sale[sku_name]
                feature_mapping_roll_error=all_test_dict_error[sku_name]
                flag=roll_num-1

                sku_roll_predict_sale=predict_sale[sku_name]  #获取sku的预测销量
                # sku_roll_predict_sale_curve_fit=curve_fit_sale[sku_name]

                feature_mapping_roll_sale[feature_mapping_columns[flag]]=sku_roll_predict_sale

                sku_roll_predict_sale=sku_roll_predict_sale.loc[:roll_year]
                sku_roll_original_sale=feature_mapping_roll_sale["original_sale"].loc[:roll_year]
                sku_roll_original_sale=sku_roll_original_sale[sku_roll_original_sale>=5]
                sku_roll_predict_sale=sku_roll_predict_sale[sku_roll_original_sale.index]

                sku_roll_error=Model3.get_all_Error_indicator_new(sku_roll_original_sale,sku_roll_predict_sale)

                feature_mapping_roll_error.loc[feature_mapping_columns[flag]]=sku_roll_error

                error_Mape.loc[sku_name,feature_mapping_columns[flag]]=sku_roll_error["MAPE"]
                error_Smape.loc[sku_name, feature_mapping_columns[flag]] = sku_roll_error["SMAPE"]
                error_Mae.loc[sku_name, feature_mapping_columns[flag]] = sku_roll_error["MAE"]
                error_Rmse.loc[sku_name, feature_mapping_columns[flag]] = sku_roll_error["RMSE"]

                try:
                    #当前的测试集新品 销量数据不够 还没有curve_fit的结果 所以需要try except guolv
                    sku_roll_predict_sale_curve_fit=curve_fit_sale[sku_name] #获得curve_fit中的sku 销量数据
                    feature_mapping_roll_sale[curve_fit_columns[flag]]=sku_roll_predict_sale_curve_fit

                    sku_roll_predict_sale_curve_fit=sku_roll_predict_sale_curve_fit.loc[sku_roll_original_sale.index]
                    sku_roll_error_curve_fit=Model3.get_all_Error_indicator_new(sku_roll_original_sale,sku_roll_predict_sale_curve_fit)
                    sku_roll_error_train=Model3.get_all_Error_indicator_new(sku_roll_predict_sale_curve_fit,sku_roll_predict_sale)

                    feature_mapping_roll_error.loc[curve_fit_columns[flag]] =sku_roll_error_curve_fit
                    feature_mapping_roll_error.loc[feature_mapping_plus_curve_fit_error_columns[flag]]=sku_roll_error_train


                    all_test_error_all_roll_year_Mape_curve_fit.loc[sku_name, curve_fit_columns[flag]] = \
                        sku_roll_error_curve_fit.loc["MAPE"]
                    all_test_error_all_roll_year_Smape_curve_fit.loc[sku_name, curve_fit_columns[flag]] = \
                        sku_roll_error_curve_fit.loc["SMAPE"]
                    all_test_error_all_roll_year_Mae_curve_fit.loc[sku_name, curve_fit_columns[flag]] = \
                        sku_roll_error_curve_fit.loc["MAE"]
                    all_test_error_all_roll_year_Rmse_curve_fit.loc[sku_name, curve_fit_columns[flag]] = \
                        sku_roll_error_curve_fit.loc["RMSE"]

                    # 训练集误差 每一年记录

                    all_test_error_all_roll_year_Mape_test.loc[sku_name,feature_mapping_plus_curve_fit_error_columns[flag]]=\
                        sku_roll_error_train["MAPE"]# 一个表记录一个模型 所有sku的mape误差

                    all_test_error_all_roll_year_Smape_test.loc[sku_name,feature_mapping_plus_curve_fit_error_columns[flag]]=\
                    sku_roll_error_train["SMAPE"]

                    all_test_error_all_roll_year_Mae_test.loc[sku_name,feature_mapping_plus_curve_fit_error_columns[flag]]=\
                        sku_roll_error_train["MAE"]

                    all_test_error_all_roll_year_Rmse_test.loc[sku_name,feature_mapping_plus_curve_fit_error_columns[flag]]=\
                    sku_roll_error_train["RMSE"]


                except:
                    pass




        all_error=dict(zip(["MAPE","SMAPE","MAE","RMSE"],
                 [error_Mape,error_Smape,error_Mae,error_Rmse]))
        all_error_curve_fit = dict(zip(["MAPE", "SMAPE", "MAE", "RMSE"],
                              [all_test_error_all_roll_year_Mape_curve_fit,
                               all_test_error_all_roll_year_Smape_curve_fit,
                               all_test_error_all_roll_year_Mae_curve_fit,
                               all_test_error_all_roll_year_Rmse_curve_fit]))

        all_error_test=dict(zip(["MAPE", "SMAPE", "MAE", "RMSE"],
                              [all_test_error_all_roll_year_Mape_test,
                               all_test_error_all_roll_year_Smape_test,
                               all_test_error_all_roll_year_Mae_test,
                               all_test_error_all_roll_year_Rmse_test]))
        for error_indicator,error_temp in all_error.items():
            for roll_column in error_temp.columns:
                    error_temp.loc["mean_Error",roll_column]=np.mean(error_temp[roll_column])
            error_temp.to_csv(return_path_error_set+"\%s_测试集每年各SKU_%s误差_feature_mapping.csv"%(model_name,error_indicator))

        for error_indicator, error_temp in all_error_curve_fit.items():
            for roll_column in error_temp.columns:
                error_temp.loc["mean_Error", roll_column] = np.mean(error_temp[roll_column])
            error_temp.to_csv(return_path_error_set + "\%s_测试集每年各SKU_%s误差_curve_fit.csv" % (
            model_name, error_indicator))
        for error_indicator, error_temp in all_error_test.items():
            for roll_column in error_temp.columns:
                error_temp.loc["mean_Error", roll_column] = np.mean(error_temp[roll_column])
            error_temp.to_csv(return_path_error_set + "\%s_测试集每年各SKU_%s误差_test.csv" % (
            model_name, error_indicator))

        os.makedirs(return_path,exist_ok=True)
        return_path_temp=return_path+"\%s.csv"
        for sku_name,sku_data in all_test_dict_sale.items():
            sku_data_index=sku_data.index
            error_temp=all_test_dict_error[sku_name].reset_index(drop=False)
            sku_data_temp=sku_data.reset_index(drop=False)
            res=pd.concat([sku_data_temp,pd.DataFrame(columns=["",""]),error_temp],axis=1)
            # res=res.set_index(sku_data_index)

            res.to_csv(return_path_temp%(sku_name),index=False)
        '''
            需要注意一下 测试集中的sku数量与最终的新品误差计算的sku数量不同
            有两个地方：
            1.测试集不包括第五轮滚动的14个完全新品
            2.测试集包括了第一轮滚动中在销新品18年没有销量的8个sku 
            
            所以最终的测试集sku数量为133
            新品预测的sku数量为139=133+14-8
            
        '''