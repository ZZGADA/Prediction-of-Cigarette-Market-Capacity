#特征映射
import os

from scipy.optimize import leastsq, curve_fit
import sympy as sp
import numpy as np  # 科学计算库
import matplotlib.pyplot as plt  # 绘图库
import pandas as pd
from math import e

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.model_selection import train_test_split  #训练集和测试集分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing as prep
from Dao import Data_Acquisition as Da
import sys
sys.path.append(r"../")  #返回上级目录
from Service import Bass_Model as BM
from copy import deepcopy
from boruta import BorutaPy
from sklearn.preprocessing import OneHotEncoder
from Dao import static_file as sf
key_feature=["prices_type","package_type","cigarette_type","nicotine_content","CO_content","filter_length","cigarette_circumference","wholesale_prices"]
feature_need_feature_scaling=["filter_length","cigarette_circumference","wholesale_prices"]
region_names=dict(zip(['苏州', '徐州', '宿迁', '无锡', '淮安', '南京', '常州', '扬州', '连云港', '南通', '镇江', '泰州', '盐城'],range(1,14)))

#将离散数值 设置为虚拟变量
def Discrete_data_processing(feature:pd.DataFrame,sku_all:pd.DataFrame):
    sku_name=[]
    sku_final=["year"]
    feature_sku=feature.index.to_list()
    for name in sku_all.iloc[:,1:].columns:
        if name in feature_sku:
            sku_name.append(name)
    feature_has = feature.loc[sku_name, key_feature]
    sku_final.extend(sku_name)
    sku_all=sku_all.loc[:,sku_final]
    category_features = ["prices_type", "package_type"]
    sku_feature= pd.get_dummies(feature_has, drop_first=False,
                                 columns=category_features)  # 生成dummy变量 7个特征 变成了12个7-2+5+2
    return sku_feature,sku_all


#将离散数值 设置为虚拟变量 region 的 注意地市的名字处理
def Discrete_data_processing_region(feature:pd.DataFrame,sku_all:pd.DataFrame,roll_num,end_year):
    sku_name=[]  #地市+品规
    sku_name_strip=[]  #品规
    sku_final=["year"]
    append_Macro_index=sf.static_class.append_Macro_index

    region_type_return_path="../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/相关数据"%(roll_num)
    Macro_data_by_region=Da.get_Data("../源文件/地市/江苏各市宏观数据.xlsx",0)
    # append_feature=Da.get_Data('../源文件/地市/江苏省可用指标.xlsx',0).loc[append_Macro_index,end_year]

    feature_sku=feature.index.to_list()
    for name in sku_all.iloc[:,1:].columns:
        name_strip=name.split("_")[-1]  #品规名字  南京_利群(硬)
        if name_strip in feature_sku:
            sku_name.append(name)
            sku_name_strip.append(name_strip)

    sku_name_strip=set(sku_name_strip)  #列表去重 转换为set就好
    feature_has = feature.loc[sku_name_strip, key_feature]  #提取特征
    sku_final.extend(sku_name)  #将sku_final扩充
    sku_all_res=sku_all.loc[:,sku_final] #提取出有特征的地市+品规组合
    category_features = ["prices_type", "package_type",'cigarette_type']
    sku_feature= pd.get_dummies(feature_has, drop_first=False,
                                 columns=category_features)  # 生成dummy变量 最终一共15个特征

    sku_feature_extend=pd.DataFrame(columns=sku_feature.columns,index=sku_name,dtype=float)  #dataFrame大小为 375*12  原可用品规为393个
    sku_feature_region_type=pd.DataFrame(columns=["region_type"],index=sku_name)
    for name in sku_name:
        region_name_temp,name_strip=name.split("_")  #品规名字  南京_利群(硬)
        if name_strip in sku_feature.index:   #如果去掉地市名字后的品规在sku_feature中那么就加入到sku_feature_extend中
            sku_feature_extend.loc[name]=sku_feature.loc[name_strip]
            sku_feature_extend.loc[name,'percentage_of_population_%d'%end_year]=\
                Macro_data_by_region.loc[region_name_temp,"percentage_of_population_%d"%end_year]
            sku_feature_extend.loc[name, 'total_population_%d' % end_year] =\
                Macro_data_by_region.loc[region_name_temp, "total_population_%d" % end_year]
            # sku_feature_extend.loc[name,append_Macro_index]=append_feature


            sku_feature_region_type.loc[name,"region_type"]=region_names[region_name_temp]
    # sku_feature_extend["region_type"]=sku_feature_extend["region_type"].astype('int')  #将float转换为int值

    sku_feature_region_type=pd.get_dummies(sku_feature_region_type,drop_first=False,columns=["region_type"])#将region type转换为dummy变量
    os.makedirs(region_type_return_path,exist_ok=True)
    sku_feature_region_type.to_csv(region_type_return_path+"/地市特征分类.csv")

    return sku_feature_extend,sku_all_res

def supplementary_Feature_Matching(sku_feature_all,feature_supplementary):
    #将补充的特征的原来的特征放在一起并匹配
    final=sku_feature_all.join(feature_supplementary,how="inner")
    final.to_csv("../源文件/地市/最终补充后的特征集合.csv")
    return final  #返回左连接后的

def special_processing(self_attributes_all:pd.DataFrame,brand_new_sku):
    #对于完全新品的品规 因为最开始阈值处理的问题 所以新品特征存在偏误 我们需要调整一下
    print("特征调整")
    feature_need_processing=[]
    feature:str
    for feature in self_attributes_all.columns:
        if feature.__contains__("previous_year"):
            feature_need_processing.append(feature)
    size=len(feature_need_processing)
    for sku_name in brand_new_sku:
        self_attributes_all.loc[sku_name,feature_need_processing]=np.zeros(size)

    return self_attributes_all


#获取关键特征

#这个是 省份用的
def get_key_feature(feature:pd.DataFrame,params:dict):
    #自行分析后 相关性分析后 提取关键特征
    key_feature=["prices_type","package_type","nicotine_content","CO_content","filter_length","cigarette_circumference","wholesale_prices"]
    feature_need_feature_scaling=["filter_length","cigarette_circumference","wholesale_prices"]
    print("it is getting key feature")
    sku_params:pd.DataFrame
    sku_name_all_in_feature=feature.index.to_list()
    sku_feature_all={}
    params_all={}
    params_name_all=[]
    for classify_name,sku_params in params.items():
        temp=[i for i in sku_params.index if  i in sku_name_all_in_feature]
        params_all[classify_name]=temp
        params_name_all=params_name_all+temp

    # 虚拟处理 区分离散数据和连续数据 保证离散的数据在高维空间中 欧氏距离都是相等的取分类的作用
    feature_has=feature.loc[params_name_all,key_feature]
    category_features = ["prices_type", "package_type"]
    sku_feature = pd.get_dummies(feature_has, drop_first=False,
                                          columns=category_features)  # 生成dummy变量 7个特征 变成了12个7-2+5+2


    #特征值分类
    for sku_classification_name,sku_params in params.items():
        sku_classify_feature=sku_feature.loc[params_all[sku_classification_name]]  #属性特征
        sku_classify_params=sku_params.loc[params_all[sku_classification_name]]   #拟合的参数结果 m,p,q
        sku_feature_all[sku_classification_name]=sku_classify_feature
        params_all[sku_classification_name]=sku_classify_params


        # print(sku_classify_feature)

        # sku_classify_feature.corr().to_csv("D:\非常有用的大学资料\中国烟草\测试数据\测试结果\Bass_Test\%s.csv"%(sku_classification_name))
        # sku_classify_feature['prices_type']=sku_classify_feature['prices_type'].astype()
        #随机森林+boruta
        # for i in ['m','p','q']:
        #     rfr=RandomForestRegressor(n_estimators=200,criterion='mse',max_depth=4)
        #     boruta=BorutaPy(rfr,n_estimators='auto',verbose=2,alpha=0.13)
        #     t1=boruta.fit(np.array(sku_classify_feature), sku_classify_params[i])
        #
        #     green_area = sku_classify_feature.columns[boruta.support_].to_list()
        #     blue_area = sku_classify_feature.columns[boruta.support_weak_].to_list()
        #     print(i)
        #     print('features in the green area:', green_area)
        #     print('features in the blue area:', blue_area)


        # boruta.fit()

    # sku_feature_all_copy=deepcopy(sku_feature_all)  #深拷贝 拷贝后对象内部子对象修改之后 不会对原对象的父对象产生影响

    return sku_feature_all,params_all

def sales_calculation(sales_previous:pd.Series):
    #截至今年为止的累计销量计算
    return np.sum(sales_previous)


def get_region_supplementary_feature_other1(sku_name,previous_year,columns_index,ifBrandNew=False):
    #该函数用于提取卷烟生命周期前三年的上面那些指标 PFprice_CV_previous_year这些
    #previous_year表示品规生命周期前三年的年份列表

    path="../源文件/地市/特征结果/%s.csv"%(sku_name)
    sku_data_temp=pd.read_csv(path,index_col=0).fillna(0)
    temp_feature:pd.Series
    columns_set=["sale","PFprice_SD","PFprice_CV","LSprice_average","LSprice_SD","LSprice_CV"]
    feature_need=[]#需要提取出来的特征

    for year in previous_year:
        if year ==-1:
            sku_feature=np.zeros(len(columns_set))
        else:
            sku_feature=sku_data_temp.loc[year,columns_set].to_list()
        feature_need.extend(sku_feature)

    feature_supplementary=pd.Series(data=feature_need,index=columns_index,name=sku_name)
    return feature_supplementary

def get_region_supplementary_feature_other2(sku_name,columns_index,sku_sale:pd.Series,t:pd.Series,ifBrandNew=False):
    '''
    获取当前生命周期内 累计销量 销量最大值时候的销量和生命周期时间 销量增速最大值的销量增速、销量、以及生命周期时间
    以及各个时间点所占生命周期的比例
    columns_index_other2_max=['total_sales','sale_max_t','sale_max',
    'growth_rate_max_t','growth_rate_max_sale','sale_max_t_percentage','grow_rate_max_t_percentage']
    '''
    # print("get supplementary_feature_other2")
    columns_index_other2_max_final_test = ['total_sales']

    feature_supplement=pd.DataFrame(index=columns_index,columns=[sku_name])
    if ifBrandNew:
        feature_supplement[sku_name]=np.zeros(len(columns_index))   #如果是新品 那么全部为0
    else:
        sku_feature_data=pd.DataFrame(data={"t":t.values,"sale":sku_sale.values},index=sku_sale.index)
        growth_rate_temp=np.append(np.nan,np.diff(sku_sale))
        sku_feature_data["growth_rate"]=growth_rate_temp    #两两做差 得到斜率
        sku_feature_data=sku_feature_data[sku_feature_data['t'].notna()]
        feature_supplement.loc['total_sales']=np.sum(sku_sale)  #获取总的销量特征
        index_sale_max=['sale_max_t','sale_max','sale_max_t_percentage']
        feature_supplement.loc[index_sale_max,sku_name]=\
            find_sale_max(sku_name,sku_feature_data,index_sale_max)
        index_rate_max=[ 'growth_rate_max_t','growth_rate_max_sale','growth_rate_max'
            ,'grow_rate_max_t_percentage'
            ,'growth_rate_max_t_percentage_in_sale_max']

        feature_supplement.loc[index_rate_max,sku_name]=\
            find_growth_rate_max(sku_name,sku_feature_data,index_rate_max,feature_supplement.loc['sale_max_t',sku_name])

    return feature_supplement.loc[columns_index_other2_max_final_test,sku_name]

def get_region_supplementary_feature_other3(sku_name,columns_index,sku_sale:pd.Series,t:pd.Series,front_three_size,params_old=None):
    '''
        "该函数是为了补全老品2006年之前的销量，从而获得前三你的平均销量和平均销量增速"
    '''
    # print("前值补全")
    feature_supplement:pd.Series=None
    As=None
    Asgr=None
    sku_sale_final=sku_sale.iloc[:front_three_size]#提取前三年的销量数据
    sku_sale_final=sku_sale_final[sku_sale_final>0]  #保证前三年的销量数据没有0值
    year_index = sku_sale_final.index.to_list()
    year_gap = year_index[-1] - year_index[0]



    if params_old is None: #表明是新品  新品不需要前值补全
        As = np.mean(sku_sale_final)
        sku_sale_final = sku_sale_final[sku_sale_final > 0]

        if year_gap==0:
            Asgr=sku_sale_final.iloc[0]/2
        else:
            Asgr = (sku_sale_final.iloc[-1] - sku_sale_final.iloc[0]) / year_gap

        feature_supplement=pd.Series(name=sku_name,index=columns_index,data=[As,Asgr])
    else: #表示为老品

        # first_year=sku_sale_final.index.to_list()[0]
        # if first_year==2006:
        #     # print("需要前值补全")
        #     year_count=first_year-1
        #     original_sales=pd.DataFrame(data={"sale":sku_sale_final,'t':t.loc[sku_sale_final.index]},index=sku_sale_final.index)
        #     pass_year_sale=pd.DataFrame(columns=['sale','t'])
        #     temp_t=0
        #     m,p,q=params_old.loc[sku_name,['m','p','q']]  #获得m,p,q的值
        #     sale_new = BM.func(temp_t, m, p, q)
        #     while sale_new>6.7:  #设置阈值
        #         pass_year_sale.loc[year_count]=[sale_new,temp_t]
        #         sale_new = BM.func(temp_t, m, p, q)   #前补销量
        #         year_count-=1
        #         temp_t-=1
        #     index_reverse=pass_year_sale.index.to_list()
        #     index_reverse.reverse()  #index 转置
        #     pass_year_sale=pass_year_sale.reindex(index_reverse)
        #     all_sale=pd.concat([pass_year_sale,original_sales],axis=0)   #所有年份数据合并
        #     year_size=all_sale.shape[0]
        #     if year_size>=3:
        #         As=np.mean(all_sale["sale"].iloc[:3])
        #         Asgr=(all_sale['sale'].iloc[2]-all_sale['sale'].iloc[0])/2
        #     else:
        #         As=np.mean(all_sale["sale"].iloc[:year_size])
        #         Asgr=(all_sale["sale"].iloc[-1]-all_sale["sale"].iloc[0])/(1 if year_size==1 else year_size-1)
        #         if year_size==1:
        #             Asgr=all_sale["sale"].iloc[0]
        #
        #     feature_supplement = pd.Series(name=sku_name, index=columns_index, data=[As, Asgr])
        # else: #否则不需要前值补全
            As = np.mean(sku_sale_final)
            if year_gap==0:
                Asgr = sku_sale_final.iloc[0]/2
            else:
                Asgr = (sku_sale_final.iloc[-1] - sku_sale_final.iloc[0]) / year_gap

            feature_supplement=pd.Series(name=sku_name, index=columns_index, data=[As, Asgr])


    return feature_supplement


def find_sale_max(sku_name,sku_feature_data:pd.DataFrame,index_max)->pd.Series:
    #获取和最大销量有关的特征

    max_sale_index=pd.Series.argmax(sku_feature_data['sale'])  #提取销量最大值的索引
    sale_max_temp:list=sku_feature_data.iloc[max_sale_index,[0,1]].to_list()
    sale_max_temp.append(sale_max_temp[0]/sku_feature_data.iloc[-1,0])  #最大销量占全生命周期的百分比
    return pd.Series(index=index_max,data=sale_max_temp,name=sku_name)

def find_growth_rate_max(sku_name,sku_feature_data:pd.DataFrame,index_max,sale_max_t)->pd.Series:

    max_growth_rate_index = pd.Series.argmax(sku_feature_data['growth_rate'])
    if max_growth_rate_index==-1:   #则表明只有一个数据
        growth_rate_max_temp=sku_feature_data.iloc[0,[0,1]].to_list()
        growth_rate_max_temp.extend([growth_rate_max_temp[1],1,1])
        return pd.Series(index=index_max,
                         data=growth_rate_max_temp, name=sku_name)
    else:
        growth_rate_max_temp:list = sku_feature_data.iloc[max_growth_rate_index, [0, 1,2]].to_list()
        growth_rate_max_temp.append(growth_rate_max_temp[0]/sku_feature_data.iloc[-1,0]) #最大销量增速占全生命周期的百分比
        growth_rate_max_temp.append(growth_rate_max_temp[0]/sale_max_t)  #最大销量增速的生命周期占销量增长生命周期的百分比


        return pd.Series(index=index_max,
                         data=growth_rate_max_temp,name=sku_name)

def get_region_supplementary_feature_other4(new_feature_old:pd.DataFrame, new_feature_semi:pd.DataFrame,
                                            new_feature_brand:pd.DataFrame,self_attributes_all:pd.DataFrame,
                                            new_feature_all,
                                            roll_num):
    '''
    使用KNN回归将新品残缺的特征补全"
    '''
    print("使用KNN回归将新品残缺的特征补全")

    region_feature_type_path="../源文件/地市/销量预测误差统计_test/第%d轮滚动结果/相关数据/地市特征分类.csv"%(roll_num)
    region_feature_type=pd.read_csv(region_feature_type_path,index_col=0)
    # n_neighbors = [k for k in range(1, 8)]
    # weight = ['uniform', 'distance']
    # search_space = {
    #     'n_neighbors': n_neighbors,
    #     'weights': weight,
    #     'p': [p for p in range(1, 8)]
    # }

    n_estimators = [i for i in range(100, 501, 20)]
    xrange = np.logspace(1e-10, 0, 20)
    search_space = {
        'n_estimators': n_estimators,
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [6, 7, 8, 9, 10],
        'verbosity': [0]

    }

    feature_x_all=pd.concat([self_attributes_all,region_feature_type],axis=1,join="inner")

    for i in range(1,3):
        start_flag=0 if i==1 else 4
        end_flag=4 if i==1 else 8
        #获得老品和在销新品中的有前一年的特征数据 设为y
        old_train_y_set:pd.DataFrame=new_feature_old[(new_feature_old['sale_%d'%i]>0 )& (new_feature_old['LSprice_average_%d'%i]>0)].loc[:,sf.static_class.KNN_feature_can_predict[start_flag:end_flag]]
        semi_train_y_set:pd.DataFrame=new_feature_semi[(new_feature_semi["sale_%d"%i]>0)&(new_feature_semi['LSprice_average_%d'%i]>0)].loc[:,sf.static_class.KNN_feature_can_predict[start_flag:end_flag]]



        #提取品规的自我属性特征 设为x
        old_train_x_set:pd.DataFrame=self_attributes_all.loc[old_train_y_set.index]
        semi_train_x_set:pd.DataFrame=self_attributes_all.loc[semi_train_y_set.index]
        train_index=old_train_y_set.index.to_list()+semi_train_y_set.index.to_list()


        #特征y 和属性x各自合并 得到训练姐数据
        # train_x_set=pd.concat([old_train_x_set,semi_train_x_set],axis=0).join(region_feature_type,how="inner")
        train_x_set=feature_x_all.loc[train_index]
        train_y_set=pd.concat([old_train_y_set,semi_train_y_set],axis=0)

        train_x_set_semi=feature_x_all.loc[new_feature_semi.index]
        train_y_set_semi=new_feature_semi.loc[:,['As']] #'sale_1',"sale_2"


        for param in train_y_set.columns:  #遍历训练集 获得要补充的特征
            search = GridSearchCV(xgb.XGBRegressor(), search_space, cv=4, n_jobs=-1)
            search.fit(train_x_set.values,train_y_set[param].values)   #注意如果是xgboost的话 记得添加.value 变成完全的二维矩阵
            for sku in new_feature_all.index:
                if new_feature_all.loc[sku,param]==0:
                    # print(feature_x_all.loc[sku].to_list())
                    y_test=search.predict([feature_x_all.loc[sku].to_list()])
                    # print(y_test)
                    new_feature_all.loc[sku,param]=y_test[0]
        for param in train_y_set_semi.columns:
            search=GridSearchCV(xgb.XGBRegressor(), search_space, cv=4, n_jobs=-1)
            search.fit(train_x_set_semi.values,train_y_set_semi[param].values)
            for sku in new_feature_brand.index:
                y_test:list=search.predict([feature_x_all.loc[sku].to_list()])
                new_feature_all.loc[sku,param]=y_test[0]  #通过在销新品补全 完全新品的特征

    # print(new_feature_all.loc[new_feature_brand.index])
    # input("使用KNN回归将新品残缺的特征补全")
    return new_feature_all


def get_train_set_columns(new_feature_old_columns:list):
    target="sale_1"
    train_set_columns=[]
    for i in new_feature_old_columns:
        if i==target:
            return train_set_columns
        else:
            train_set_columns.append(i)



    input("KNN 回归特征补全input")
