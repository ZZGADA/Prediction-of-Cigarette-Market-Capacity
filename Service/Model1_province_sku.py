#省份+sku
import math
import random

from scipy.optimize import leastsq, curve_fit
import sympy as sp
import numpy as np  # 科学计算库
import matplotlib.pyplot as plt  # 绘图库
import pandas as pd
from math import e


from sklearn.model_selection import train_test_split  #训练集和测试集分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score
from sklearn import preprocessing as prep
import sys
sys.path.append(r"../")  #返回上级目录
from Service import Bass_Model as BM
from DataAnalysis import Projected_evaluation_indicators as pei
from copy import deepcopy
from sklearn import svm
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

now_year=2023
Bass_params = ['m', 'p', 'q']
columns_name=['Accuracy', 'MSE', 'MAE', 'RMSE','MAPE','SMAPE']
index_name=['MLR','SVR','RT']

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
    semi_flag_year_index = end_year_index - 4  # 非完全新品的标志下标
    sku_semi_brand_new=pd.DataFrame(data={"year":[i for i in  range(end_year-4,last_year+1)]})  #非完全新品数据 在end年之前有连续的小于等于5年的销量

    for sku in Data.columns.to_list()[1:]:

        data_temp:pd.Series
        data_sku=Data[sku][start_year_index:end_year_index_plus1]
        data_temp=np.trim_zeros(data_sku)

        if data_temp.size==0 :#如果首尾去0 之后没有数据则表明该sku的销量 在当前阶段没有意义和价值
            if Data[sku][end_year_index_plus1]!=0:  #下一年有数据 则表示该sku为下一年的完全新品
                sku_brand_new[sku]=Data[sku][end_year_index_plus1:].to_list() #需要新品一直到2022年的数据 判断模型的拟合效果
            continue
        elif data_temp.index.to_list()[0]>=semi_flag_year_index:  #则表示为非完全新品数据  !!!
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
            four_FenWeiDian=round((end_year-start_year+1)/4)
            if Data[sku][end_year_index_plus1]==0 and data_temp_all<3 :
                #小于巴斯模型参数个数 而且不属于完全新品 则直接跳过 该sku 没有价值
                continue
            elif Data[sku][end_year_index_plus1]!=0 and data_temp_all<=four_FenWeiDian:
                if data_temp2>=data_temp1/four_FenWeiDian:
                    sku_semi_brand_new[sku] = Data[sku][semi_flag_year_index:].to_list()
            else:
                data_sku_old[sku] = Data[sku].to_list()


    # print(sku_brand_new)
    sku_all={"data_sku_old":data_sku_old,"sku_brand_new":sku_brand_new,"sku_semi_brand_new":sku_semi_brand_new}
    return sku_all
# data_sku_old:pd.DataFrame,sku_brand_new:pd.DataFrame,sku_semi_brand_new:pd.DataFrame
#每sku_all中每一个dataFrame 的index 都是从0开始的
def get_Bass_params(sku_all:dict,start_year:int,end_year:int):
    # print("it is getting Bass params")
    params_old=pd.DataFrame(columns=Bass_params)
    params_brand_new=pd.DataFrame(columns=Bass_params)
    params_semi_brand_new=pd.DataFrame(columns=Bass_params)
    params_all={"data_sku_old":params_old,"sku_brand_new":params_brand_new,"sku_semi_brand_new":params_semi_brand_new}
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

            # if sku=="雄狮(红老版)":
            #     print("iiiiiiiiiiiii",sku_classification_name)
            if temp.size<3 or temp.size<round((end_year-start_year)/4):
                # 如果样本数量小于3 表示该样本没有价值 无法进行拟合
                #  也就是说 在时段内 没有销量数据了 但是 end_year之后还有数据 所以要将该品规放入其他其他品规分类结果中
                # 可能为 brand new

                # temp=np.trim_zeros(sku_data[sku][end_year_index+1:])
                # if temp.index.to_list[0]>=semi_flag_year_index:
                #     sku_classification_name="sku_semi_brand_new"
                    #如果重新切片 的数据起始点大于等于部分新品起始点则表示为部分新品
                # print("ahahh",sku,temp,sku_classification_name)
                continue
            if sku_classification_name=="sku_brand_new":
                temp=np.trim_zeros(sku_data[sku])
            t=np.add(temp.index.to_list(),1)  #获取时间序列
            try:
                params,pcov=curve_fit(BM.func,t,temp,p0=[temp.sum(),5,10],maxfev=10000)  #样本数必须大于参数的个数 所以如果样本数 小于3 则该样本没有意义
                '''pcov 返回参数的协方差矩阵 
                参数的协方差矩阵（pcov）是一个二维数组，表示参数的不确定性。
                对角线上的元素是每个参数的方差，非对角线上的元素是参数之间的协方差。
                协方差矩阵可以用来估计参数的置信区间和相关性。
                '''
                params_all[sku_classification_name].loc[sku]=params
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
def Self_attributing_features(roll_num,params_all:dict,self_attributes:dict,sku_sales_data_all:dict,start_year,end_year):
    # print("it is Self-attributing features")
    #end_year 为滚动中时间范围的终止点 end_year+1为新品预测的起始年份   end_year==2022 那么就完全没有新的销量数据了

    sku_params:pd.DataFrame
    sku_feature:pd.DataFrame
    sales_old:pd.DataFrame
    sales_semi_brand_new:pd.DataFrame
    sales_brand_new:pd.DataFrame

    #整合数据 将老品规数据和非完全新品数据整合
    params_total_init=pd.concat([params_all["data_sku_old"],params_all["sku_semi_brand_new"],params_all["sku_brand_new"]])
    self_attributes_total_init=pd.concat([self_attributes["data_sku_old"],self_attributes["sku_semi_brand_new"],self_attributes["sku_brand_new"]])


    #有一些虽然在预测年份之后有销量但是在前面的bass参数拟合的过程中 由于数据量太少 导致无法拟合出bass 参数
    #预测年份有销量的sku 没有对应的bass 参数
    params_can_ues_sku=params_total_init.index.to_list()
    # question=["利群(江南忆)","雄狮(红老版)","利群(钱塘)"]


    #获取end_year+1 到now_year有销量的数据做预测
    sku_sales_data_predict_init,sku_t=sku_sales_init_processing(params_can_ues_sku,sku_sales_data_all,start_year,end_year)

    res_m=pd.DataFrame(columns=columns_name,index=index_name,data=0)
    res_p=pd.DataFrame(columns=columns_name,index=index_name,data=0)
    res_q=pd.DataFrame(columns=columns_name,index=index_name,data=0)
    res = {"res_m": res_m, "res_p": res_p, "res_q": res_q}




    #一共10次10折交叉验证 那么就要为每一次的10折交叉验证存放模型拟合的Bass参数结果
    #最终是一个sku 在一个机器学习中有10个模型拟合结果
    sku_params_modelFit_Ten_times_cross_validation={}
    #存放的是每一个机器学习模型 对预测目标sku的销量误差分析值
    sku_sales_predict_modelFit_Ten_times_cross_validation=None
    if sku_sales_data_predict_init.__class__==pd.DataFrame:
        sku_sales_predict_modelFit_Ten_times_cross_validation=dict(zip(index_name,[pd.DataFrame(data=0,columns=columns_name,index=sku_sales_data_predict_init.columns) for i in range(len(index_name))]))
    else :
        sku_sales_predict_modelFit_Ten_times_cross_validation=dict(zip(index_name,[pd.DataFrame(data=0,columns=columns_name,index=sku_sales_data_predict_init.index) for i in range(len(index_name))]))
    plot_bass_params_true=dict(zip(Bass_params,[list() for i in range(len(Bass_params))]))
    plot_bass_params_hat=dict(zip(Bass_params,[list() for i in range(len(Bass_params))]))






    for i in range(1,2):  #10次 10折
        self_attributes_total = deepcopy(self_attributes_total_init)
        params_total = deepcopy(params_total_init)
        '''
         错误记录：
         不能用np.random 的方法 用该方法打乱 原始父类的顺序也会被打乱 不知道为什么 
         理论来说 深拷贝完 不存在该问题
        '''
        reindex_target=self_attributes_total.index.to_list() #
        random.shuffle(reindex_target)
        self_attributes_total=self_attributes_total.reindex(reindex_target,axis="rows")
        params_total=params_total.reindex(reindex_target,axis="rows")
        #params_total 和 self_attributes_total 的sku排列顺序是一样的
        self_attributes_total_index=self_attributes_total.index.to_list()   #self_attribute的index和params的index 是一样的

        #  后期改一下 不要10折 改小一点
        kf=KFold(n_splits=10,shuffle=True)
        count=1
        res_child_m = pd.DataFrame(columns=columns_name, index=index_name, data=0)
        res_child_p = pd.DataFrame(columns=columns_name, index=index_name, data=0)
        res_child_q = pd.DataFrame(columns=columns_name, index=index_name, data=0)
        res_child={"res_child_m":res_child_m,"res_child_p":res_child_p,"res_child_q":res_child_q}
        #10折交叉验证中 每一个机器学习都要进行10折交叉验证  机器学习名称为key dataFrame 为value
        sku_params_modelFit_Ten_fold_cross_validation=dict(zip(index_name,[pd.DataFrame(columns=Bass_params) for i in range(len(index_name))]))


        plot_feature,scaler_plot=feature_scaling(self_attributes_total)
        plot_feature=pd.DataFrame(data=plot_feature,index=self_attributes_total_index,columns=self_attributes_total.columns)
        plot_Bass_param=deepcopy(params_total)
        plot_Bass_param,cc=feature_scaling(plot_Bass_param)
        plot_Bass_param=pd.DataFrame(data=plot_Bass_param,columns=Bass_params,index=self_attributes_total_index)


        for train_index ,test_index in kf.split(self_attributes_total,self_attributes_total_index):
            def train_test_processing():
                x_train, x_test = self_attributes_total.iloc[train_index], self_attributes_total.iloc[test_index]
                y_train, y_test = params_total.iloc[train_index], params_total.iloc[test_index]
                # x_columns  为各个属性的名称 x_train_index为训练集的sku x_test_index 为测试集的sku
                # y_columns  为三个Bass参数 y_train_index 为训练集的sku  y_test_index 为测试集的sku
                x_columns, x_train_index, x_test_index = x_train.columns.to_list(), x_train.index.to_list(), x_test.index.to_list()
                y_columns, y_train_index, y_test_index = y_train.columns.to_list(), y_train.index.to_list(), y_test.index.to_list()
                # 数据现在都变为二维数组  特征缩放
                scaler_y: prep.StandardScaler
                scaler_x: prep.StandardScaler
                x_train, x_test, scaler_x = feature_scaling(x_train, x_test)
                # y_train,y_test,scaler_y=feature_scaling(y_train,y_test)
                y_train['m'] = y_train['m'] * (1e-6)
                y_test['m'] = y_test['m'] * (1e-6)
                # print('----------特征缩放-----------:',i,count)
                # print("方差: ",np.sqrt(scaler_y.var_),"均值: ",scaler_y.mean_)
                # 将二维数组重新转换为DataFrame
                x_train = pd.DataFrame(data=x_train, columns=x_columns, index=x_train_index)
                x_test = pd.DataFrame(data=x_test, columns=x_columns, index=x_test_index)
                y_train = pd.DataFrame(data=y_train, columns=y_columns, index=y_train_index)
                y_test = pd.DataFrame(data=y_test, columns=y_columns, index=y_test_index)
                return x_train,x_test,y_train,y_test
            x_train,x_test,y_train,y_test=train_test_processing()
            def MLR(x_train,x_test,y_train,y_test,count):
                def MLR_func(x_train,x_test,y_train,y_test,param):
                    reg:LR
                    reg=LR().fit(np.array(x_train),y_train)  #多元线性回归模型
                    # y_hat表示 m,p,q的参数拟合结果  y_test 表示参数m,p,q 真实值结果
                    # 顺序为x_test 的排列顺序
                    y_hat:np.ndarray
                    y_hat=reg.predict(np.array(x_test))
                    get_all_Error_indicator(y_test,y_hat,"MLR",res_child["res_child_"+param],count)
                    print("------%s-------"%param)
                    print(y_hat)
                    print(y_test)
                    for i in range(len(y_hat)):
                        plot_bass_params_true[param].append(y_test[i])
                        plot_bass_params_hat[param].append(y_hat[i])
                    return y_hat.tolist()


                params_temp=[]   #返回Bass 的三个参数 m,p,q
                for param in Bass_params:
                    # print("it is all BASS params")
                    # MLR(np.array(y_train[param]),np.array(y_test[param]))
                    y_hat=MLR_func(x_train, x_test, y_train[param], y_test[param], param)
                    if param=='m':
                        y_hat=[i*(1e6) for i in y_hat]
                    params_temp.append(y_hat)
                # print("bass params   ",params_temp)
                param_temp=np.array([list(row) for row in zip(*params_temp)])  #list 转置
                # a=scaler_y.inverse_transform(param_temp)  #将预测数据的缩放回原始值
                MLR_One_fold_validation = pd.DataFrame(data=param_temp,columns=Bass_params, index=y_test.index)
                # 将一次结果存入10折交叉验证的字典中对应的DataFrame中
                sku_params_modelFit_Ten_fold_cross_validation["MLR"]=pd.concat([sku_params_modelFit_Ten_fold_cross_validation["MLR"],MLR_One_fold_validation])


            MLR(x_train, x_test, y_train, y_test,count)

            count+=1



        # 第k次10折的m,p,q拟合误差数据放入存储表中
        for res_param in res:
            for index in index_name:
                original=res[res_param].loc[index]*(i-1)/i
                new_data=res_child["res_child_"+res_param[-1]].loc[index]/i
                final=np.add(original,new_data)
                res[res_param].loc[index]=final


        #第k次10折的销量预测误差分析
        #先将第k次10折获得的参数结果存入sku_params_modelFit_Ten_times_cross_validation 中
        # sku_params_modelFit_Ten_times_cross_validation    key:dict value:dict
        sku_params_modelFit_Ten_times_cross_validation[i]=sku_params_modelFit_Ten_fold_cross_validation
        # print("第%d次10折交叉验证"%i)
        # print(sku_params_modelFit_Ten_fold_cross_validation["MLR"])
        # print(sku_sales_predict_modelFit_Ten_times_cross_validation["MLR"])
        get_sale_error(sku_sales_predict_modelFit_Ten_times_cross_validation,sku_params_modelFit_Ten_fold_cross_validation,sku_sales_data_predict_init,i,sku_t)

    #最终打印输出的位置
    def file_save():
        roll_year="%d_%d"%(start_year,end_year)
        Bass_params_path_head="../源文件/Bass参数特征拟合结果_test/第%d轮滚动结果/"% (roll_num)
        Sales_path_head="../源文件/销量预测误差统计_test/第%d轮滚动结果/"%(roll_num)
        res["res_m"].to_csv(Bass_params_path_head+"参数m估计误差.csv",index=True)
        res["res_p"].to_csv(Bass_params_path_head+"参数p估计误差.csv",index=True)
        res["res_q"].to_csv(Bass_params_path_head+"参数q估计误差.csv",index=True)

        res_sales_predict:pd.DataFrame
        for Model_name,res_sales_predict in sku_sales_predict_modelFit_Ten_times_cross_validation.items():
            if Model_name== "MLR":
                res_sales_predict.to_csv(Sales_path_head+roll_year+".csv",index=True)
        print("----结束----")
    # file_save()


    # for param in Bass_params:
    #     figure=plt.figure()
    #     plt.scatter(plot_bass_params_true[param],plot_bass_params_hat[param],alpha=0.4)
    #     plt.title("-----%s-----"%param)
    #     plt.xlabel("%s真实值的真实值"%param)
    #     plt.ylabel("%s参数的拟合值"%param)
    # plt.show()

    for param in Bass_params:
        for feature in plot_feature.columns:
            figure=plt.figure()
            plt.scatter(plot_feature[feature],plot_Bass_param[param],alpha=0.4)
            plt.xlabel("----特征值：%s----"%feature)
            plt.ylabel("----Bass参数值:%s----"%param)
    plt.show()








#获取所有的误差评价指标 MAE,MSE...
def get_all_Error_indicator(y_test,y_hat,Model_name,res_child:pd.DataFrame,count):
    Accuracy=None
    if y_test.size==1:
        Accuracy= 1 if y_test==y_hat else 0
    else :
        Accuracy = accuracy_score(y_test.astype("int64"), y_hat.astype("int64"))
    # R2=r2_score(y_test, y_hat)
    MSE=pei.get_MSE(y_test, y_hat)
    MAE=pei.get_MAE(y_test, y_hat)
    RMSE=pei.get_RMSE(y_test, y_hat)
    MAPE=pei.get_MAPE(y_test,y_hat)
    SMAPE=pei.get_SMAPE(y_test,y_hat)
    indicator_list=np.array([Accuracy, MSE, MAE, RMSE,MAPE,SMAPE])/count
    original=res_child.loc[Model_name].to_numpy()*(count-1)/count
    res=np.add(original,indicator_list)
    # print(res/count)
    res_child.loc[Model_name]=res

def get_all_Error_indicator_new(y_test,y_hat):
    Accuracy=None
    if y_test.size==1:
        Accuracy= 1 if y_test==y_hat else 0
    else :
        Accuracy = accuracy_score(y_test.astype("int64"), y_hat.astype("int64"))
    # R2=r2_score(y_test, y_hat)
    MSE=pei.get_MSE(y_test, y_hat)
    MAE=pei.get_MAE(y_test, y_hat)
    RMSE=pei.get_RMSE(y_test, y_hat)
    MAPE=pei.get_MAPE(y_test,y_hat)
    SMAPE=pei.get_SMAPE(y_test,y_hat)
    indicator_list=np.array([Accuracy, MSE, MAE, RMSE,MAPE,SMAPE])
    res=pd.Series(data=indicator_list,index=columns_name)
    return res



#特征缩放  特征缩放需要区分训练集和测试集
#一个样本一行数据
def feature_scaling(train,test=None):
    train_scale: np.ndarray
    test_scale: np.ndarray
    scaler = prep.StandardScaler()
    train_scale = scaler.fit_transform(train)     #纵轴方向数据拟合
    # if not bool(test):  #判断test 是否为NoneType
    if test is None:  #判断test 是否为None
        #如果test  没有传入 则test值为空 为NoneType类型
        return np.array(train_scale),scaler
    # elif test.empty :
    #     return np.array(train_scale)

    test_scale =scaler.transform(test)   #缩放器已经拟合不能再 fit
    return np.array(train_scale),np.array(test_scale),scaler


#销量数据表中的year 做为index
def sku_sales_index_processing(sku_sales_all:dict):
    sku_classify_sales:pd.DataFrame
    for sku_classify_name,sku_classify_sales in sku_sales_all.items():
        sku_classify_sales.set_index(["year"],inplace=True)
#提取end_year 后的销量数据
def sku_sales_init_processing(sku_can_use,sku_sales:dict,start_year,end_year):
    sku_classify_sales: pd.DataFrame
    predict_year=end_year+1
    bool:np.ndarray
    data:pd.DataFrame
    index_year=[i for i in range(predict_year,now_year)]
    sku_sales_data_all_init=pd.DataFrame(index=index_year)
    sku_t=pd.DataFrame(index=index_year)
    drop_column = []
    for sku_classify_name, sku_classify_sales in sku_sales.items():
        if not sku_classify_sales.empty :  #DataFrame 非空
            data=sku_classify_sales.loc[predict_year:,:]
            for sku in data.columns:
                # if sku == "卷烟合计":
                #     drop_column.append(sku)
                #     continue  # 需要剔除掉卷烟合计 因为卷烟合计没有特征
                bool=np.trim_zeros(data[sku].to_numpy())
                if bool.size==0 or sku not in sku_can_use:
                    #size 为0 表示该品规没有再销售了 也就没有做预测的价值
                    # sku_can_use 表示当前阶段有拟合结果的sku
                    drop_column.append(sku)
                else:
                    year_get=np.trim_zeros(sku_classify_sales[sku]).index.to_list()
                    a=np.subtract(year_get,year_get[0]-1)[predict_year-year_get[0]:]   #获取各个品规各自生命周期的预测年份对应的时间
                    #如 [4,5,6,7]  表示从预测年份开始 该品规生命周期的年份
                    a=pd.Series(name=sku,data=a,index=index_year[:len(a)])
                    for i in data.index:
                        if data.loc[i][sku]==0:
                            a.loc[i]=np.nan
                    sku_t=pd.concat([sku_t,a],axis=1)

            sku_sales_data_all_init=pd.concat([sku_sales_data_all_init,data],axis=1)
    sku_sales_data_all_init=sku_sales_data_all_init.drop(drop_column,axis=1)

    # 只返回predict_year的数据  返回对象变成了一个series 而不是dataFrame
    res_sales=sku_sales_data_all_init.loc[predict_year]
    res_t=sku_t.loc[predict_year]
    res_sales=res_sales[res_sales>0]
    res_t=res_t[res_t.notnull()]
    return res_sales,res_t


# 处理销量预测误差
def get_sale_error(sku_sales_predict_modelFit_Ten_times_cross_validation,
                   sku_params_modelFit_Ten_fold_cross_validation,
                   sku_sales_data_predict_init,
                   i,sku_t):
    sku_sales_predict: pd.DataFrame
    sku_Bass_params_one_model: pd.DataFrame
    t: pd.Series
    for model_name in index_name[:1]:
        # 一次10折交叉验证之后将包含所有sku sku_Bass_params中每一次10折交叉验证的sku顺序是不一样的 但是包含所有sku的Bass数据
        # sku_sales_predict 包含全部sku index 且为默认顺序
        # print("第%d次10折交叉验证",i)
        sku_sales_predict = sku_sales_predict_modelFit_Ten_times_cross_validation[model_name]
        sku_Bass_params_one_model = sku_params_modelFit_Ten_fold_cross_validation[model_name]
        # print("final------------------------------------------------")
        sales_predict_total=pd.DataFrame(columns=sku_sales_predict.index)

        for sku in sku_sales_predict.index:
            m, p, q = sku_Bass_params_one_model.loc[sku]
            sales_true=None
            sales_hat=None
            t=None
            if sku_sales_data_predict_init.__class__==pd.DataFrame:  #为dataFrame 表示为为predict_year 后所有的
                sales_true = sku_sales_data_predict_init[sku][sku_sales_data_predict_init[sku] > 0].to_numpy()
                # if sku_t
                t = sku_t[sku][sku_t[sku].notnull()].to_numpy()
                sales_hat = np.array([BM.func(i, m, p, q) for i in t])
            else :#否则为series  表示为只有predict 一年的数据
                sales_true=sku_sales_data_predict_init.loc[sku]
                t = sku_t.loc[sku]
                sales_hat = np.array([BM.func(t, m, p, q)])


            # print("---------销售年份--------",sku,t)
            # print("sku 的bass 参数",m,p,q)
            # print(t)
            # print("sku真实值",sales_true)
            # print(t)

            # print("sku拟合值",sales_hat)
            # print(sku_sales_data_predict_init[sku])
            # print(t)
            get_all_Error_indicator(sales_true, sales_hat, sku, sku_sales_predict, i)
        # print(sku_sales_predict_modelFit_Ten_times_cross_validation[model_name])





