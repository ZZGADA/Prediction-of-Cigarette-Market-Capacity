'''
数据处理的module
'''
import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


#数据清洗 现在可以跑了 一定不要动它
def revise_data(sku_data:pd.Series):   #修订存在偏误的数据
    max_index=sku_data.argmax()#返回最大值的Series索引
    max_num=sku_data.iloc[max_index]
    sku_data_size=sku_data.size
    sales_growth_rate=np.diff(sku_data)  #销量增速度(斜率)  前后做差 必须保证最大值索引<max_index,最小值索引(销量减小最大值)>=max_index

    sales_growth_rate_size=None
    diff_sales_growth_rate=np.diff(sales_growth_rate)  #销量增速的变化程度 三个区段均近似为一条直线
    max_sales_growth_rate_index=None
    min_sales_growth_rate_index=None

    flag=np.zeros(sku_data_size)  #-1为下调 0不变 1为上调
    flag_point_index=None
    FenWeiDian=None
    temp=None

    #销量下降的部分
    def decre(sales_growth_rate:np.array,sku_data:pd.Series,sku_past=None):
        # print("下降的")
        # print(sales_growth_rate)
        temp=None
        sales_growth_rate_size = sales_growth_rate.size
        if sales_growth_rate_size<3:
            return
        # 斜率大于0为异常值
        flag_point_index = [i for i in range(sales_growth_rate_size) if sales_growth_rate[i] > 0]
        if len(flag_point_index) == 0:
            # print("一切正常")
            return
        flag_point_index.append(flag_point_index[-1] + 1)
        '''
            加入年份极其异常数据的判断  销量逐渐下降的过程中从某一年份开始 销量再次连续增加 
            则可以判断之前的下降过程为人为操作 该品规销量将按照后期销量增加的趋势增长 
            否则继续下降的数值处理
        '''
        if len(flag_point_index)-1>=sales_growth_rate_size/2:
            # print("极其异常判断")  #将原先下降趋势的曲线 重新判定为增长的曲线

            # print(sku_data)
            if flag_point_index[-1]<sku_data.size-1:
                if sku_data.iloc[flag_point_index[-1]+1]>sku_data.iloc[flag_point_index[-1]]:
                    incre(sales_growth_rate, sku_data,sku_past)
                    return



        FenWeiDian = flag_point_index[-1] - flag_point_index[0] + 3  #确定准确的分位点 然后+1
        # 寻找销量增速的最小值
        min_sales_growth_rate_index = sales_growth_rate.argmin()  # 增速的最小值索引
        if flag_point_index[0] < min_sales_growth_rate_index:
            # print("向上调整")
            temp = np.linspace(sku_data.iloc[flag_point_index[0] - 1], sku_data.iloc[flag_point_index[-1]],
                               FenWeiDian)
            sku_data.iloc[flag_point_index[0]:flag_point_index[-1]] = temp[1:-2]

        else:
            # print("向下调整")
            if flag_point_index[-1] == sales_growth_rate_size:
                kTemp = sales_growth_rate[flag_point_index[0] - 1]
                temp_point=np.subtract(sku_data.iloc[flag_point_index[0] - 1],kTemp*flag_point_index[-1])
                temp=np.linspace(sku_data.iloc[flag_point_index[0]],0,FenWeiDian)
                # temp = [kTemp + sku_data.iloc[flag_point_index[0] - 1]]
                # print(temp)
                sku_data.iloc[flag_point_index[0]+1:flag_point_index[-1] + 1] = temp[2:-1]

            else:
                temp = np.linspace(sku_data.iloc[flag_point_index[0]],
                                    sku_data.iloc[flag_point_index[-1]+1], FenWeiDian)
                sku_data.iloc[flag_point_index[0]+1:flag_point_index[-1] + 1] = temp[2:-1]
    #销量上升的部分
    def incre(sales_growth_rate:np.array,sku_data:pd.Series):
        sku_data_size=sku_data.size
        # print("上升的")
        # print(sales_growth_rate)
        temp=None
        sales_growth_rate_size = sales_growth_rate.size
        # 斜率小于0为异常值
        flag_point_index = [i for i in range(sales_growth_rate_size) if sales_growth_rate[i] < 0]
        if len(flag_point_index) == 0:  # 表明一切正常
            # print("一切正常")
            return
        flag_point_index.append(flag_point_index[-1] + 1)
        FenWeiDian = flag_point_index[-1] - flag_point_index[0] + 3

        # 寻找销量增速的最大值
        max_sales_growth_rate_index = sales_growth_rate.argmax()  # 增速最大值索引
        if flag_point_index[0] < max_sales_growth_rate_index:
            # print("向下调整")  # 记得对初始点做处理
            if flag_point_index[0] == 0:   #初始点存在问题
                # try:
                kTemp = sales_growth_rate[flag_point_index[-1]]
                # except:
                    # print(sales_growth_rate)
                    # print(flag_point_index)
                    # print(sku_data)
                temp_point=np.subtract(sku_data.iloc[flag_point_index[-1]],kTemp*flag_point_index[-1])
                temp=np.linspace(0,sku_data.iloc[flag_point_index[-1]],FenWeiDian)
                sku_data.iloc[flag_point_index[0]:flag_point_index[-1]]=temp[1:-2]   # 修改数据
            else:
                temp = np.linspace(sku_data.iloc[flag_point_index[0] - 1], sku_data.iloc[flag_point_index[-1]],
                                   FenWeiDian)
                sku_data.iloc[flag_point_index[0]:flag_point_index[-1]]=temp[1:-2]
        else:
            # print("向上调整")
            temp = np.linspace(sku_data.iloc[flag_point_index[0]], sku_data.iloc[flag_point_index[-1] + 1],
                               FenWeiDian)
            sku_data.iloc[flag_point_index[0]+1:flag_point_index[-1]+1] = temp[2:-1]

    # print(sales_growth_rate)
    if sku_data_size<2:
        # print("只有唯一或者没有数据，可以跳过")
        pass
    elif max_index==sku_data_size-1:
        # print("sku的销量是持续增加的")
        incre(sales_growth_rate,sku_data)

    elif max_index==0:
        # print("sku的销量是持续减少的")
        decre(sales_growth_rate,sku_data)
    else :
        # print("sku的销量有峰值")
        incre(sales_growth_rate[:max_index],sku_data[:max_index+1])
        decre(sales_growth_rate[max_index:],sku_data[max_index:],sku_data)
#补充生命周期前端缺失的数据
def supplementary_data():
    print("补充数据")
#数据预处理
def data_PreProcessing(Data_init:pd.DataFrame,start_year,end_year,count,path):#np.array or pd.DataFrame or pd.Series
    # print("it is data_PreProcessing")
    '''销量数据的话 横轴shape[1]方向 为品规SKU 纵轴shape[0]方向 为年份 '''
    sku:str
    predict_year=end_year+1
    end_year_index=end_year-start_year
    predict_year_index=end_year-start_year+1
    drop_name=[]
    # for sku in Data_init.columns.to_list():
        # if sku.__contains__("L"):
            # sku_new=sku.replace("L","")
            # Data_init[sku_new]=np.add(Data_init[sku_new],Data_init[sku])
            # drop_name.append(sku)
            # print(Data_init[sku_new])
    # input()
    Data=Data_init.drop(columns=drop_name)
    Data_predict=Data.iloc[predict_year_index,:]
    Data=Data.iloc[:predict_year_index,]
    for  i in range(Data_predict.size):
        if Data_predict.iloc[i]<5:
            Data_predict.iloc[i]=0
    for sku in Data.columns.to_list()[1:]:  #第一列year不要
        data_temp = np.trim_zeros(Data[sku])  #去掉数据首尾的0元素 保留全生命周期的记录数据
        if data_temp.size==0:
            continue
        for i in data_temp.index:
            if data_temp[i]<5: #3.08
                data_temp[i]=0

        # if sku!="利群(阳光)":
            # revise_data(data_temp)   #data_temp 传入函数为引用  地址不变 所以不需要改变
        Data[sku][data_temp.index] = data_temp
        if Data[sku][predict_year_index-1]/Data_predict[sku]>9:
            Data_predict[sku]=0
    Data.loc[predict_year_index]=Data_predict

    # print("清洗结束")
    os.makedirs(path%count,exist_ok=True)
    return_path=path%count+"/清洗后的全sku数据.csv"
    Data.to_csv(return_path,index=False)

    return return_path


if __name__=='__main__':
    print("it is data processing module ")