import numpy as np
import pandas as pd
import os
from Service import Characteristic_Mapping as CM
'''
从excel或者csv格式的文件中获取数据
'''
def get_Data(path:str,index_col=None)-> pd.DataFrame:
    '''

    :param path:
        path 为字符串
    :return:
        返回类型为DataFrame
    '''
    Data=None
    #判断path后缀
    if path.__contains__(".xlsx"):#后缀为.xlsx 读取excel
        Data=pd.read_excel(path,index_col=index_col)
    elif path.__contains__(".csv") :#后缀为.csv 读取.csv
        Data=pd.read_csv(path,index_col=index_col)
    return Data
    # return Data

def get_feature_province(path:str,index_col=None)->pd.DataFrame:
    if index_col!=None:
        return get_Data(path,index_col)
    return get_Data(path)


def get_feature_region(path:str,index_col=None)->pd.DataFrame:
    if index_col!=None:
        return get_Data(path,index_col)
    return get_Data(path)
#获取地市下的所有品规销量



def get_region_data(path):
    '''这个函数返回的是当前文件夹名称，
    子文件夹名以及当前文件夹下的所有文件名这三个值，
    然后每个值都是一个列表的形式，所以我们用for循环一下当前文件夹下所有文件'''
    Data_all_sku_region=pd.DataFrame(index=[i for i in range(2006,2023)])
    Data_all_sku_region.index.name="year"
    for folderName, subFolders, filenames in os.walk(path):
        for sku_name_csv in filenames:
            data_temp=pd.read_csv(path+"/"+sku_name_csv,index_col=0)
            sku_name=sku_name_csv.strip(".csv")
            Data_all_sku_region[sku_name]=data_temp['sale']
    Data_all_sku_region=Data_all_sku_region.reset_index()
    Data_all_sku_region.to_csv("../源文件/地市/全部sku的原始销量.csv",index=False)
    return Data_all_sku_region

def get_region_supplementary_feature(path,previous_year):
    #该函数用于获得后续补充的特征
    temp_feature:pd.Series
    columns_set=["sale_previous_year","PFprice_SD_previous_year","PFprice_CV_previous_year","LSprice_average_previous_year","LSprice_SD_previous_year","LSprice_CV_previous_year"]
    columns_original=[i[0:i.find("_previous_year")] for i in columns_set]

    feature_supplementary = pd.DataFrame(columns=columns_set)
    for folderName, subFolders, filenames in os.walk(path):
        for sku_name_csv in filenames:
            data_temp = pd.read_csv(path + "/" + sku_name_csv, index_col=0).fillna(0)
            sku_name = sku_name_csv.strip(".csv")
            temp_feature=pd.Series(data=data_temp.loc[previous_year,columns_original].to_list(),index=columns_set)
            feature_supplementary.loc[sku_name]=temp_feature
            # feature_supplementary.loc[sku_name,"total_sales"]=CM.sales_calculation(data_temp.loc[:previous_year,"sale"])

    feature_supplementary.to_csv("../源文件/地市/补充特征.csv")
    return feature_supplementary





if __name__=="__main__":
    print("it is module Data_Acquisition")
