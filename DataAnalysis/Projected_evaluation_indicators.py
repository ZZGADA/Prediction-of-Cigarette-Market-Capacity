import math
import numpy as np
def get_MAE(y_true,y_hat)->float :
    """
    :return
        返回类型：int
        返回预拟合和真实值的平均绝对误差 算法公式为：求拟合值与真实值差值的绝对值，然后对所有绝对值数据进行加总求和 最后除以数据总体量

    :param y_hat: 拟合值数据 数据类型为数组、列表（Arrays_like）

    :param y_true: 拟合值数据 数据类型为数组、列表（Arrays_like）

    """
    y_hat = np.array(y_hat)
    y_true = np.array(y_true)
    size_t=len(y_true) #真实值数据长度
    size_h=len(y_hat) #拟合值数据长度
    size=0
    if(size_h!=size_t):
        raise Exception("拟合值数据和真实值数据长度不匹配")

    size=size_t

    y_subtract=np.subtract(y_hat,y_true)
    abs_sum=np.absolute(y_subtract).sum()
    return abs_sum/size

    # abs_sum=0
    # for i in range(size):
    #     abs_sum+=abs(y_hat[i]-y_true[i]) #差值的绝对值
    # return abs_sum/size
    #这里也可以直接转换为np.array 简化计算

def get_MSE(y_true,y_hat)->float:
    """

    :param y_hat:  拟合值数据 type 为列表
    :param y_true: 真实值数据 type 为列表
    :return: square_sum/size 返回值为拟合值与真实值差值的平方加总求和的均值 为残差平方和的均值
    """


    y_hat = np.array(y_hat)
    y_true = np.array(y_true)
    size_h=len(y_hat)
    size_t=len(y_true)

    if(size_h!=size_t):
        raise Exception("拟合值数据和真实值数据长度不匹配")

    size=size_t

    y_subtract=np.subtract(y_hat,y_true)
    square_sum=np.power(y_subtract,2).sum()
    # for i in range(size):
    #     square_sum+=pow(y_hat[i]-y_true[i],2)     #差值的平方
    return square_sum/size

def get_RMSE(y_true,y_hat)->float:
    res=get_MSE(y_hat,y_true)
    return math.sqrt(res)

def get_MAPE(y_true, y_hat)->float:
    y_hat = np.array(y_hat)
    y_true = np.array(y_true)

    size_h = len(y_hat)
    size_t = len(y_true)


    if (size_h != size_t):
        raise Exception("拟合值数据和真实值数据长度不匹配")
    size=size_t


    a=np.abs(np.subtract(y_hat,y_true)/y_true)
    res=np.sum(a)/size*100
    return res


    # return np.mean(np.abs((y_hat - y_true) / y_true)) * 100



def get_SMAPE(y_true, y_hat)->float:
    y_hat = np.array(y_hat)
    y_true = np.array(y_true)

    size_h = len(y_hat)
    size_t = len(y_true)

    if (size_h != size_t):
        raise Exception("拟合值数据和真实值数据长度不匹配")



    return 2.0 * np.mean(np.abs(y_hat - y_true) / (np.abs(y_hat) + np.abs(y_true)))



