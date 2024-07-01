from scipy.optimize import leastsq, curve_fit
import sympy as sp
from math import e
from math import log

def func(t,m,p,q):
    #m, p, q = params
    fz = (p * (p + q) ** 2) * e ** (-(p + q) * t)  # 分子的计算
    fm = (p + q * e ** (-(p + q) * t)) ** 2  # 分母的计算
    nt = m * fz / fm  # nt值
    return nt



def func_contain_t_max(t,m,p,q,t_max):
    t_max_test = get_sales_max_t(p, q)
    fz = (p * (p + q) ** 2) * e ** (-(p + q) * t)  # 分子的计算
    fm = (p + q * e ** (-(p + q) * t)) ** 2  # 分母的计算
    nt = m * fz / fm  # nt值

    return nt
# 误差函数函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
# 一般第一个参数是需要求的参数组，另外两个是x,y
# def func_test()
def error(params, t, y):
    ret=func(t,*params) - y
    ret = np.append(ret, 0.01 * params)  # 增加penalize item

#获得最大销量时的时间
def get_sales_max_t(p,q):
    sales_max_t=-log(p/q)/(p+q)
    return sales_max_t

#获得最大销量
def get_sales_max(m,p,q):
    return (p+q)**2/(4*q)*m



if __name__=='__main__':
    print('it is Bass_Model ')

