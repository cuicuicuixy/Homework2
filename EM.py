import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

    
#概率密度函数
def Gauss(X,mu,sigma):
    result = (1 / (np.sqrt(2 * math.pi) * sigma)) * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
    return result
#E_step
def E_step(heights,a1,u1,o1,a2,u2,o2):
    gama1 = a1*Gauss(heights,u1,o1)
    gama2 = a2*Gauss(heights,u2,o2)
    sum = gama1 + gama2
    gama1 = gama1 / sum
    gama2 = gama2 / sum
    return(gama1,gama2)
#M_step
def M_step(u1,u2,gama1,gama2,heights):
    u1_new = np.dot(gama1,heights)/np.sum(gama1)
    u2_new = np.dot(gama2,heights)/np.sum(gama2)
    
    o1_new = math.sqrt(np.dot(gama1,(heights - u1)**2)/np.sum(gama1))
    o2_new = math.sqrt(np.dot(gama2,(heights - u2)**2)/np.sum(gama2))
    
    a1_new = np.sum(gama1)/len(gama1)
    a2_new = np.sum(gama2)/len(gama2)
    
    return u1_new,u2_new,o1_new,o2_new,a1_new,a2_new
if __name__ == '__main__':
    start = time.time()    
    #读入身高数据
    df= pd.read_csv('./height_data.csv', header=0, names=['height'])
    heights = [row['height'] for index, row in df.iterrows()]
    #划分训练集和测试集3：1  
    np.random.shuffle(heights)
    train_heights = heights[:1500]
    test_heights  = heights[1500:]
    #设定初始值    
    u1,u2,o1,o2,a1,a2 = 175,150,10,10,0.5,0.5
    print('---------------------------')
    print('初始设置参数为：')
    print('男生比例a1:%.6f\n男生均高u1:%.6f\n男生方差o1:%.6f\n女生比例a2:%.6f\n女生均高u2:%.6f\n女生方差o2:%.6f' %(a1,u1,o1,a2,u2,o2))
    #开始训练，循环500次
    iter = 500
    step = 0
    train_heights = np.array(train_heights)
    while (step < iter):
        step += 1
        gama1,gama2 = E_step(train_heights, a1, u1, o1, a2, u2, o2)
        u1,u2,o1,o2,a1,a2 = M_step(u1,u2,gama1,gama2,train_heights)
    #输出训练结果
    print('----------------------------')
    print('EM算法迭代得到参数为:')
    print('男生比例a1:%.6f\n男生均高u1:%.6f\n男生方差o1:%.6f\n女生比例a2:%.6f\n女生均高u2:%.6f\n女生方差o2:%.6f' %(a1,u1,o1,a2,u2,o2))
    #输出运行时间
    print('----------------------------')
    print('运行时间：', time.time() - start)    
    #测试集上对数似然估计
    p1 = a1 * np.exp(-0.5 * (test_heights - u1) ** 2 / o1 ** 2) / np.sqrt(2 * np.pi * o1 ** 2)
    p2 = a2 * np.exp(-0.5 * (test_heights - u2) ** 2 / o2 ** 2) / np.sqrt(2 * np.pi * o2 ** 2)
    log_likelihood = np.sum(np.log(p1 + p2))
    print('似然估计:', log_likelihood)
    #做出直方图
    print('----------------------------')
    print('     图片：训练集混合高斯分布直方图')
    x = np.linspace(min(train_heights), max(train_heights), 200)
    def y(x):
        y = a1 * norm.pdf(x, u1, o1) + a2 * norm.pdf(x, u2, o2)
        return y * len(train_heights)
    plt.hist(train_heights, bins=20, color="w", label="直方分布图", edgecolor='k')
    plt.xlabel('height/cm')
    plt.ylabel('Count')
    plt.plot(x, y(x), 'b', linewidth=2)
    plt.show()

    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    
    