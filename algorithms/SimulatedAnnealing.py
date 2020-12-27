'''
Author: Theo_hui
Date: 2020-12-14 20:14:23
Descripttion: 模拟退火算法
'''
import random
import math
import time

from .Heuristic import Heuristic

class SA(Heuristic):
    ''' 模拟退火算法 '''

    def __init__(self,
                init_t,         # 初始温度
                lowest_t,       # 最低温度
                rate,           # 降温系数
                TSP,            # TSP实例
                iteration=1000  # 每次迭代次数
                ):

        self.current_t  = init_t    # 当前温度为初始温度
        self.lowest_t   = lowest_t
        self.rate       = rate
        self.TSP        = TSP
        self.iteration  = iteration
    
    def random_change(self,path):
        ''' 随机的扰动，改变路径 '''
        #1. 先随机的交换城市位置
        new_path=path.copy()
        i=j=0
        while j==i: i,j=random.randint(1,len(new_path)-1),random.randint(1,len(new_path)-1)
        new_path[i],new_path[j]=new_path[j],new_path[i]
        
        #2. 再概率的交换同一种物品中的城市
        for i in range(len(new_path)):
            if random.random()>0.3:
                for group in self.TSP.city_index_groups:
                    if new_path[i] in group:
                        new_path[i]=random.choice(group)
                        break
        return new_path 

    
    def solve(self,record_step=100):
        ''' 执行退火算法'''

        #初始化TSP路径
        path,distance=self.TSP.set_random_path(),self.TSP.distance

        star_t=time.time()

        print("======================[ SA ]===========================")
        
        total_count=0
        while (self.current_t>self.lowest_t):
            # 外循环 改变温度
            print("[current temperature: {}  lowest:{}]".format(self.current_t,self.lowest_t))
            count_iter=0 #迭代计数

            while count_iter < self.iteration:
                if total_count % record_step==0:
                    self.costs.append(distance)
                # 内循环 迭代次数超过 退出
                #1.=============随机扰动=====================
                new_path=self.random_change(path)
                #2.=============计算Δ======================
                new_disrance =  self.TSP.cal_path_distance(new_path)  # 计算新花费
                delta        =  new_disrance-distance    # 计算花费差

                #3.=============判断更新======================
                if delta <=0 :
                    # 接受新解
                    path,distance=new_path,new_disrance
                elif math.exp(-delta/self.current_t) > random.random():
                    # 否则按照Metropolis准则接受新解
                    path,distance=new_path,new_disrance

                #历史最优解 更新
                if distance<self.TSP.distance: self.TSP.set_path(path)    
                
                print("[{}] path:{} dist:{}".format(count_iter,path,distance))

                count_iter+=1 # 迭代次数增加
                total_count+=1 
                
            #=================改变温度================
            self.current_t=self.rate*self.current_t
        
        end_t=time.time()

        print("=============[ path:{} cost:{}  time:{:.4}s]===============".format(self.TSP.path,self.TSP.distance,end_t-star_t))
        
        return self.TSP

