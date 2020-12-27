'''
Author: Theo_hui
Date: 2020-12-17 15:18:24
Descripttion: 
'''
'''
Author: Theo_hui
Date: 2020-12-14 20:14:23
Descripttion: 禁忌搜索算法
'''
import random
import math
import time

from .Heuristic import Heuristic

class TS(Heuristic):
    ''' 禁忌搜索算法 '''

    def __init__(self,
                table_len,      # 禁忌表长度
                TSP,            # TSP实例
                iteration=1000  # 迭代次数
                ):

        self.taboo_table    =[]     # 禁忌表
        self.path_initial   =[]    # 路径
        
        self.table_len  =   table_len
        self.TSP        =   TSP
        self.iteration  =   iteration # 迭代次数
    
    def find_new_path(self,best_path):
        ''' 寻找上一个最优解的所有领域解 '''
        
        paths=[]

        # 1. 第一类 交换城市顺序
        for i in range(1,len(best_path)-1):
            for j in range(i+1,len(best_path)):
                path=best_path.copy()
                path[i],path[j]=path[j],path[i] 
                paths.append(path)

        # 2. 第二类 交换同类物品中的城市
        for i in range(len(best_path)):
            for group in self.TSP.city_index_groups:
                if best_path[i] in group:
                    group.remove(best_path[i]) # 移除自己
                    for other_city in group: 
                        path=best_path.copy()
                        path[i]=other_city
                        paths.append(path)
        return paths
    
    def find_best_dist_path(self,paths):
        ''' 找到路径中最好的路径 和对应的距离'''
        best_dist,best_path=float('inf'),None

        for path in paths:
            dist=self.TSP.cal_path_distance(path) #计算距离
            if dist<=best_dist:
                best_dist,best_path=dist,path
        return best_dist,best_path

    
    def solve(self, record_step=100):
        ''' 执行禁忌搜索算法'''

        #初始化TSP
        path,dist=self.TSP.set_random_path(),self.TSP.distance #初始化随机路径
        self.taboo_table.append(path)     #加入禁忌表

        star_t=time.time()

        print("======================[ TS ]===========================")
        
        # 初始的期望距离与路径
        best_path,best_dist=path,dist
        for i in range(self.iteration):
            # 记录下消耗
            if i % record_step==0:
                self.costs.append(best_dist)

            #1.=========== 获得领域解=============
            paths=self.find_new_path(best_path)                     # 所有领域解
            best_dist,best_path=self.find_best_dist_path(paths)     # 领域解中的最优解

            #2.=========== 更新解 ===============
            if best_dist<self.TSP.distance:
                #2.1====小于历史最优解  更新解==========
                self.TSP.set_path(best_path)          #历史最优解
            elif best_path in self.taboo_table:
                #2.2===大于期望  且最优解在禁忌表===
                # 选择不在禁忌表中的最优解
                while best_path in self.taboo_table:
                    paths.remove(best_path)
                    best_dist,best_path=self.find_best_dist_path(paths)

            #3.=========== 更新禁忌表 ==========
            self.taboo_table.append(best_path)       #把当前最优解加入到禁忌表中
            if len(self.taboo_table)>=self.table_len: del self.taboo_table[0] # 保持长度
            
            print("[{}] path:{} cost:{}".format(i,best_path,best_dist))

        end_t=time.time()

        print("=============[ path:{} cost:{}  time:{:.4}s]===============".format(self.TSP.path,self.TSP.distance,end_t-star_t))
        
        return self.TSP

