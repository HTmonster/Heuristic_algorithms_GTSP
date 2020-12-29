'''
Author: Theo_hui
Date: 2020-12-14 20:14:23
Descripttion: 遗传算法
'''
import random
import math
import time

from .Heuristic import Heuristic

class GA(Heuristic):
    ''' 遗传算法 '''

    def __init__(self,
                M,              # 种群规模
                Pc,             # 交叉概率
                Pm,             # 变异概率
                TSP,            # TSP实例
                iteration=1000  # 迭代次数
                ):

        self.M          =   M
        self.Pc         =   Pc
        self.Pm         =   Pm
        self.TSP        =   TSP
        self.iteration  =   iteration # 迭代次数
    
    def Generate_initial_pops(self):
        ''' 生成初始种群 '''
        pops=[]

        for _ in range(self.M):
            path=[]
            city_index=self.TSP.city_index_groups.copy()
            
            # 随机生成个体
            for __ in range(len(self.TSP.city_index_groups)):
                group=random.choice(city_index)
                path.append(random.choice(group))
                city_index.remove(group)
            pops.append(path)


        
        return pops

    def cal_adaptation(self,pops):
        ''' 计算种群的适应度 适应度为 10000/距离'''

        adaps=[]
        for pop in pops:
            dist    =self.TSP.cal_path_distance(pop) # 计算距离
            adap    =10000.0/dist                    # 适应度
            adaps.append(adap)
        return adaps
    
    def select_prop_operation(self,adaps,pops):
        ''' 比例选择运算 轮盘赌算法 '''
        
        # 积累分布函数
        adap_sum,y=[],0
        for adap in adaps:
            y+=adap
            adap_sum.append(y)
        
        # 保留M*(1-Pc-Pm)个
        new_pops=[]
        for _ in range(int(self.M*(1-self.Pc-self.Pm))):
            rand=random.uniform(0,adap_sum[-1])#产生随机数 转一次轮盘
            for j in range(len(adap_sum)):
                if rand<adap_sum[j]:
                    new_pops.append(pops[j])
                    break
        return new_pops
    
    def cross(self,pops):
        ''' 单点交叉 '''
        new_pops=pops.copy()

        # 变异M*Pc个
        for _ in range(int(self.M*self.Pc)):
            which = random.randint(0,len(new_pops)-1) # 选择一个进行交叉
            pop=new_pops[which].copy()
            while pop in new_pops:
                # 必须产生新的
                p,q=random.randint(0,len(self.TSP.city_index_groups)-1),random.randint(0,len(self.TSP.city_index_groups)-1)
                pop[p],pop[q]=pop[q],pop[p]          # 第一次交叉
                m,n=random.randint(0,len(self.TSP.city_index_groups)-1),random.randint(0,len(self.TSP.city_index_groups)-1)
                pop[m],pop[n]=pop[n],pop[m]          # 第二次交叉
                # m,n=random.randint(0,len(self.TSP.city_index_groups)-1),random.randint(0,len(self.TSP.city_index_groups)-1)
                # pop[m],pop[n]=pop[n],pop[m]          # 第三次交叉
                # m,n=random.randint(0,len(self.TSP.city_index_groups)-1),random.randint(0,len(self.TSP.city_index_groups)-1)
                # pop[m],pop[n]=pop[n],pop[m]          # 第四次交叉
            new_pops.append(pop)
        return new_pops
    
    def variation(self,pops):
        ''' 变异操作 '''
        new_pops=pops.copy()
        # 变异M*Pm个
        for _ in range(int(self.M*self.Pm)):
            which = random.randint(0,len(new_pops)-1) # 选择一个进行变异
            pop=new_pops[which].copy()
            while pop in new_pops:
                # 选择同一个物品的城市进行随机替换
                group = random.choice([x for x in self.TSP.city_index_groups if len(x)>1]).copy()
                for i in range(len(pop)):
                    if pop[i] in group:                       
                        group.remove(pop[i])
                        pop[i] = random.choice(group)
            new_pops.append(pop)
        return new_pops

    def get_best_pop(self,pops,adaps):
        ''' 找到最适应的种群 '''
        max_adap=max(adaps)
        index=adaps.index(max_adap)

        return pops[index]

    def solve(self, record_step=100):
        ''' 执行遗传算法'''

        #0.========== 初始化种群 ===============
        pops    = self.Generate_initial_pops()   # 初始种群
        adaps   = self.cal_adaptation(pops)      # 适应度
        
        star_t=time.time()

        print("======================[ GA ]===========================")
        
        for i in range(self.iteration):
            
            # 记录信息
            if i % record_step==0:
                pop=self.get_best_pop(pops,adaps)
                self.costs.append(self.TSP.cal_path_distance(pop))

            #1.======== 比例选择运算 ==========
            new_pops=self.select_prop_operation(adaps,pops)

            #2.======== 单点交叉     ==========   
            new_pops=self.cross(new_pops)

            #3.======== 变异操作     ==========
            new_pops=self.variation(new_pops)

            pops=new_pops
            adaps=self.cal_adaptation(pops)

            # 记录历史最优解
            best_path = self.get_best_pop(pops,adaps)
            
            if self.TSP.distance and self.TSP.cal_path_distance(best_path)<self.TSP.distance: self.TSP.set_path(best_path)
            if not self.TSP.distance: self.TSP.set_path(best_path)
                

            print("[{}] path:{} cost:{}".format(i,self.get_best_pop(pops,adaps),10000/max(adaps)))

        end_t=time.time()

        print("=============[ path:{} cost:{}  time:{:.4}s]===============".format(self.TSP.path,self.TSP.distance,end_t-star_t))
        
        return self.TSP

