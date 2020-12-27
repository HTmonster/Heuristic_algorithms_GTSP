'''
Author: Theo_hui
Date: 2020-12-14 20:14:23
Descripttion: 蚁群算法
'''
import random
import math
import time
import numpy as np

from .Heuristic import Heuristic

class AC(Heuristic):
    ''' 蚁群算法 '''

    def __init__(self,
                ants_num,       # 蚂蚁数量
                alpha,          # 信息素影响因子
                beta,           # 期望影响因子
                Rho,            # 信息素挥发率
                Q,              # 信息素释放的量  常量
                TSP,            # TSP实例
                iteration=1000  # 迭代次数
                ):

        self.ants_num   =   ants_num
        self.alpha      =   alpha
        self.beta       =   beta
        self.Rho        =   Rho
        self.Q          =   Q
        self.TSP        =   TSP
        self.iteration  =   iteration # 迭代次数
    
    def cal_path_distance(self,path_mat):
        ''' 计算距离'''
        dis_list=[]
        for path in path_mat:
            dis_list.append(self.TSP.cal_path_distance(path))
        return dis_list
    
    def _remove_same_group_city(self,citys,choose):
        ''' 移除与choose是同一物品的城市 (包括自己)'''
        for group in self.TSP.city_index_groups:
            if choose in group:
                for city in group:
                    try:
                        citys.remove(city)
                    except Exception as e:
                        print(city,citys,group)
                        raise e
                return citys
    
    
    def solve(self, record_step=100):
        ''' 执行蚁群算法'''

        #初始化
        items_num = len(self.TSP.city_index_groups)                # 物品数量
        citys_num = len(self.TSP.city_locations)                   # 城市数量
        distMat=self.TSP.distMat                                   # 距离矩阵

        pheromone_mat=np.ones((citys_num,citys_num))               # 信息素浓度矩阵 初始为1
        path_mat=np.zeros((self.ants_num,items_num)).astype(int)   # 路径矩阵

        star_t=time.time()

        print("======================[ AC ]===========================")
        
        for count in range(self.iteration):

            #1.=== 随机放置蚂蚁 并让所有蚂蚁爬行完毕 ====
            for ant in range(self.ants_num):
                unvisit_list=list(range(citys_num))                 # 未访问的
                choose=random.choice(unvisit_list)                  # 初始随机的放置蚂蚁
                unvisit_list=self._remove_same_group_city(unvisit_list,choose)   # 移除有同类物品的城市
                
                # 记录开始城市
                path_mat[ant,0]=choose
                
                for j in range(1,items_num):
                    #轮盘法选择下一个城市
                    trans_list,trans=[],0
                    for k in range(len(unvisit_list)):
                        # 计算概率 与信息素浓度成正比, 与距离成反比
                        trans +=np.power(pheromone_mat[choose][unvisit_list[k]],self.alpha)*np.power(1.0/distMat[choose][k],self.beta)
                        trans_list.append(trans)
                    rand=random.uniform(0,trans)#产生随机数

                    # 转一次轮盘 选择
                    for t in range(len(trans_list)):
                        if rand <= trans_list[t]:
                            choose_next=unvisit_list[t]
                            break
                        else:
                            continue
                    else:
                        choose_next=unvisit_list[-1]
                    path_mat[ant,j]=choose_next#填路径矩阵

                    # 更新候选城市
                    visit=choose_next
                    unvisit_list=self._remove_same_group_city(unvisit_list,visit)

            #所有蚂蚁的路径表填满之后，算每只蚂蚁的总距离
            dis_all_ant_list=self.cal_path_distance(path_mat)

            #3.=========== 更新信息素矩阵 ==============
            pheromone_change=np.zeros((citys_num,citys_num))
            for i in range(self.ants_num):
                # 在路径上的每两个相邻城市间留下信息素，与路径距离反比
                for j in range(items_num-1):
                    pheromone_change[path_mat[i,j]][path_mat[i,j+1]] += self.Q/distMat[path_mat[i,j]][path_mat[i,j+1]]
                pheromone_change[path_mat[i,items_num-1]][path_mat[i,0]] += self.Q/distMat[path_mat[i,items_num-1]][path_mat[i,0]]
            pheromone_mat=(1-self.Rho)*pheromone_mat+pheromone_change

            #记录历史最佳值       
            if count == 0 or min(dis_all_ant_list) < dis_new:
                dis_new=min(dis_all_ant_list)
                path_new=path_mat[dis_all_ant_list.index(dis_new)].copy()
                self.TSP.set_path(path_new)

            # 记录迭代的值
            if count % record_step==0:
                self.costs.append(dis_new)

            print("[{}] path:{}      cost:{}".format(count,path_new,dis_new).replace("\n",''))

        end_t=time.time()

        print("=============[ path:{} cost:{}  time:{:.4}s]===============".format(self.TSP.path,self.TSP.distance,end_t-star_t))
        
        return self.TSP

