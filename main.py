'''
Author: Theo_hui
Date: 2020-12-27 13:57:47
Descripttion: 主程序
'''
import random

import numpy as np
import matplotlib.pyplot as plt

from algorithms.SimulatedAnnealing import SA
from algorithms.TabuSearch         import TS
from algorithms.GeneticAlgorithm   import GA
from algorithms.AntColonyAlgorithm import AC

class GTSP:
    ''' 广义旅行商问题类 '''

    path,distance=[],None

    #(9,5) 城市实例 9个城市 5种物品
    citys_9_5=[
        # 城市编号 ， 城市坐标
        [[0,[78, 34]],   [1,[16, 56]]                 ],#物品0 共计2城市
        [[2,[98, 23]]                                 ],#物品1 共计1城市
        [[3,[34, 15]],   [4,[79, 46]],  [5,[67,51]]   ],#物品2 共计3城市
        [[6,[86, 13]],   [7,[99, 46]]                 ],#物品3 共计2城市
        [[8,[47, 83]],                                ] #物品4 共计1城市                       
    ]

    citys_17_11=[
        # 城市编号 ， 城市坐标
        [[ 0,[78, 34]],   [ 1,[16, 56]]                  ],#物品0 共计2城市
        [[ 2,[98, 23]]                                   ],#物品1 共计1城市
        [[ 3,[34, 15]],   [ 4,[79, 46]],  [ 5,[67,51]]   ],#物品2 共计3城市
        [[ 6,[86, 13]],   [ 7,[99, 46]]                  ],#物品3 共计2城市
        [[ 8,[47, 83]],                                  ],#物品4 共计1城市
        [[ 9,[34, 55]],   [10,[ 9, 21]],                 ],#物品5 共计2城市
        [[11,[45, 45]],   [12,[100,34]],                 ],#物品6 共计2城市
        [[13,[88, 90]],                                  ],#物品7 共计1城市
        [[14,[ 4, 51]],                                  ],#物品8 共计1城市
        [[15,[73, 11]],                                  ],#物品9 共计1城市
        [[16,[56, 37]],                                  ],#物品10 共计1城市                      
    ]

    citys_24_15=[
        # 城市编号 ， 城市坐标
        [[ 0,[78, 34]],   [ 1,[16, 56]]                  ],#物品0 共计2城市
        [[ 2,[98, 23]]                                   ],#物品1 共计1城市
        [[ 3,[34, 15]],   [ 4,[79, 46]],  [ 5,[67,51]]   ],#物品2 共计3城市
        [[ 6,[86, 13]],   [ 7,[99, 46]]                  ],#物品3 共计2城市
        [[ 8,[47, 83]],                                  ],#物品4 共计1城市
        [[ 9,[34, 55]],   [10,[ 9, 21]],                 ],#物品5 共计2城市
        [[11,[45, 45]],   [12,[100,34]],                 ],#物品6 共计2城市
        [[13,[88, 90]],                                  ],#物品7 共计1城市
        [[14,[ 4, 51]],                                  ],#物品8 共计1城市
        [[15,[73, 11]],   [16,[26, 27]],                 ],#物品9 共计1城市
        [[17,[56, 37]],                                  ],#物品10 共计1城市
        [[18,[100,13]],  [19,[101,20]],                  ],#物品11 共计2城市
        [[20,[90, 87]],  [21,[90, 87]],                  ],#物品12 共计2城市
        [[22,[ 1,  1]],                                  ],#物品13 共计1城市
        [[23,[71, 33]],                                  ],#物品14 共计1城市                     
    ]

    citys_31_16=[
        # 城市编号 ， 城市坐标
        [[ 0,[78, 34]],   [ 1,[16, 56]]                  ],#物品0 共计2城市
        [[ 2,[98, 23]]                                   ],#物品1 共计1城市
        [[ 3,[34, 15]],   [ 4,[79, 46]],  [ 5,[67,51]]   ],#物品2 共计3城市
        [[ 6,[86, 13]],   [ 7,[99, 46]]                  ],#物品3 共计2城市
        [[ 8,[47, 83]],                                  ],#物品4 共计1城市
        [[ 9,[34, 55]],   [10,[ 9, 21]],                 ],#物品5 共计2城市
        [[11,[45, 45]],   [12,[100,34]],                 ],#物品6 共计2城市
        [[13,[88, 90]],                                  ],#物品7 共计1城市
        [[14,[ 4, 51]],   [15,[18, 52]],                 ],#物品8 共计2城市
        [[16,[73, 11]],   [17,[26, 27]],                 ],#物品9 共计2城市
        [[18,[56, 37]],   [19,[27, 70]],  [20,[16, 16]], ],#物品10 共计3城市
        [[21,[100,13]],   [22,[101,20]],                  ],#物品11 共计2城市
        [[23,[90, 87]],   [24,[90, 87]],                  ],#物品12 共计2城市
        [[25,[10, 10]],   [26,[ 1,  1]],                  ],#物品13 共计2城市
        [[27,[7, 13]],    [28,[17, 23]], [29,[71, 33]],   ],#物品14 共计3城市
        [[30,[100,25]],                                  ],#物品15 共计1城市                   
    ]

    citys_39_25=[
        # 城市编号 ， 城市坐标
        [[ 0,[78, 34]],   [ 1,[16, 56]]                  ],#物品0 共计2城市
        [[ 2,[98, 23]]                                   ],#物品1 共计1城市
        [[ 3,[34, 15]],   [ 4,[79, 46]],  [ 5,[67,51]]   ],#物品2 共计3城市
        [[ 6,[86, 13]],   [ 7,[99, 46]]                  ],#物品3 共计2城市
        [[ 8,[47, 83]],                                  ],#物品4 共计1城市
        [[ 9,[34, 55]],   [10,[ 9, 21]],                 ],#物品5 共计2城市
        [[11,[45, 45]],   [12,[100,34]],                 ],#物品6 共计2城市
        [[13,[88, 90]],                                  ],#物品7 共计1城市
        [[14,[ 4, 51]],   [15,[18, 52]],                 ],#物品8 共计2城市
        [[16,[73, 11]],   [17,[26, 27]],                 ],#物品9 共计2城市
        [[18,[56, 37]],   [19,[27, 70]],                 ],#物品10 共计2城市
        [[20,[100,13]],   [21,[120,20]],                 ],#物品11 共计2城市
        [[22,[90, 87]],   [23,[90, 87]],                 ],#物品12 共计2城市
        [[24,[10, 10]],                                  ],#物品13 共计1城市
        [[25,[7, 13]],                                   ],#物品14 共计1城市
        [[26,[ 90,75]],                                  ],#物品15 共计1城市
        [[27,[ 10,99]],                                  ],#物品16 共计1城市
        [[28,[103,105]],                                 ],#物品17 共计1城市
        [[29,[ 61,61]],    [30,[ 51,51]],                ],#物品18 共计2城市
        [[31,[100, 5]],                                  ],#物品19 共计1城市
        [[32,[ 50,25]],                                  ],#物品20 共计1城市
        [[33,[ 50,50]],                                  ],#物品21 共计1城市
        [[34,[ 10,20]],                                  ],#物品22 共计1城市
        [[35,[ 31,20]],  [36,[30,26]],  [37,[34,35]]     ],#物品23 共计3城市
        [[38,[ 48,25]],                                  ],#物品24 共计1城市
    ]


    citys_map={
        '9_5':      citys_9_5,
        '17_11':    citys_17_11,
        '24_15':    citys_24_15,
        '31_16':    citys_31_16,
        '39_25':    citys_39_25
    }


    def __init__(self,mode="9_5"):
        # 选择城市实例
        self.mode = mode
        self.citys=self.citys_map[mode]
        
        # 获得二级信息
        self.city_locations         =   self._get_city_list()           #城市坐标
        self.city_index_groups      =   self._get_city_index_group_by_item()   #城市序号分组
        self.distMat                =   self._get_distMat_mat()         #城市距离矩阵

    
    def _get_city_list(self):
        ''' 获得城市数组 按照序号进行排列 '''
        rst=[]
        for city_group in self.citys:
            for city in city_group:
                rst.append(city[1])
        return np.array(rst)
    
    def _get_city_index_group_by_item(self):
        ''' 获得根据物品来分类的一个城市序号'''
        rst=[]
        for city_group in self.citys:
            group=[]
            for city in city_group:
                group.append(city[0])
            rst.append(group)
        return rst
    
    def _get_distMat_mat(self):
        ''' 获得城市之间的距离 '''
        all_citys=self._get_city_list()
        c_num=len(all_citys)
        distMat=np.zeros((c_num,c_num))

        for i in range(c_num):
            for j in range(i,c_num):
                # 使用范式来求解距离
                distMat[i][j]=distMat[j][i]=np.linalg.norm(all_citys[i]-all_citys[j])
        
        return distMat
    
    def set_path(self,path):
        ''' 设定指定的路径 '''
        self.path       = path
        self.distance   = self.cal_path_distance(path)
    
    def set_random_path(self):
        ''' 设定随机的初始路径 '''
        path=[]
        for c_group in self.citys:
            # 随机选择该物品城市中的任意一个
            path.append(random.choice(c_group)[0])
        # 打乱访问顺序
        random.shuffle(path)

        # 设定路径
        self.set_path(path)

        return path

    
    def cal_path_distance(self,path):
        ''' 计算路径的距离 '''
        dis=0
        for i in range(len(path)-1):
            dis+=self.distMat[path[i]][path[i+1]]
        dis+=self.distMat[path[i+1]][path[0]]  # 回家
        return dis
    
    def plot_path(self):
        ''' 画出路径 '''

        # 画出城市坐标
        colors=['black',    'blue',    'green',    'red',  'yellow',   'orange',
                'purple',   'darkblue','lightblue','gold', 'lime',     'maroon',
                'olive',    'silver',   'orchid',   'salmon','tomato',  'yellowgreen',
                'rosybrown','plum',    'peru',     'tan',    'sienna',  'saddlebrown',
                'palevioletred'    
                ]
        for i,c_group in enumerate(self.citys):
            citys=np.array([x[1] for x in c_group])
            plt.scatter(citys[:,0],citys[:,1],color=colors[i])

        # 画出路径
        path=self.path
        for i in range(len(path)-1):
            x=[self.city_locations[path[i]][0],self.city_locations[path[i+1]][0]]
            y=[self.city_locations[path[i]][1],self.city_locations[path[i+1]][1]]
            plt.plot(x,y,color='grey',linestyle=":")
        x=[self.city_locations[path[0]][0],self.city_locations[path[i+1]][0]]
        y=[self.city_locations[path[0]][1],self.city_locations[path[i+1]][1]]
        plt.plot(x,y,color='grey',linestyle=":")
        plt.text(0,0,"Total distance:{:f}".format(self.distance))
        plt.title(self.mode)
        plt.show()

if __name__ == "__main__":
    mode,iteration='39_25',1000    #城市例子模式,迭代次数

    algorithms=[
        SA(2,1,0.9,GTSP(mode=mode),iteration//10),              #模拟退火
        TS(20,GTSP(mode=mode),iteration),                       #禁忌搜索
        GA(100,0.25,0.05,GTSP(mode=mode),iteration),            #遗传算法
        AC(20,1,1,0.3,1,GTSP(mode=mode),iteration)              #蚁群算法
    ]

    for algo in algorithms:
        tsp=algo.solve()            # 应用算法解决TSP问题
        algo.plot_iter_cost()       # 画出迭代曲线
        tsp.plot_path()             # 画出解的图   
    
