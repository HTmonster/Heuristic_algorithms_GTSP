'''
Author: Theo_hui
Date: 2020-12-15 10:30:50
Descripttion: 启发式算法的公共父类 
'''
import matplotlib.pyplot as plt

class Heuristic:

    # 每次迭代的花费
    costs=[]

    def plot_iter_cost(self):
        ''' 画出迭代的收敛曲线 '''

        x=list(range(len(self.costs)))
        y=self.costs

        plt.plot(x,y,color='b')
        plt.title(self.__class__.__name__)
        plt.xlabel('iters')
        plt.ylabel('distance')
        plt.show()