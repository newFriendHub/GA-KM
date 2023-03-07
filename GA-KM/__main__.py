import configparser

import numpy as np
import pandas as pd

from cluster import Clustering
from generation import Generation
from genetic import Genetic

NORMALIZATION = True


# 定义读取配置文件的函数
def readVars(config_file):
    # 创建一个 ConfigParser 对象
    config = configparser.ConfigParser()
    # 读取配置文件
    config.read(config_file)
    # 提取配置文件中的变量
    budget = int(config.get("vars", "budget"))  # 聚类的预算
    kmax = int(config.get("vars", "kmax"))  # 最大聚类数
    numOfInd = int(config.get("vars", "numOfInd"))  # 种群中的个体数
    Ps = float(config.get("vars", "Ps"))  # 选择压力
    Pm = float(config.get("vars", "Pm"))  # 变异概率
    Pc = float(config.get("vars", "Pc"))  # 交叉概率

    # 返回提取的变量
    return budget, kmax, Ps, Pm, Pc, numOfInd


'''
这段代码实现了最小-最大归一化，也称为离差标准化。其目的是将数据缩放到一个特定的范围，通常是[0, 1]或[-1, 1]。
这有助于消除不同特征之间的量纲差异，使得不同特征可以在同一尺度上进行比较。
具体实现方法是将每个特征的取值范围映射到[0, 1]或[-1, 1]之间，具体做法是对每个特征进行如下变换：

x_norm = (x - min) / (max - min)

其中x是原始数据，min和max是该特征的最小值和最大值，x_norm是归一化后的数据。

这段代码中，首先将原始数据复制到normData中，然后将数据类型转换为float。
接下来，对每一列进行归一化操作，通过np.amax()和np.amin()函数获取每一列的最大值和最小值，并对每个数据进行归一化，
最后将归一化后的数据保存到csv文件中。

需要注意的是，这段代码的实现方式比较简单，但在数据量较大时，可能会出现性能问题。
可以考虑使用numpy的向量化操作来提高性能。另外，在进行归一化时，需要注意除数为0的情况，可以添加一些特殊处理，
比如将最大值和最小值之差加上一个较小的数，避免除数为0的情况。
'''


# 最小最大归一化
def minmax(data):
    # 复制原始数据
    normData = data
    # 将数据类型转换为float
    data = data.astype(float)
    normData = normData.astype(float)
    # 针对每一列进行归一化
    for i in range(0, data.shape[1]):
        tmp = data.iloc[:, i]
        # 获取每一列的最大值和最小
        maxElement = np.amax(tmp)
        minElement = np.amin(tmp)

        # 对每个数据进行归一化
        # norm_dat.shape[0] : size of row
        for j in range(0, normData.shape[0]):
            normData[i][j] = float(
                data[i][j] - minElement) / (maxElement - minElement)

    # 将归一化后的数据保存到csv文件中
    normData.to_csv('result/norm_data.csv', index=None, header=None)
    return normData


if __name__ == '__main__':
    config_file = "config.txt"
    if (NORMALIZATION):
        # 如果需要归一化，则读取数据并进行归一化处理
        data = pd.read_csv('data/iris.csv', header=None)
        data = minmax(data)  # 归一化
    else:
        # 如果不需要归一化，则读取已归一化数据
        data = pd.read_csv('result/norm_data.csv', header=None)
    # 数据维度（即特征数量）
    dim = data.shape[1]

    # K-Means参数和遗传算法参数
    generationCount = 0
    budget, kmax, Ps, Pm, Pc, numOfInd = readVars(config_file)

    print("-------------GA Info-------------------")
    print("budget", budget)
    print("kmax", kmax)
    print("numOfInd", numOfInd)
    print("Ps", Ps)
    print("Pm", Pm)
    print("Pc", Pc)
    print("---------------------------------------")

    # 染色体长度为kmax * dim，即每个簇负责一个维度
    # dim or pattern id 
    chromosome_length = kmax * dim

    # -------------------------------------------------------#
    # 							main 						#
    # -------------------------------------------------------#
    initial = Generation(numOfInd, 0)
    initial.randomGenerateChromosomes(
        chromosome_length)  # 随机生成初始种群

    clustering = Clustering(initial, data, kmax)  # 评估种群的适应度

    # ------------------计算适应度------------------#
    generation = clustering.calcChromosomesFit()

    # ------------------------遗传算法----------------------#
    while generationCount <= budget:
        GA = Genetic(numOfInd, Ps, Pm, Pc, budget, data, generationCount, kmax)
        generation, generationCount = GA.geneticProcess(
            generation)
        iBest = generation.chromosomes[0]
        clustering.printIBest(iBest)

    # ------------------输出结果-------------------#
    clustering.output_result(iBest, data)
