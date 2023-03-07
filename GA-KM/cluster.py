import json
import math

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

'''
这段代码定义了一个名为 Point 的类。每个 Point 对象表示一个数据点，其中包含了数据点所表示的模式序列 ID、
数据点所属聚类的编号和 Point 对象的长度等信息。
该类还定义了两个方法，str() 方法返回 Point 对象的模式序列 ID，toJSON() 方法以 JSON 格式返回 Point 对象的信息。
'''


class Point:
    def __init__(self, pattern_id):
        self.length = len(pattern_id)
        self.pattern_id = pattern_id  # 该数据点所表示的模式序列ID
        self.z = -1  # 所属聚类的编号，-1表示未指定

    def __str__(self):
        return str(self.pattern_id)

    def toJSON(self):
        return {
            'pattern_id': self.pattern_id
        }


class Cluster:
    def __init__(self, dim, centroid):
        self.dim = dim  # 聚类维数
        self.centroid = centroid  # 聚类的质心
        self.points = []  # 聚类中的数据点
        self.distances = []  # 每个数据点到聚类质心的距离

    # 计算聚类的S值，即所有数据点到质心的平均距离
    # this method finds the average distance of all elements in cluster to its centroid
    def computeS(self):
        n = len(self.points)
        if n == 0:
            return 0
        s = 0
        for x in self.distances:
            s += x
        return float(s / n)


class Clustering:
    # 构造函数，初始化聚类对象。传入参数有generation、data、kmax，分别表示迭代次数、数据和最大聚类数。
    def __init__(self, generation, data, kmax):
        # 构造函数，用于初始化实例变量
        # generation：表示聚类的迭代次数
        # data：表示聚类的数据集，这里使用的是Numpy数组格式
        # dim：表示数据集的维数，即数据集有多少个特征
        # penalty：表示类间惩罚项
        # kmax：表示最大的聚类数

        self.generation = generation
        self.data = data
        self.dim = data.shape[1]
        self.penalty = 1000000
        self.kmax = kmax

    # 计算Davies-Bouldin指数，用于评估聚类质量。传入参数为聚类结果clusters，返回值为DBIndex。
    def daviesBouldin(self, clusters):
        # 计算Davies-Bouldin指数
        # clusters：表示已经进行聚类后的簇
        sigmaR = 0.0
        # nc：表示聚类的数目
        nc = len(clusters)
        for i in range(nc):
            # 计算i簇的R值
            sigmaR = sigmaR + self.computeR(clusters)
            # print(sigmaR)
            # 计算Davies-Bouldin指数
        DBIndex = float(sigmaR) / float(nc)
        return DBIndex

    # 计算Rij，Rij用于计算Davies-Bouldin指数，代表聚类i和聚类j之间的距离。传入参数为聚类结果clusters，返回值为所有Rij的最大值。
    def computeR(self, clusters):
        # 计算两个簇之间的R值
        listR = []
        # 遍历所有簇，计算簇与其他簇的R值
        for i, iCluster in enumerate(clusters):
            for j, jCluster in enumerate(clusters):
                if (i != j):
                    # 计算i簇与j簇的Rij值
                    temp = self.computeRij(iCluster, jCluster)
                    # 添加到R列表中
                    listR.append(temp)
        # 返回最大的R值
        return max(listR)

    # 计算两个聚类iCluster和jCluster之间的Rij。传入参数为iCluster和jCluster，返回值为Rij。
    def computeRij(self, iCluster, jCluster):
        # 计算两个簇之间的Rij值
        Rij = 0

        # 计算两个簇之间的欧几里得距离
        d = self.euclidianDistance(
            iCluster.centroid, jCluster.centroid)
        # print("d",d)
        # print("icluster",iCluster.computeS())
        # 计算Rij值
        Rij = (iCluster.computeS() + jCluster.computeS()) / d

        # print("Rij:", Rij)
        # 返回Rij值
        return Rij

    # 计算两个点之间的欧几里得距离（即两点间的直线距离）。
    def euclidianDistance(self, point1, point2):
        sum = 0
        for i in range(0, point1.length):
            square = pow(
                point1.pattern_id[i] - point2.pattern_id[i], 2)  # 计算点之间对应位置的差的平方
            sum += square

        sqr = math.sqrt(sum)  # 对平方和进行开根号运算
        return sqr  # 返回欧氏距离

    # 计算数据集中每个数据点到k个簇中心的距离，并将数据点分配到最近的簇中。
    def calcDistance(self, clusters):
        kmax = self.kmax
        dim = self.dim
        data = self.data
        dis = 0
        disSet = []

        for z in range(data.shape[0]):  # 遍历数据集中每个点
            point = Point(data.loc[z][0:dim])  # 将数据集中每个点转化为Point对象
            point.z = z

            for i in range(kmax):  # 遍历每个聚类中心
                dis = self.euclidianDistance(clusters[i].centroid, point)  # 计算点到聚类中心的欧氏距离
                disSet.append(dis)  # 存储每个点与聚类中心的距离
                dis = 0

            clusters = self.findMin(
                disSet, clusters, point)  # 将点分配到距离最近的聚类中心中
            disSet = []  # clear disSet	# calculate distance  # 清空disSet

        return clusters  # 返回每个点所属的聚类

    # 找到距离数据点point最近的簇，并将该点添加到该簇中。其中，disSet是一个包含所有簇与数据点之间距离的列表，
    # clusters是一个Cluster类的列表，point是一个Point类的实例。
    def findMin(self, disSet, clusters, point):
        n = disSet.index(min(disSet))  # n is index  # 找到距离最近的聚类中心的索引n
        minDis = disSet[n]  # 获取距离最近的聚类中心的距离minDis
        clusters[n].points.append(point)  # 将点添加到距离最近的聚类中心中
        clusters[n].distances.append(minDis)  # 将距离minDis添加到距离最近的聚类中心的distances列表中

        return clusters  # 返回更新后的聚类中心

    # childChromosome, kmax
    # 根据每个子染色体的基因，构造对应的聚类中心，计算其聚类效果（使用Davies-Bouldin指数衡量），并将衡量结果作为个体适应度返回。
    # 计算某个子代的适应度
    def calcChildFit(self, childChromosome):
        kmax = self.kmax  # 聚类数
        dim = self.dim  # 数据维度
        clusters = []  # 空聚类列表
        # 遍历每个聚类中心点
        for j in range(kmax):
            # 根据个体染色体信息创建点对象
            point = Point(childChromosome.genes[j * dim: (j + 1) * dim])
            # 将点对象添加到聚类列表中
            clusters.append(Cluster(dim, point))

        # 计算聚类之间的距离并返回 Davies-Bouldin 指数
        clusters = self.calcDistance(clusters)
        DBIndex = self.daviesBouldin(clusters)

        # 将适应度设置为 DBI 的倒数
        childChromosome.fitness = 1 / DBIndex

        return childChromosome

    # 计算每个种群中所有染色体的适应度，并更新到对应的染色体。
    def calcChromosomesFit(self):
        kmax = self.kmax  # 聚类数
        generation = self.generation  # 当前种群
        numOfInd = generation.numberOfIndividual  # 种群中的个体数量
        data = self.data  # 数据
        chromo = generation.chromosomes  # 种群中的染色体列表

        # 遍历每个个体，计算其适应度
        for i in range(0, numOfInd):

            dim = self.dim  # 数据维度
            clusters = []  # 空聚类列表
            # 遍历每个聚类中心点
            for j in range(kmax):
                # 根据个体染色体信息创建点对象
                point = Point(chromo[i].genes[j * dim: (j + 1) * dim])
                # 将点对象添加到聚类列表中
                clusters.append(Cluster(dim, point))

            # 计算聚类之间的距离并返回 Davies-Bouldin 指数
            clusters = self.calcDistance(clusters)
            DBIndex = self.daviesBouldin(clusters)
            # 将适应度设置为 DBI 的倒数
            generation.chromosomes[i].fitness = 1 / DBIndex

        # 返回更新后的种群
        return generation

    # 给定一个染色体iBest，根据其中的基因构造聚类中心，并计算其在数据集上的聚类结果和准确率，同时打印出对应的聚类中心坐标和Davies-Bouldin指数。
    def printIBest(self, iBest):
        kmax = self.kmax  # 聚类数
        dim = self.dim  # 数据维度
        clusters = []  # 空聚类列表
        # 遍历每个聚类中心点
        for j in range(kmax):
            # 根据最优个体染色体信息创建点对象
            point = Point(iBest.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))

        clusters = self.calcDistance(clusters)  # 调用计算距离的函数计算各个聚类中心点之间的距离
        DBIndex = self.daviesBouldin(clusters)  # 计算戴维森堡丁指数
        z = (np.zeros(150)).tolist()  # 以 0 填充长度为 150 的列表 z
        for i, cluster in enumerate(clusters):  # 遍历聚类
            for j in cluster.points:  # 遍历聚类的所有点
                z[j.z] = i  # 将第 j 个点的类别设置为 i

        correct_answer = 0  # 正确分类数目初始化为 0
        for i in range(0, 50):  # 遍历前 50 个点
            if z[i] == 2:  # 如果聚类结果正确
                correct_answer += 1  # 正确分类数目加一
        for i in range(50, 100):  # 遍历 50 到 100 个点
            if z[i] == 1:  # 如果聚类结果正确
                correct_answer += 1  # 正确分类数目加一
        for i in range(100, 150):  # 遍历后 50 个点
            if z[i] == 0:  # 如果聚类结果正确
                correct_answer += 1  # 正确分类数目加一

        accuracy = (correct_answer / 150) * 100  # 计算分类准确率

        print("accuracy :", accuracy)  # 打印分类准确率
        print("iBest Fitness:", 1 / DBIndex)  # 打印最优个体的适应度
        print("all index:", z)  # 打印所有点的聚类结果
        print("Clusters centroid:")  # 打印所有聚类的中心点
        for i, cluster in enumerate(clusters):  # 遍历所有聚类
            print("centroid", i, " :", cluster.centroid)  # 打印第 i 个聚类的中心点


    '''
    它读入已经聚类好的最优个体iBest和原始数据data，然后通过最优个体iBest中的基因信息计算得到每个聚类的聚类中心，并将聚类结果保存在cluster_center.json中。
    同时，代码也将聚类结果合并到原始数据中，生成一个新的csv文件result.csv，其中每个数据点都被标记了它所属的聚类编号。
    '''


    def output_result(self, iBest, data):
        print("Saving the result...")  # 输出提示信息
        kmax = self.kmax  # 聚类数
        dim = self.dim  # 数据维度
        clusters = []  # 空聚类列表
        for j in range(kmax):
            point = Point(iBest.genes[j * dim: (j + 1) * dim])  # 根据最优个体的染色体信息，生成新的点
            clusters.append(Cluster(dim, point))  # 添加新的点到聚类列表

        clusters = self.calcDistance(clusters)  # 计算聚类间的距离
        centroids = []  # 聚类中心列表
        for i in range(kmax):
            centroids.append(clusters[i].centroid)  # 将每个聚类的中心点添加到聚类中心列表中
        z = (np.zeros(150)).tolist()  # 创建一个长度为150的全零列表
        for i, cluster in enumerate(clusters):  # 遍历聚类列表中的每个聚类
            for j in cluster.points:  # 遍历每个聚类中的点
                z[j.z] = i  # 将点的簇索引赋值给z列表对应位置

        # 将聚类中心的坐标信息存储为json格式
        with open('result/cluster_center.json', 'w') as outfile:
            json.dump([e.toJSON() for e in centroids], outfile, sort_keys=True,
                      indent=4, separators=(',', ': '))

        # 重命名dataframe的列名
        # rename df header
        col_name = list()
        for i in range(data.shape[1]):
            col_name.append("f{0}".format(i))
        data.columns = col_name

        # 添加聚类结果到dataframe中
        # insert cluster result
        data['Cluster Index'] = pd.Series(z, index=data.index)
        data.to_csv('result/result.csv', index=None)  # 将结果存储到csv文件中
        print("Done.")  # 输出提示信息
