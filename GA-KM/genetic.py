import random

import numpy as np

from chromosome import Chromosome
from cluster import Clustering
from generation import Generation

random.seed(1)


class Genetic:
    # 初始化函数，接受聚类问题的参数。
    def __init__(self, numberOfIndividual, Ps, Pm, Pc, budget, data, generationCount, kmax):
        self.numberOfIndividual = numberOfIndividual  # 种群中个体的数量
        self.Ps = Ps  # 选择概率
        self.Pm = Pm  # 变异概率
        self.Pc = Pc  # 交叉概率
        self.budget = budget  # 预算限制
        self.data = data  # 输入数据
        self.generationCount = generationCount  # 当前遗传过程的迭代次数
        self.kmax = kmax  # 最大迭代次数

    # 执行一次遗传过程，包括选择、交叉和变异，返回一个新的种群。
    def geneticProcess(self, generation):
        budget = self.budget  # 预算限制
        Ps = self.Ps  # 选择概率
        Pm = self.Pm  # 变异概率
        Pc = self.Pc  # 交叉概率
        numOfInd = self.numberOfIndividual  # 种群中个体的数量

        print("------------Generation:",
              self.generationCount, "-----------------")  # 输出当前迭代次数
        generation.sortChromosomes()  # 对种群中个体进行排序

        # ------------------------简单排序选择-------------------------

        generation = self.selection(generation)  # 选择

        #  ------------------------------Crossover---------------------------------

        generation = self.crossover(generation)  # 交叉

        #  ------------------------------Mutation---------------------------------

        generation = self.mutation(generation)  # 变异

        self.generationCount += 1  # 更新迭代次数
        return generation, self.generationCount  # 返回新的种群和更新后的迭代次数

    # 选择操作，将表现差的一部分个体替换成表现好的个体。
    def selection(self, generation):
        numOfInd = self.numberOfIndividual
        Ps = self.Ps

        # 替换最差的 Ps*numOfInd 个体为最好的 Ps*numOfInd 个体
        for i in range(0, int(Ps * numOfInd)):
            generation.chromosomes[numOfInd -
                                   1 - i] = generation.chromosomes[i]

        # 排序个体，以便后续操作
        generation.sortChromosomes()
        return generation

    # 交叉操作，按一定概率选择个体进行交叉操作。
    def crossover(self, generation):
        numOfInd = self.numberOfIndividual
        Pc = self.Pc

        # 随机选择要进行交叉操作的个体
        index = random.sample(
            range(0, numOfInd - 1), int(Pc * numOfInd))

        # 进行交叉操作
        for i in range(int(len(index) / 2), +2):  # 进行多少次交叉操作
            generation = self.doCrossover(
                generation, i, index)

        # 排序个体，以便后续操作
        generation.sortChromosomes()

        return generation

    # 执行一次交叉操作，随机选择两个个体进行交叉。
    def doCrossover(self, generation, i, index):

        # 获取种群染色体列表及长度
        chromo = generation.chromosomes
        length = chromo[0].length
        # 随机选择交叉点位置
        cut = random.randint(1, length - 1)
        # 获取两个亲本染色体
        parent1 = chromo[index[i]]
        parent2 = chromo[index[i + 1]]
        # 生成两个后代染色体
        genesChild1 = parent1.genes[0:cut] + parent2.genes[cut:length]
        genesChild2 = parent1.genes[cut:length] + parent2.genes[0:cut]
        child1 = Chromosome(genesChild1, len(genesChild1))
        child2 = Chromosome(genesChild2, len(genesChild2))

        # ----聚类----
        clustering = Clustering(generation, self.data, self.kmax)
        child1 = clustering.calcChildFit(child1)
        child2 = clustering.calcChildFit(child2)
        # -------------------

        listA = []
        listA.append(parent1)
        listA.append(parent2)
        listA.append(child1)
        listA.append(child2)
        # 根据适应度对父代和后代染色体排序
        listA = sorted(listA, reverse=True,
                       key=lambda elem: elem.fitness)

        # 更新种群中的染色体
        generation.chromosomes[index[i]] = listA[0]
        generation.chromosomes[index[i + 1]] = listA[1]

        return generation

    # 变异操作，将一些基因突变，以增加种群的多样性。
    def mutation(self, generation):
        numOfInd = self.numberOfIndividual  # 个体数量
        fitnessList = []  # 存放适应度列表
        generationAfterM = Generation(numOfInd, generation.generationCount)   # 存放变异后下一代个体的种群
        flagMutation = (np.zeros(numOfInd)).tolist()  # 记录哪些个体已经被变异过的标志位

        # 记录当前种群的每个个体的适应度
        for i in range(numOfInd):
            temp = generation.chromosomes[i]
            fitnessList.append(temp.fitness)

        for i in range(numOfInd):
            if i == 0:  # Ibest doesn't need mutation  # Ibest 不需要变异，直接复制到下一代
                generationAfterM.chromosomes.append(generation.chromosomes[0])
                flagMutation[0] = 0
            else:  # 对其他个体进行变异
                generationAfterM = self.doMutation(
                    generation.chromosomes[i], generationAfterM, flagMutation, fitnessList, i)

        # 对下一代个体进行排序
        generationAfterM.sortChromosomes()
        return generationAfterM

    # 执行一次变异操作，随机选择一些基因进行变异。
    def doMutation(self, chromosomeBeforeM, generationAfterM, flagMutation, fitnessList, i):
        Pm = self.Pm  # 突变概率
        dice = []  # 存储每个基因是否突变的概率
        length = len(chromosomeBeforeM.genes)
        chromosome = Chromosome([], length)  # 新个体的染色体
        geneFlag = []  # 存储每个基因是否突变

        # 根据突变概率和随机数来判断每个基因是否突变
        for j in range(length):
            dice.append(float('%.2f' % random.uniform(0.0, 1.0)))
            if dice[j] > Pm:
                chromosome.genes.append(chromosomeBeforeM.genes[j])
                geneFlag.append(0)

            if dice[j] <= Pm:
                chromosome.genes.append(
                    float('%.2f' % random.uniform(0.0, 1.0)))
                geneFlag.append(1)

        check = sum(geneFlag)  # 统计有多少基因突变

        if check == 0:  # 没有基因突变
            flagMutation[i] = 0
            chromosome.fitness = fitnessList[i]  # 新个体的适应度等于原个体的适应度
        else:  # 有基因突变
            flagMutation[i] = 1

            # ---clustering----
            clustering = Clustering(chromosomeBeforeM, self.data, self.kmax)  # 对新个体进行聚类
            chromosome = clustering.calcChildFit(
                chromosome)
            # ------------------

        generationAfterM.chromosomes.append(chromosome)  # 添加新个体到下一代
        return generationAfterM
