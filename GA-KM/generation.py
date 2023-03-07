from chromosome import Chromosome

'''
这段代码定义了一个Generation类，包括个体数量、染色体对象列表和代数等属性，以及对这些属性的操作方法。
其中，sortChromosomes()方法对染色体对象列表按照适应度从高到低排序；randomGenerateChromosomes()方法随机生成每个染色体对象。
'''


class Generation:
    def __init__(self, numberOfIndividual, generationCount):
        # 初始化Generation类
        self.numberOfIndividual = numberOfIndividual  # 个体数量
        self.chromosomes = []  # 一个列表，用于存储每个染色体对象
        self.generationCount = generationCount  # 代数

    def sortChromosomes(self):
        # 对染色体对象列表进行排序
        self.chromosomes = sorted(
            self.chromosomes, reverse=True, key=lambda elem: elem.fitness)
        return self.chromosomes

    def randomGenerateChromosomes(self, lengthOfChromosome):
        # 随机生成每个染色体对象
        for i in range(0, self.numberOfIndividual):
            chromosome = Chromosome([], lengthOfChromosome)  # 生成一个空列表的染色体对象
            chromosome.randomGenerateChromosome()  # 为该染色体对象随机生成基因
            self.chromosomes.append(chromosome)  # 将生成的染色体对象添加到染色体对象列表中
