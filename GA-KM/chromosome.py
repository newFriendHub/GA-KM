import random

'''
这段代码定义了一个染色体类Chromosome，包含了基因序列genes、基因序列长度length、适应度fitness，
以及随机生成染色体的函数randomGenerateChromosome。该函数的作用是随机生成一个长度为length的基因序列，将其封装成染色体对象并返回。
'''


class Chromosome:
    def __init__(self, genes, length):
        self.genes = genes  # 基因序列
        self.length = length  # 基因序列长度
        self.fitness = 0  # 适应度

    def randomGenerateChromosome(self):
        # 随机生成基因序列
        for i in range(0, self.length, +1):
            gen = float('%.2f' % random.uniform(0.0, 1.0))  # 生成一个介于0到1之间的浮点数
            self.genes.append(gen)

        return self  # 返回生成的染色体对象
