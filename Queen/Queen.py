# Copyright (c) 2021 郎督 版权所有
#
# 文件名：Queen.py
# 功能描述：遗传算法求解8皇后问题
#
# 作者：郎督
# 时间：2021年5月22日
#
# 版本：V1.0.0

import numpy as np
import random

def initialize(population_size, g_number=8):
    population = []
    for i in range(population_size):
        p = []
        while len(p) < g_number:
            r = random.randint(1, g_number)
            if r not in p:
                p.append(r)
        population.append(p)
    return population


def fitnessFunction(population):
    g_number = len(population[0])
    fitness = []    
    for i in range(len(population)):
        f = 28  
        for j in range(0, 7):
            for k in range(j+1, g_number):
                if population[i][j] == population[i][k]:    
                    f -= 1
                elif abs(population[i][j] - population[i][k]) == abs(j-k):  
                    f -= 1
        fitness.append(f)
    return fitness

def selectParent(population, number_parent=2, sample=5):
    rand_population = random.sample(population, sample) 
    rand_fitness = fitnessFunction(rand_population)     

    rand_sample = [(rand_f, rand_p) for rand_f, rand_p in zip(rand_fitness, rand_population)]
    rand_sample.sort(reverse=True, key=lambda x:x[0])
    parents = []
    for i in range(number_parent):
        parents.append(rand_sample[i][1])
    return parents

def recombination_insert(child_head, child_tail, child_implement, child_length=8):
    child = child_head
    for c in child_tail:
        if c not in child:
            child.append(c)
    if len(child) == child_length:
        return child
    for c in child_implement:
        if c not in child:
            child.append(c)
    return child

def recombination(parents, number_offspring=2):
    assert len(parents) == 2, 'recombination 目前仅支持 parents 包含2个元素'
    parent1, parent2 = parents
    len_parent = len(parent1)
    rand_point = random.randint(0, len_parent - 1)

    child1_head = parent1[0:rand_point]
    child2_head = parent2[0:rand_point]
    child1_tail = parent2[rand_point:len(parent2)]
    child2_tail = parent1[rand_point:len(parent1)]

    child1 = recombination_insert(child1_head, child1_tail, child2_head)
    child2 = recombination_insert(child2_head, child2_tail, child1_head)
    assert len(child1) == len(child2) == len(parent1), "recombination error"
    return [child1, child2]

def mutation(children, mutation_prob=0.8):
    for child in children:
        if random.random() > mutation_prob: # 不变异
            continue
        randindex = random.sample(range(0, len(child)), 2)
        child[randindex[0]], child[randindex[1]] = child[randindex[1]], child[randindex[0]]
    return children

def survivalSelection(population, children, force_replace=True):
    population_fitness = fitnessFunction(population)
    children_fitness = fitnessFunction(children)
    population_data = [(p_f, p) for p_f, p in zip(population_fitness, population)]
    children_data = [(c_f, c) for c_f, c in zip(children_fitness, children)]
    population_data.sort(key=lambda x:x[0]) # 从小到大排列
    children_data.sort(key=lambda x: x[0])

    if force_replace:
        # 强制取代
        population_data[0:len(children_data)] = children_data
    else:
        # 非强制取代
        pass

    population = [p[1] for p in population_data]
    return population


def evolve(population, max_iter=10000):
    for iter in range(max_iter):
        parents = selectParent(population)

        children = recombination(parents)

        children = mutation(children)

        population = survivalSelection(population, children, force_replace=True)
    return population


def main():
    population = initialize(5)
    population = evolve(population)
    population_fitness = fitnessFunction(population)

    population_data = [(p_f, p) for p_f, p in zip(population_fitness, population)]
    population_data.sort(reverse=True, key=lambda x: x[0])  # 从小到大排列
    # 打印最好的5个结果
    print(population_data[0:5])


if __name__ == '__main__':
    main()
