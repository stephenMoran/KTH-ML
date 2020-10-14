import monkdata as m
import dtree as t
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#DATA
#training
m1 = m.monk1
m2 = m.monk2
m3 = m.monk3
#testing 
m1_test = m.monk1test
m2_test = m.monk2test
m3_test = m.monk3test

attr = m.attributes


#BUILD TREES
tree_m1 = t.buildTree(m1, m.attributes)
tree_m2 = t.buildTree(m2, m.attributes)
tree_m3 = t.buildTree(m3, m.attributes)


"""
Partitions code into training and test sets
"""
def partition(data, fraction): 
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(data) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

"""
Prunes the tree to find the best performance on a given dataset
"""
def prune(tree, valid_data):
    best_tree = tree 
    acc = t.check(best_tree, valid_data)
    updated = True
    while updated: 
        updated = False
        trees = t.allPruned(best_tree)
        for x in trees: 
            new_acc = t.check(x, valid_data)
            if new_acc >= acc: 
                acc = new_acc
                best_tree = x
                updated = True 
    return best_tree



def optimise(train, test): 
    mean = []
    spread = []
    for i in range(3,9,1):
        i = i * 0.1
        n = 1000
        err = []
        for j in range(n): 
            train_data, valid_data = partition(train, i)
            tree = t.buildTree(train_data, m.attributes)
            pruned_tree = prune(tree, valid_data)
            prune_err = 1 - t.check(pruned_tree, test)
            err.append(prune_err)
        mean.append(statistics.mean(err))
        spread.append(statistics.stdev(err))
    print (mean)
    print (spread)
    return mean, spread
    




#RESULTS
#MONK-1
m1_mean, m1_spread = optimise(m1, m1_test)

#MONK-3


#PLOTTING MEAN ERROR
#monk 1
m1_x = range(3,9,1) 
m1_x = np.array(m1_x)
m1_x = m1_x * 0.1
m1_y = np.array(m1_mean)
plt.plot(m1_x, m1_y, label='MONK-1')
#monk 3 
m3_mean, m3_spread = optimise(m3, m3_test)
m3_y = np.array(m3_mean)
plt.plot(m1_x, m3_y, label='MONK-3')

plt.xlabel('Dataset partitioning rate')
plt.ylabel('Mean Error')
plt.legend()
plt.show()


# PLOTTING SPREAD 
#monk 1
m1_y = np.array(m1_spread)
plt.plot(m1_x, m1_y, label='MONK-1')
m3_y = np.array(m3_spread)
plt.plot(m1_x, m3_y, label='MONK-3')

plt.xlabel('Dataset partitioning rate')
plt.ylabel('Standard Deviation of Error')
plt.legend()
plt.show()




