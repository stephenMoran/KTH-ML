import random as r 
import numpy as np
import matplotlib.pyplot as plt
def get_data(N, seed, var): 
    np.random.seed(seed)
    half_n = int(N/2)
    quarter_n = int(N/4)
    total_n = half_n + (quarter_n * 2)
    #ensures that the same random data is generated each time
    np.random.seed(100)
    #randn parameters specify the shape of the vector
    classA = np.concatenate((np.random.randn(quarter_n,2) * var + [1.5, 1], 
        np.random.randn(quarter_n,2) * var + [-1.5, 1]))
    classB = np.random.randn(half_n, 2) * var + [0.0, -0.8]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    #shuffling data
    permute = list(range(total_n))
    r.shuffle(permute)
    #give us the first column in the order of permute and give us the second column as is
    inputs = inputs[permute,:]
    targets = targets[permute]
    #print(targets)
  
    """
    plt.plot([p[0] for p in classA], 
         [p[1] for p in classA], 
         'b+')

    plt.plot([p[0] for p in classB], 
            [p[1] for p in classB], 
            'ro')
    plt.axis('equal')
    plt.savefig('svmplot.pdf')
    plt.show()
    """
 
    


    return inputs, targets, total_n
