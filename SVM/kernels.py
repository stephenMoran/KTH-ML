import numpy as np

def linear_kernel(x,y): 
    return np.dot(x,y)

def poly_kernel(x,y,p): 
    return pow((np.dot(x,y) + 1), p)

def rbf_kernel(x,y, sigma): 
    return np.exp(-((pow(np.linalg.norm(x - y),2)) / (pow(2*sigma, 2))))
    

    