import numpy as np
import math, random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import gen_data
import matplotlib.pyplot as plt
from matplotlib.widgets import *

global var, C
var = 0.3
C = 10
sigma = 1
p_ker = 2


#Kernel functions
def linear_kernel(x,y): 
    return np.dot(x,y)

def poly_kernel(x,y): 
    return pow((np.dot(x,y) + 1), p_ker)

def rbf_kernel(x,y): 
    return np.exp(-((pow(np.linalg.norm(x - y),2)) / (pow(2*sigma, 2))))


#Helper functions
def pre_compute():
    p = np.empty((N,N))
    for i in range(N):                 
        for j in range(N): 
            res = targets[i] * targets[j]
            res = res * ker(inputs[i], inputs[j])
            p[i][j] = res
    return p


def objective(alpha): 
    a = alpha
    res = 0
    for i in range(N):
        for j in range(N): 
            res += a[i] * a[j] * p[i][j]
    return (res * 0.5) - np.sum(a)

def zerofun(alpha): 
    return np.dot(alpha, targets)

def non_zero(alpha): 
    f1 = (0.00001 < alpha)
    non_zalpha = alpha[f1]
    non_zinput = inputs[f1]
    non_ztargets = targets[f1]
    f2 = (non_zalpha < C)
    non_zalpha = non_zalpha[f2]
    non_zinput = non_zinput[f2]
    non_ztargets = non_ztargets[f2]
 
    non_z = {'a':non_zalpha, 'x':non_zinput, 't':non_ztargets}
    return non_z

def compute_b(non_z):
    b = 0
    a = non_z['a']
    x = non_z['x']
    t = non_z['t']
    n = a.size
    #change n because not all points are included in this array 
    if n > 0: 
        for i in range(n): 
            if a[i] < C: 
                for j in range(n): 
                    b += a[j] * t[j] * ker(x[i], x[j])
                b = b - t[i]
                return b
    return b

    
def indicator(input, non_z, b): 
    res = 0
    a = non_z['a']
    x = non_z['x']
    t = non_z['t']
    n = a.size
    if n > 0:
        for i in range(n): 
            res += a[i] * t[i] * ker(input, x[i])
        res = res - b
    return res

#globals
ker = linear_kernel
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
nz = 0
num_nz = 0


#MAIN
def compute(): 
    global inputs, targets, p, N
    N = 40
    inputs, targets, N = gen_data.get_data(N, 100, var)
    #pre-compute matrix 
    p = pre_compute()

    #Minimizing alpha
    start = np.zeros(N)
    bounds = [(0,C) for b in range(N)]
    constraint = {'type':'eq', 'fun':zerofun}
    ret = minimize(objective, start, bounds=bounds, constraints=constraint)
    #minimize uses a dictionary so we must use x to pick out the values of a 
    alpha = ret['x']
    success = ret['success']
    print("Success in finding minimum alpha: " + str(success))

    non_z = non_zero(alpha)
    num_nz = len(non_z['a'])
    ax.clear()
    b  = compute_b(non_z)
    #testing  
    test_inputs, test_targets, N = gen_data.get_data(N, 10, var)

    """
    predictions = np.zeros(N)
    for i in range(N): 
        predictions[i] = indicator(inputs[i], non_z, b)
    """ 

    #PLOTTING
    #data point plot
    filter_a = (targets == 1)


    class_a = inputs[filter_a]
    filter_b = (targets == -1)
    class_b = inputs[filter_b]


    ax.plot([p[0] for p in class_a], 
            [p[1] for p in class_a], 
            'b.')

    ax.plot([p[0] for p in class_b], 
            [p[1] for p in class_b], 
            'r.')


    #decision boundary plot
    x_grid = np.linspace(-5,5)
    y_grid = np.linspace(-4,4)

    grid = np.array([[indicator((x,y), non_z, b) for x in x_grid] for y in y_grid])

    ax.contour(x_grid, y_grid, grid, 
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidth=(1,3,1))

    #support vector plotting 

    ax.plot([sv[0] for sv in non_z['x']], [sv[1] for sv in non_z['x']], 'g+')

    plt.axis('equal')
    plt.savefig('svmplot.pdf')
    plt.show()

#GUI

gui_objects = []


def changeC(cc):
    global C
    C = cc
    compute()

def changeNZ(nzp):
    global nz
    nz = nzp
    compute()

def changeVar(v):
    global var
    var = v
    compute()

def changeSig(s):
    global sigma
    sigma = s
    compute()

def changeP(p_new):
    global p_ker
    p_ker = p_new
    compute()

def kernel(val):
    print("kernel")
    global ker
    if val == 'linear':
        ker = linear_kernel
    elif val == 'polynomial':
        ker = poly_kernel
    else:
        ker = rbf_kernel
    compute()



def draw_gui():
    gui_objects.clear()

    sfreq_b = Slider(plt.axes([0.25, 0.1, 0.2, 0.03]), 'C', 0.1, 10, valinit=10, valstep=0.1)
    sfreq_b.on_changed(changeC)
    gui_objects.append(sfreq_b)

    #sfreq_u = Slider(plt.axes([0.25, 0.05, 0.2, 0.03]), 'NonZeroPt', 0, num_nz, valinit=0, valstep=1)
    #sfreq_u.on_changed(changeNZ)
    #gui_objects.append(sfreq_u)
    sfreq_v = Slider(plt.axes([0.25, 0.05, 0.2, 0.03]), 'Data variance', 0.1, 1, valinit=0.2, valstep=0.1)
    sfreq_v.on_changed(changeVar)
    gui_objects.append(sfreq_v)

    sfreq_s = Slider(plt.axes([0.25, 0.15, 0.2, 0.03]), 'Sigma', 0.1, 1, valinit=0.2, valstep=0.1)
    sfreq_s.on_changed(changeSig)
    gui_objects.append(sfreq_s)

    sfreq_p = Slider(plt.axes([0.25, 0.20, 0.2, 0.03]), 'P', 1, 10, valinit=2, valstep=1)
    sfreq_p.on_changed(changeP)
    gui_objects.append(sfreq_p)

    rax = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio2 = RadioButtons(rax, ('linear', 'polynomial', 'rbf'))
    radio2.on_clicked(kernel)
    gui_objects.append(radio2)

plt.ion()
compute()
draw_gui()
input()

"""
1. Move clusters - note when the minimise function no longer works

    - Multiplying by a higher number increases the varinace of the cluster making them harder to seperate
    - When the data is not linearly seperabale then this SVM fails
        - this can happen when the cluster centres are too close 
        - or if teh variance increases too much 
    - moving the centre of the cluster closer to the centre of another class makes it harder to seperate the classes i.e. maximise the margin 


2. - show the use of non-linear kernels - radial and poynomial


3. Hyparamaters of the kernel functions
    - polynomial 
        - high p = higher variance and less bias
    -Radial 
        - higher sigma = higher bias and less variance 

4. C is the penalty strength - how hard do we lenalise points that fall within the margins ? 
    - this allows us to prevent overfitting 
    - low C values dont penalise much so the margins will be larger 
    - high C values puts a high penalty on these points which reduces the margins

    -  slack variables are those that fall inside the margins - the error that we allow so happen so that the model accoutns for outliers 
            -  the level of slack we allow is determined by C 
5. 


"""
