import numpy as np 
import matplotlib.pyplot as plt

# geometric upper lower 
def lower_geometric_func(x):
    l = np.where(x  < 8/129,   0, np.where(x < 12/129,  0.6,0+x*0))
    l += np.where(x < 24/129,  0, np.where(x < 28/129,  0.2,0+x*0))
    l += np.where(x < 40/129,  0, np.where(x < 44/129,  0.2,0+x*0))
    l += np.where(x < 56/129,  0, np.where(x < 60/129,  0.6,0+x*0))
    l += np.where(x < 72/129,  0, np.where(x < 76/129,  0.7,0+x*0))
    l += np.where(x < 88/129,  0, np.where(x < 92/129,  0.2,0+x*0))
    l += np.where(x < 104/129, 0, np.where(x < 108/129, 0.4,0+x*0))
    l += np.where(x < 120/129, 0, np.where(x < 124/129, 0.6,0+x*0))
    return l

# geometric upper lower 
def lower_geometric_func2(x):
    l = np.where(x  < 8/129,   0, np.where(x < 12/129,  0.6,0+x*0))
    l += np.where(x < 24/129,  0, np.where(x < 28/129,  0.2,0+x*0))
    l += np.where(x < 40/129,  0, np.where(x < 44/129,  0.2,0+x*0))
    l += np.where(x < 56/129,  0, np.where(x < 60/129,  0.6,0+x*0))
    l += np.where(x < 72/129,  0, np.where(x < 76/129,  0.6,0+x*0))
    l += np.where(x < 88/129,  0, np.where(x < 92/129,  0.2,0+x*0))
    l += np.where(x < 104/129, 0, np.where(x < 108/129, 0.4,0+x*0))
    l += np.where(x < 120/129, 0, np.where(x < 124/129, 0.6,0+x*0))
    return l

# geometric upper lower 
def lower_sides_geometric_func(x):
    l = np.where(x  < 8/129,   0, np.where(x < 12/129,  0.6,0+x*0))
    l += np.where(x < 24/129,  0, np.where(x < 28/129,  0.2,0+x*0))
    l += np.where(x < 40/129,  0, np.where(x < 44/129,  0.2,0+x*0))
    l += np.where(x < 56/129,  0, np.where(x < 60/129,  0.2,0+x*0))
    l += np.where(x < 72/129,  0, np.where(x < 76/129,  0.2,0+x*0))
    l += np.where(x < 88/129,  0, np.where(x < 92/129,  0.2,0+x*0))
    l += np.where(x < 104/129, 0, np.where(x < 108/129, 0.4,0+x*0))
    l += np.where(x < 120/129, 0, np.where(x < 124/129, 0.6,0+x*0))
    return l

def upper_sin_func(x): 
    u = 1.1 * np.where(x <= 1,  0.7 + 0.3 * np.cos(np.pi * (2 + x * 4)), np.where(x < 0.7, 1, np.where(x < 0.7, 2,2)))
    return u + 0.1 * np.sin(11 * np.pi * x) - 0.1
    
def half_upper_sin_func(x): 
    u = 1.1 * np.where(x<0.5, 1, 0.7 + 0.3 * np.cos(np.pi * (2 + x * 4)))
    return u + 0.1 * np.sin(11 * np.pi * x) - 0.1 

if __name__ == '__main__':
    x = np.linspace(0,1,1001)
    plt.figure()
    plt.plot(x,lower_geometric_func(x))
    plt.plot(x,lower_geometric_func2(x))
    plt.plot(x,lower_sides_geometric_func(x))
    plt.figure()
    plt.plot(x,upper_sin_func(x))
    plt.plot(x,half_upper_sin_func(x))
    plt.show()