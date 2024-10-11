import numpy as np 


# default restrictions to improve performance 

def zero_func(x): 
    return 0 * x 

def one_func(x): 
    return 0 * x + 1

def neg_one_func(x): 
    return 0 * x - 1 

def two_func(x): 
    return 0 * x + 2 

# standard obstacle
def standard_obstacle_func(x): 
    return np.where(x < 0.25, 0, np.where(x > 0.75, 0, 0.8 + 0.1 * np.cos(20 * np.pi*x)))

# improved obstacle
def improved_obstacle_func(x): 
    return 0.5 - 2 * (x-0.5)**2 + 0.2 * (1 - np.cos(2 * np.pi * x**6)) + 0.15 * (1 - np.cos(2 * np.pi * x**(1/2))) + 0.1 * (1 - np.cos(20 * np.pi * x))

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
    u = np.where(x<0.5, 1, 0.7 + 0.3 * np.cos(np.pi * (2 + x * 4)))
    return u + 0.1 * np.sin(11 * np.pi * x) 

# not touched boundary 
def upper_not_touched_func(x): 
    return np.where(x < 0.45,  2, np.where(x < 0.55,  0.3,2)) 

if __name__ == '__main__': 
    # asdf
    
    print(0) 