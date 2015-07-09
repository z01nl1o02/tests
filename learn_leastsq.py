import numpy as np
from scipy.optimize import leastsq

def fun(x, p):
    a,b = p
    return a * x + b

def residuals(p,x,y):
    prd = np.zeros_like(y)
    for k in range(len(y)):
        prd[k] = fun(x[k],p)
    return prd - y

x1 = np.array([1,2,3,4,5,6], dtype=float)
y1 = np.array([3,5,7,9,11,13], dtype=float)

#residuals : function with at least one parameter(the value wanted)
#            input/output should be array
#[1,1]: initial value for the wanted value
#args: others parameters for residuals() in order
r = leastsq(residuals, [1,1], args=(x1,y1))

print r[0]
print r[1]


