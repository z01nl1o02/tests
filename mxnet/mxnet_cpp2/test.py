import os,sys,pdb
import numpy as np
import mxnet as mx
from mxnet.base import _LIB
import mxnet.ndarray as nd
import time
import ctypes
      

class CALC:
    def __init__(self):
        dllpath = os.path.join( os.path.dirname(__file__), 'src/build/release/calc.dll' )
        self.lib = None
        if os.path.exists(dllpath):
            self.lib = ctypes.CDLL(dllpath)
            self.fun_calc_sum = getattr(self.lib, 'c_calc_sum') #only support cpu due to no gpu implementation
        return
    def get_pointer(self,v):
        ptr = ctypes.c_void_p()
        _LIB.MXNDArrayGetData(v.handle, ctypes.byref(ptr))     
        return ptr
    def calc_sum(self,matA, matB):
        height,width = matA.shape
        ptrA = self.get_pointer(matA)
        ptrB = self.get_pointer(matB)
        matC = nd.zeros( matA.shape, ctx = matA.context)
        ptrC = self.get_pointer(matC)
        self.fun_calc_sum(ptrA, ptrB, ptrC, width, height)
        return matC
        
        
        
        
        


def calc_sum(matA, matB):
    height,width = matA.shape
    matC = nd.zeros( matA.shape, ctx=matA.context)
    for y in range(height):
        for x in range(width):
            matC[y,x] = matA[y,x] + matB[y,x]
    return matC




def main(ctx):
    calcEngine = CALC()
    
    tmp = np.asarray( [k for k in range(6)] )
    matA = nd.array( np.reshape( tmp ,(2,3) ) ).as_in_context( ctx )

    tmp = np.asarray( [k*10 for k in range(6)] )
    matB = nd.array( np.reshape( tmp, (2,3) ) ).as_in_context( ctx )

    
    num = 1000
    
    if 1:
        t0 = time.time()
        for k in range(num):
            matD = calcEngine.calc_sum(matA, matB)
        t1 = time.time() 
        print 'dll: time cost {}ms'.format( float(t1 - t0)*1000/num)
        print matD

    if 1:
        t0 = time.time()
        for k in range(num):
            matC = calc_sum(matA, matB)
        t1 = time.time() 
        print 'py: time cost {}ms'.format( float(t1 - t0)*1000/num)
        print matC

if __name__=="__main__":
    main( mx.cpu() )

