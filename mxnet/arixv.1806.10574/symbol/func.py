import os,sys,pdb
import ctypes
from mxnet.base import _LIB

class CHLEPDLL:
    def __init__(self):
        libpath = os.path.join( os.path.dirname(__file__), '../build/release/chelp.dll')
        self.lib = ctypes.CDLL(libpath)
        return
theLIB = CHLEPDLL()

def get_pointer(v):
    ptr = ctypes.c_void_p()
    _LIB.MXNDArrayGetData(v.handle, ctypes.byref(ptr))
    return ptr

class CHLEPFUNC:
    def __init__(self, name):
        self.name = name
        self.func = getattr(theLIB.lib, self.name)
        return
    def __call__(self, *args, **kwargs):
        dev_id, in_mat, out_mat = args
        channels, height, width = in_mat.shape
        in_ptr = get_pointer(in_mat)
        out_ptr = get_pointer(out_mat)
        self.func(dev_id, in_ptr, channels, height, width, out_ptr)
        return

patch2col = CHLEPFUNC('patch2col')

if 0: # testing
    import mxnet as mx
    from mxnet import ndarray as nd
    import numpy as np
    a = np.random.random((4,8,8))
    a = nd.array(a, ctx = mx.cpu())
    b = nd.zeros((6*6,4*3*3),ctx=a.context,dtype=np.float32)
    patch2col(0,a,b)
    print a
    print b


    c = nd.zeros((6,6),ctx=a.context,dtype=np.float32)



