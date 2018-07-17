import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd
import cPickle
import numpy as np

def im2col(img,ks):
    ctx = img.context
    chNum, height, width=img.shape
    imgPadding = nd.pad(nd.expand_dims(img,axis=0),mode='constant',pad_width=(0,0,0,0,0,2,0,2),constant_value=0)[0]
    patchNum = width*height
    output = nd.zeros((patchNum, ks*ks*chNum), ctx=ctx)
    for y in range(imgPadding.shape[1] - 2):
        for x in range(imgPadding.shape[2] - 2):
            output[y*width + x] = imgPadding[:, y:y+ks, x:x+ks].reshape(ks*ks*chNum)
    return output


class Proto2D(mx.operator.CustomOp):
    def __init__(self, channels,kernelSize):
        self.kernelSize=kernelSize
        assert kernelSize==3
        self.strides = 1
        self.channels=channels
        self.verbose = False
        self.minDist = None
        self.projWeight = None
        return

    
    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        data = in_data[0]
        proj = aux[0]
        weight = in_data[1]


        if self.verbose:
            print 'forward input start'
            print 'data {} {} {}'.format(data.min(),data.max(),data.mean())
            print 'weight {} {} {}'.format(weight.min(), weight.max(), weight.mean())
            print 'forward input end'

        batchSize, inChNum, height, width = data.shape
        outChNum, inChNum, kernelSize, _ = weight.shape

        weightMat = nd.zeros((outChNum,width*height,inChNum * kernelSize * kernelSize),ctx=ctx)
        for outchidx in range(outChNum):
            w = nd.reshape(weight[outchidx],(1,-1))
            w = nd.tile( w, (width * height, 1) )
            weightMat[outchidx] = w

        output = nd.zeros((batchSize, outChNum, height, width), ctx = ctx)
        for batchidx in range(batchSize):
           dataCur=data[batchidx]
           dataCur=im2col(dataCur,kernelSize)
           for outchidx in range(outChNum):
               weightCur= weightMat[outchidx]
               df = ((dataCur - weightCur)**2).sum(axis=1) + 0.00001
               output[batchidx,outchidx] = nd.reshape(-1*nd.log(df),(height,width))
        
        if self.verbose:
            print 'forward output start'
            print 'output {} {} {}'.format(output.min(), output.max(), output.mean())
            print 'forward output end'

        if self.minDist is None:
            self.minDist = nd.zeros(outChNum,ctx = ctx) - 99999.0
            self.projWeight = nd.zeros(weight.shape, ctx=ctx)

        if proj[0] == 1: #start
            dataPading = nd.pad(data,mode='constant',pad_width=[0,0,0,0,0,2,0,2],constant_value=0)
            for batchidx in range(batchSize):
                dataCur =  dataPading[batchidx]
                for outchidx in range(outChNum):
                    locx,locy = 0, 0
                    for y in range(height):
                        for x in range(width):
                            if output[batchidx,outchidx][y,x] > output[batchidx,outchidx][locy,locx]:
                                locy,locx = y, x
                    if output[batchidx,outchidx][locy,locx] >  self.minDist[outchidx]:
                        self.projWeight[outchidx] = dataPading[batchidx][:,locy:locy+kernelSize, locx:locx+kernelSize]
        elif proj[0] == 2: #end
           for outchidx in range(outChNum):
               if self.minDist[outchidx] < -99:
                   continue
               weight[outchidx] = self.projWeight[outchidx]
           self.assign(in_data[2],"write",weight)
        self.assign(out_data[0],req[0],output)

        return

    def norm_grad(self,out_data_ch, out_grad_ch, inChNum):
        out = nd.exp((-1)*out_data_ch)
        out = out_grad_ch / out
        out = nd.pad( nd.expand_dims(nd.expand_dims(out,axis = 0), axis=0), mode = 'constant', pad_width=[0,0,0,0,2,2,2,2], constant_value=0)[0]
        out = nd.tile(out, (inChNum, 1, 1))
        return out

    def calc_grad_z(self, norm_grad, in_data, rot_weight, ctx):
        inChNum, height, width = in_data.shape
        _, ks, _ = rot_weight.shape
        dataMat = nd.zeros((width * height, inChNum, ks * ks), ctx=ctx )
        if 0:
            for y in range(height):
                for x in range(width):
                    val = nd.tile( nd.reshape(in_data[:, y, x],(inChNum, 1, 1)), (1, ks, ks) )
                    dataMat[y * width + x, :, ] = nd.reshape(val, (inChNum, ks * ks ))
        else:
            dataMat0 = nd.transpose( nd.reshape(in_data,(inChNum,-1)),(1,0))
            dataMat1 = nd.expand_dims(dataMat0,axis=-1)
            dataMat = nd.tile(dataMat1,(1,1,ks * ks))
        weightMat = nd.tile( nd.reshape(rot_weight,  (1, inChNum, ks*ks)), (width*height, 1, 1) )
        gradMat = nd.zeros((width * height, inChNum, ks * ks),ctx=ctx)
        for y in range(height):
            for x in range(width):
                gx, gy = x + 2, y + 2
                gradMat[y * width + x,:,] =  nd.reshape(norm_grad[:,gy-2:gy+1, gx-2:gx+1],(inChNum,ks*ks))
        output = (gradMat * (dataMat - weightMat)).sum(axis = 2)
        output = nd.reshape(output, (height, width, inChNum))
        output = nd.transpose(output, (2, 0, 1))
        return output

    def calc_grad_w(self, norm_grad, in_data, weight, ctx):
        inChNum, height, width = in_data.shape
        _, ks, _ = weight.shape
        dataPad = nd.pad(nd.expand_dims(in_data,axis=0),mode='constant',pad_width=[0,0,0,0,0,2,0,2], constant_value=0)[0]
        output = nd.zeros((inChNum,ks,ks),ctx=ctx)
        for y in range(ks):
            for x in range(ks):
                weightCur = nd.tile( nd.reshape(weight[:,y,x],(inChNum,1,1)), (1,height,width) )
                val = 2 * norm_grad[:, 2:-2, 2:-2] * ( dataPad[:,y:y+height,x:x+width] - weightCur  )
                output[:,y,x] = nd.reshape(val, (inChNum, width*height)).sum(axis=1)
        return output
    def get_R2(self,dataOut,l2): #second part of cost function
        batchSize, chNum, height, width = dataOut.shape
        val = nd.exp( (-1) * (nd.reshape(dataOut,(batchSize,-1)).max(axis=1) ) ).mean()
        return val * l2
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dataIn = in_data[0]
        dataOut = out_data[0]
        weight = in_data[1]
        lambdaR2 = 0.00005
        costR2 = self.get_R2(dataOut,lambdaR2)
        grad = out_grad[0] + costR2
        if self.verbose:
            print 'grad max = {} R2 = {}'.format( out_grad[0].max(), costR2 )
            print 'backward input start'
            print 'len of out_grad {}, len of in_grad {}'.format(len(out_grad), len(in_grad))
            print 'shape out_grad[0]:{} in_grad[0]:{} in_grad[1]:{}'.format(out_grad[0].shape, in_grad[0].shape, in_grad[1].shape)
            print 'backward input end'

        batchidx, inChNum, height, width = dataIn.shape
        batchSize, outChNum, _, _ = grad.shape
        outChNum, _, kernelSize, _ = weight.shape

        ctx = dataIn.context

        weightRot = nd.flip( nd.flip(weight, axis=3), axis=2 )

        dz = nd.zeros((batchSize, inChNum, height, width))
        dw = nd.zeros((outChNum, inChNum, kernelSize, kernelSize))

        for batchidx in range(batchSize):
            inDataCur = dataIn[batchidx]
            outDataCur = dataOut[batchidx]
            for outchidx in range(outChNum):
                weightCur = weight[outchidx]
                weightRotCur = weightRot[outchidx]
                normGrad = self.norm_grad(outDataCur[outchidx], grad[batchidx,outchidx], inChNum)
                # should be (w - z)
                dz[batchidx] -= self.calc_grad_z(normGrad, inDataCur, weightRotCur, ctx)
                dw[outchidx] -= self.calc_grad_w(normGrad, inDataCur, weightCur, ctx)
        if self.verbose:
            print 'backward output start'
            print 'grad input: {} {} {}'.format(grad.min(), grad.max(), grad.mean())
            print 'grad output: {} {} {}'.format(dz.min(), dz.max(),dz.mean())
            print 'dw output: {} {} {}'.format(dw.min(), dw.max(), dw.mean())
            print 'backward output end'
        self.assign(in_grad[0],req[0],dz)
        self.assign(in_grad[1],req[1],dw)
        return

@mx.operator.register("proto2d")
class Proto2DProp(mx.operator.CustomOpProp):
    def __init__(self,channels,kernelSize):
        super(Proto2DProp,self).__init__(need_top_grad=True)
        self.kernelSize = int(kernelSize)
        self.channels = int(channels)
    def list_arguments(self):
        return ["input","weight"]
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return ['project_action']
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        weight_shape = in_shape[1]
        output_shape = (data_shape[0],weight_shape[0],data_shape[2],data_shape[3]) #
        aux_shape = (1,)
        return (data_shape,weight_shape),(output_shape,),(aux_shape,)
    def infer_type(self, in_type):
        dtype = in_type[0]
        return (dtype,dtype),(dtype,),(np.int32,)
    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Proto2D(self.channels,self.kernelSize)

class Proto2DBlock(nn.Block):
    def __init__(self,in_channels, out_channels, kernel_size=3, **kwargs):
        super(Proto2DBlock,self).__init__(**kwargs)
        #configure parameters
        self.kernelSize = kernel_size
        self.channels = out_channels
        self.project_action = None
        self.ctx = None
        #learnable parameters
        self.weights = self.params.get("weight",shape = (out_channels, in_channels, kernel_size, kernel_size)) #define shape of kernel
        return
    @property
    def weight(self):
        return self.weights
    @weight.setter
    def weight(self,val):
        self.weights = val

    def set_project_action_code(self,code):
        if self.ctx is None:
            print 'run forward before projection'
            return
        self.project_action = nd.ones((1,),ctx=self.ctx,dtype=np.int32) * code
        return

    def forward(self,x, *args):
        ctx = x.context
        if self.ctx is None:
            self.ctx = ctx
            self.project_action = nd.ones((1,),ctx=ctx,dtype=np.int32) * (-1)
        y = mx.nd.Custom(x,self.weights.data(ctx), self.project_action, channels=self.channels, kernelSize=self.kernelSize, op_type="proto2d")
        return y


