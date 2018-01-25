import mxnet as mx
import pdb,os,sys
import numpy as np
import sampleio

class FEATUREEXTRACT(object):
    def __init__(self):
        self.batch_size = 1
        return
    def load(self,model_prefix, model_epoch, data_layer_name, feature_layer_name, input_size):  
        C,W,H = input_size
        load_model = mx.model.load_checkpoint( model_prefix, model_epoch)
        context = [mx.cpu()]
        sym,arg_params,aux_params = load_model
        all_layers = sym.get_internals()
        feature_layer_sym = all_layers[feature_layer_name]

        self.input_size = input_size
        self.model = mx.mod.Module(symbol = feature_layer_sym, label_names = None, context = context)
        self.model.bind(for_training=False, data_shapes=[(data_layer_name, (self.batch_size,) + input_size)])
        self.model.set_params(arg_params, aux_params)
        return self
    def apply(self,indir,outfile):
        C,W,H = self.input_size
        dataIter = sampleio.SAMPLEIO().load(indir,self.batch_size,W,H).get_data_iter()    
        feats = []
        labels = []
        for batch in dataIter:
            self.model.forward(batch)
            label = dataIter.getlabel()[0].asnumpy()[0]
            mx_out = self.model.get_outputs()[0].asnumpy()
            feats.append(mx_out)
            labels.append(label)
        X = np.vstack( feats )
        Y = np.asarray( labels )
        
        lines = []
        for row in range(len(Y)):
            x = X[row,:].tolist()
            y = Y[row]
            line = [y]
            line.extend(x)
            line = ','.join( [ str(k) for k in line] )
            lines.append(line)
        with open(outfile,'wb') as f:
            f.writelines( '\r\n'.join(lines) )



if __name__=="__main__":
    extractor = FEATUREEXTRACT().load('model/colornaming',200,'data','fullyconnected0_output',(3,32,32))
    extractor.apply('train','train.txt')
    extractor.apply('test','test.txt')
