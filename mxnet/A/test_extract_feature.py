import mxnet as mx
import pdb,os,sys
import numpy as np

MODELS_URL_ROOT = 'wxnet-models'
MODEL_NAME = 'squeezenet_v1.1'
MODEL_EPOCH = 0


DATA_LAYER = 'data'
OUTPUT_LAYER = 'prob_output'
LAYER_LAYER = 'prob_label'
FEATURE_LAYER = 'flatten_output'
INPUT_SIZE = (3,277,277)
load_model = mx.model.load_checkpoint( os.path.join(MODELS_URL_ROOT,MODEL_NAME), MODEL_EPOCH)
context = [mx.cpu()]


sym,arg_params,aux_params = load_model
all_layers = sym.get_internals()
feature_layer_sym = all_layers[FEATURE_LAYER]


batch_size = 1
image_shape = (3,227,227)
model = mx.mod.Module(symbol = feature_layer_sym, label_names = None, context = context)
model.bind(for_training=False, data_shapes=[(DATA_LAYER, (batch_size,) + image_shape)])
model.set_params(arg_params, aux_params)

dataIter = mx.io.ImageRecordIter(
        path_imgrec="catdog.rec",
        data_shape=image_shape,
        batch_size = batch_size,
        round_batch = False
        )
feats = []
labels = []
for batch in dataIter:
    model.forward(batch)
    label = dataIter.getlabel().asnumpy()[0]
    mx_out = model.get_outputs()[0].asnumpy()
    feats.append(mx_out)
    labels.append(label)
X = np.vstack( feats )
Y = np.asarray( labels )
print X.shape, Y.shape

lines = []
for row in range(len(Y)):
    x = X[row,:].tolist()
    y = Y[row]
    line = [y]
    line.extend(x)
    line = ','.join( [ str(k) for k in line] )
    lines.append(line)
with open('results.csv','wb') as f:
    f.writelines( '\r\n'.join(lines) )



