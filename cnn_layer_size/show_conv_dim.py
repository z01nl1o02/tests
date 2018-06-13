import os,sys,pdb
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('--height','-H',help='input height',type=int)
ap.add_argument('--width','-W',help='input width',type=int)
ap.add_argument('--layers','-F',help='layer info txt with each line for one layer')
ap.add_argument('--deconv','-D',help='0 for conv 1 for deconv',type=int,default=0)
args = ap.parse_args()


def conv(CKSP,HW):
    C,K,S,P = CKSP
    H,W = HW
    H = int((H - K + 2*P)/S + 1)
    W = int((W - K + 2*P)/S + 1)
    return H,W

def deconv(CKSP,HW):
    C,K,S,P = CKSP
    H,W = HW
    H = int((H - 1) * S + K - 2*P)
    W = int((W - 1) * S + K - 2*P)
    return H,W



def calc_conv(layers,HW):
    output = [HW]
    for CKSP in layers:
        HW  = conv(CKSP,HW)
        output.append(HW)
    return output

def calc_deconv(layers,HW):
    output = [HW]
    for CKSP in layers:
        HW  = deconv(CKSP,HW)
        output.append(HW)
    return output


def parse_layers(filepath):
    layers = []
    with open(filepath,'rb') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            C,K,S,P = [int(x) for x in line.split(',')] #channel number, kernel size, stride, padding
            layers.append( (C,K,S,P) )
    return layers


layers = parse_layers(args.layers)
if args.deconv == 0:
    HW = calc_conv(layers, (args.height,args.width))
    print 'conv...'
else:
    HW = calc_deconv(layers, (args.height,args.width))
    print 'deconv...'
   
print('input size (h,w) = (%d,%d)'%(args.height,args.width))
for (H,W),(C,K,S,P) in zip(HW[1:],layers):
    print('(channel,kernel,stride,padding)=(%d,%d,%d,%d)   (h,w) = (%d,%d)'%(C,K,S,P,H,W))





