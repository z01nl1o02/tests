import os,sys,pdb
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import argparse


def show_train(infile):
    stat = defaultdict(list)
    with open(infile, 'rb') as f:
        line = f.readline()
        for line in f:
            line = line.strip()
            if line == "":
                continue
            NumIters,Seconds,LearningRate,loss = [np.float32(d) for d in line.split(',')]
            stat['NumIters'].append( NumIters )
            stat['Seconds'].append( Seconds )
            stat['LearningRate'].append( LearningRate )
            stat['loss'].append(loss)
    return stat
    

def show_test(infile):
    stat = defaultdict(list)
    with open(infile, 'rb') as f:
        line = f.readline()
        for line in f:
            line = line.strip()
            if line == "":
                continue
            NumIters,Seconds,LearningRate,accuracy,loss = [np.float32(d) for d in line.split(',')]
            stat['NumIters'].append( NumIters )
            stat['Seconds'].append( Seconds )
            stat['LearningRate'].append( LearningRate )
            stat['accuracy'].append(accuracy)
            stat['loss'].append(loss)
    return stat
    
def show_log(logpath):
    trainstat = show_train( logpath + '.train')
    teststat = show_test( logpath + '.test')
    plt.figure()
    plt.subplot(411)
    plt.plot(trainstat['NumIters'], trainstat['loss'], label='train loss')
    plt.legend()
    plt.subplot(412)
    plt.plot(teststat['NumIters'], teststat['loss'], label='test loss')
    plt.legend()
    plt.subplot(413)
    plt.plot(teststat['NumIters'], teststat['loss'], label='test loss',color='red')
    plt.plot(trainstat['NumIters'], trainstat['loss'], label='train loss',color='blue')
    plt.legend()
    
    plt.subplot(414)
    plt.plot(teststat['NumIters'], teststat['accuracy'], label='test accuracy')
    plt.legend()
    plt.show()
    
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('logpath',help='path to log file')
    ap.add_argument('-tmpdir',help='temp dir',default = 'temp')
    args = ap.parse_args()
    try:
        os.makedirs(args.tmpdir)
    except Exception,e:
        pass
    cmd = 'python parse_log.py "%s" "%s"'%(args.logpath, args.tmpdir)
    os.system(cmd)
    show_log( os.path.join(args.tmpdir, args.logpath.split(os.sep)[-1]) )

        
            
            


