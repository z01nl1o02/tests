import colornet
import sampleio
import os,sys,pdb

def run(traindir,testdir,w,h):
    batchsize = 300
    train_iter = sampleio.SAMPLEIO().load(traindir,batchsize,w,h).get_data_iter()
    valid_iter = sampleio.SAMPLEIO().load(testdir,batchsize,w,h).get_data_iter()
    model = colornet.COLORNET()
    model.fit(train_iter, valid_iter,batchsize)
    

if __name__=="__main__":
    run(sys.argv[1],sys.argv[2],32,32)


