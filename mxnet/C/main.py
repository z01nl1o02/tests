import colornet
import sampleio
import os,sys,pdb

def run(traindir,testdir,w,h):
    batchsize = 300
    io = sampleio.SAMPLEIO()
    io.load(traindir,batchsize,w,h)
    train_iter = io.get_data_iter()
    model = colornet.COLORNET()
    model.fit(train_iter, None,batchsize)
    acc = model.predict(train_iter)
    print 'train acc = ',acc
    io.load(testdir,batchsize,w,h)
    test_iter = io.get_data_iter();
    acc = model.predict(test_iter)
    print 'test acc = ',acc
    

if __name__=="__main__":
    run(sys.argv[1],sys.argv[2],32,32)


