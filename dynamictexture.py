import sys,os,pdb,gc,pickle
import cv2
import numpy as np
from numpy.linalg import svd
from numpy.linalg import pinv
"""
X(t) = A*X(t-1) + B
Y(t) = C*X(t) + Q
"""

"""
reference papers:
dynamic textures
by GIANFRANCO DORETTO
   ALESSANDRO CHIUSO
   YING NIAN WU
   STEFANO SOATTO
"""

class DYNAMIC_TEXTURE:
    def load_frames(self, framedir, framerange):
        Ylist = []
        scale = 2
        for idx in range(framerange[0], framerange[1], 1):
            fname = os.path.join(framedir, '%.3d.tif'%idx)
            img = cv2.imread(fname,0)
            stdw = np.int32(img.shape[1] / scale)
            stdh = np.int32(img.shape[0] / scale)
            img = cv2.resize(img,(stdw,stdh))
            frameshape = img.shape
            img = np.float32(np.reshape(img, (1,-1)))
            Ylist.append(img)
        Y = np.zeros((len(Ylist), Ylist[0].shape[1]))
        for k in range(Y.shape[0]):
            Y[k,:] = Ylist[k]  
        return (Y,frameshape)

    def create_dynamic_textures(self,Y,n,nv):
        Y = Y.transpose() #convert to featurenum * samplenum
        tau = Y.shape[1]
        Ymean = np.reshape(np.mean(Y, 1), (-1,1))
        u,s,v = svd(Y - np.tile(Ymean,(1,Y.shape[1])), 0)
        C = u #formula 9
        X = np.dot(np.diag(s), v) #formula 9
        x0 = X[:,0] 
        A = np.dot(X[:,1:] , pinv(X[:,0:tau-1]) ) #formula 10
        V = X[:,1:] - np.dot(A, X[:,0:tau-1])
        u,s,v = svd(V,0)
        s = np.diag(s)
        B = np.dot(u[:,0:nv] , s[0:nv,0:nv] ) /np.sqrt(tau - 1)
        return (x0, Ymean, A, B, C)

    def rebuild(self,x0, Ymean, A, B, C, tau):
        n,nv = B.shape
        X = np.zeros((len(x0),tau + 1))
        I = np.zeros((Ymean.shape[0], tau))
        X[:,0] = x0
        for t in range(tau):
            ax = np.reshape(np.dot(A, X[:,t]), (-1,1)) 
            b = np.dot(B, np.random.randn(nv,1))
            X[:,t+1] = np.reshape(ax + b, (-1,))
            cx = np.dot(C, np.reshape(X[:,t], (-1,1)))
            I[:,t] = np.reshape(cx + Ymean, (-1,))
        return I

    def run(self, framedir,framerange, rebuilddir):
        Y,fshape = self.load_frames(framedir, framerange)
        x0, Ymean, A, B, C = self.create_dynamic_textures(Y, 9, 8)
        if len(rebuilddir) > 1:
            I = self.rebuild(x0, Ymean, A, B,C,9)
            for k in range(I.shape[1]):
                y = I[:,k]
                y = np.uint8(y)
                y = np.reshape(y, fshape)
                outpath = rebuilddir + '%.3d.jpg'%k
                cv2.imwrite(outpath, y)

if __name__=='__main__':
    with open('config.txt', 'r') as f:
        rootdir = f.readline().strip()
        outdir = f.readline().strip()
    dyntexture = DYNAMIC_TEXTURE()
    dyntexture.run(rootdir, (1,10), outdir)
     

