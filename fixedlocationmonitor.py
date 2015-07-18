import os,sys,pdb,cv2,math,pickle
import numpy as np
"""
Robust Real-Time Unusual Event Detection Using Multiple Fixed-Location Monitors"
by Amit Adam, Ehud Rivlin, llan Shimshoni, David Reinitz
"""

class MONITOR:
    def __init__(self, centerxy, frameshape, nbr_radius, ssd_radius):
        self.x = centerxy[0]
        self.y = centerxy[1]
        height = frameshape[0]
        width = frameshape[1]
        self.left = np.maximum(self.x - nbr_radius,ssd_radius)
        self.top = np.maximum(self.y - nbr_radius,ssd_radius)
        self.right = np.minimum(self.x + nbr_radius, width - ssd_radius)
        self.bottom = np.minimum(self.y + nbr_radius, height - ssd_radius)
        self.ssd_radius = ssd_radius
        self.ssd_a = 1 / 1.0
        self.ssd_k = 1.0
        self.hists = []

    def get_centerxy(self):
        return (self.x, self.y)
    
    def save(self,filename):
        with open(filename, 'w') as f:
            pickle.dump( (self.x, self.y, self.left, self.top, self.right, self.bottom, self.sdd_radius, self.ssd_a, self.ssd_k, self.hists),f)

    def load(self,filename):
        with open(filename,'r') as f:
            self.x, self.y, self.left, self.top, self.right, self.bottom, self.sdd_radius, self.ssd_a, self.ssd_k, self.hists = pickle.load(f)


    def calc_ssd(self, f0, f1):
        winsize = (2 * self.ssd_radius + 1) * (2 * self.ssd_radius + 1) * 1.0
        probmap = np.zeros((self.bottom - self.top, self.right - self.left))
        for y in range(self.top, self.bottom):
            for x in range(self.left, self.right):
                x0 = x - self.ssd_radius
                x1 = x + self.ssd_radius
                y0 = y - self.ssd_radius
                y1 = y + self.ssd_radius
                b0 = f0[y0:y1,x0:x1]
                b1 = f1[y0:y1,x0:x1]
                ssd = np.sqrt( np.mean((b0 - b1) * (b0 - b1)) ) 
                probmap[y-self.top,x-self.left] = ssd

        probmap = self.ssd_k * np.exp(-self.ssd_a * probmap)  


        return probmap

    def histogram_on_orientation(self, probmap):
        cx = probmap.shape[1] / 2
        cy = probmap.shape[0] / 2
        binsize = 10
        binnum = 360 / binsize
        hist = np.zeros((binnum,1))
        for y in range(probmap.shape[0]):
            for x in range(probmap.shape[1]):
                if probmap[y,x] < 0.0001:
                    continue 
                dx = x - cx
                dy = y - cy
                a = math.atan2(dy,dx) * 180 / np.pi
                if  a < 0:
                    a += 360
                a = np.int32(a / binsize)
                if a >= binnum:
                    a = binnum - 1
                hist[a,0] += probmap[y,x]
        hist = hist / np.sum(hist)
        return hist

    def calc_anomaly_probability(self, queryhist):
        if len(self.hists) < 1:
             return -1.0

        #most-likely and ambiguity test
        T = 20 #degree
        [y,x] = np.nonzero(queryhist == queryhist.max())
        y = y[0]
        x = x[0]
        if type(y) is np.ndarray:
            y = y[0]
            x = x[0]
        yml = y
        s = 0
        for k in range(queryhist.shape[0]):
            s += queryhist[k] * np.abs(k - yml)
        s *= 360 / queryhist.shape[0]
        if s >= T:
            return -1.0

        #get reference histogram
        refhist = np.zeros(queryhist.shape)
        for h in self.hists:
            refhist += h
        refhist /= len(self.hists)         
        return 1 - refhist[yml]
     
    def check_add_new_frame(self, f0, f1):
        thresh = 0.

        probmap = self.calc_ssd(f0,f1)
        hist = self.histogram_on_orientation(probmap)
        prob = self.calc_anomaly_probability(hist)

        if len(self.hists) >= 20:
            self.hists.pop() #delete the last one      
        self.hists.insert(0, hist) #insert the header

        if prob > 0.5:
            return 1 #alarmed
        elif prob < 0:
            return -1
        else:
            return 0

def scan_dir_for(dirname,objext): 
    results = []
    for rdir,pdir, names in os.walk(dirname):
        for name in names:
            sname,ext = os.path.splitext(name)
            if 0 == cmp(ext, objext):
                fname = os.path.join(rdir,name)
                results.append((sname, fname))
    return results


def setup_monitors(img):
    results = []
    nbr_radius = 8
    ssd_radius = 3
    frameshape = img.shape
    for y in range(nbr_radius, img.shape[0] - nbr_radius, 2 * nbr_radius):
        for x in range(nbr_radius, img.shape[1] - nbr_radius, 2 * nbr_radius):
            centerxy = (x,y)
            monitor = MONITOR(centerxy, frameshape, nbr_radius, ssd_radius)
            results.append(monitor)
    return results 

def run_train(traindir, outdir, monitors):
    filenames = scan_dir_for(traindir, '.tif')
     
    for idx in range(len(filenames)):
        sname, fname = filenames[idx]
        f1 = cv2.imread(fname, 0)
        if len(monitors) < 1:
            monitors = setup_monitors(f1)
            print 'setup ', len(monitors)
        if idx == 0:
            f0 = f1
            continue
        alarmed = [0 for k in range(len(monitors))]
        for k in range(len(monitors)):
            alarmed[k] = monitors[k].check_add_new_frame(f0,f1)
   
        alarmed = np.array(alarmed) 
        total = len(alarmed)
        quiet = np.sum(alarmed < 0)
         
        f0 = f1 #switch 
        img = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) 
        maskcolor = np.array([0,0,255])
        for k in range(len(alarmed)):
            if alarmed[k] <= 0:
                continue
            cx,cy = monitors[k].get_centerxy()
            radius = 8
            w0 = 0.5
            w1 = 1 - w0
            for y in range(cy - radius, cy + radius,2):
                for x in range(cx - radius, cx + radius,2):
                    img[y,x,:] = np.uint8(img[y,x,:] * w0 + maskcolor * w1)
        outfilename = outdir + sname + ".jpg"
        print sname,' %d/%d'%(quiet,total)
        cv2.imwrite(outfilename, img)

if __name__ == "__main__":
    with open('config.txt', 'r') as f:
        rootdir = f.readline().strip()
    run_train(rootdir+'Train001/', 'out/', [])
            


    
                 



