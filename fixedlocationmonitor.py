import os,sys,pdb,cv2,math,pickle
import numpy as np

"""
papers:

Robust Real-Time Unusual Event Detection Using Multiple Fixed-Location Monitors"
by Amit Adam, Ehud Rivlin, llan Shimshoni, David Reinitz

dataset:
UCSD_Anomaly Dataset
"""

class MONITOR:
    def __init__(self, centerxy, frameshape, nbr_radius, ssd_radius, b_speed_mode):
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
        self.histcapacity = -1 #split train and predict
        self.hists = []
        self.nbr_radius = nbr_radius
        self.b_speed_mode = b_speed_mode

    def get_centerxy(self):
        return (self.x, self.y)
 
    def get_region(self):
        return (self.left, self.top, self.right, self.bottom) 

    def get_history_length(self):
        return len(self.hists)     
      
    def calc_ssd(self, f0, f1):
        winsize = (2 * self.ssd_radius + 1) * (2 * self.ssd_radius + 1) * 1.0
        probmap = np.zeros((self.bottom - self.top, self.right - self.left))

        x = np.int32((self.left + self.right) / 2)
        y = np.int32((self.top + self.bottom) / 2)
        x0 = x - self.ssd_radius
        x1 = x + self.ssd_radius
        y0 = y - self.ssd_radius
        y1 = y + self.ssd_radius
        b0 = np.float32(f0[y0:y1,x0:x1])
        
        for y in range(self.top, self.bottom):
            for x in range(self.left, self.right):
                x0 = x - self.ssd_radius
                x1 = x + self.ssd_radius
                y0 = y - self.ssd_radius
                y1 = y + self.ssd_radius
                b1 = np.float32(f1[y0:y1,x0:x1])
                ssd = np.mean(np.abs(b0 - b1))
                probmap[y-self.top,x-self.left] = ssd

        probmap = self.ssd_k * np.exp(-self.ssd_a * probmap)  
        if 0:
            pdb.set_trace()
            cv2.imwrite('f0.jpg', f0)
            cv2.imwrite('f1.jpg', f1)
            img = np.zeros(f0.shape)
            for y in range(probmap.shape[0]):
                for x in range(probmap.shape[1]):
                    row = y + self.top
                    col = x + self.left
                    img[row,col] = np.uint8(probmap[y,x] * 255)
            cv2.imwrite('ssd.jpg', img)

        return probmap

    def histogram_on_speed(self, probmap):
        cx = probmap.shape[1] / 2
        cy = probmap.shape[0] / 2
        binsize = 1
        binnum = np.int64(self.nbr_radius / binsize) + 1
        hist = np.zeros((binnum,1))
        for y in range(probmap.shape[0]):
            for x in range(probmap.shape[1]):
                if probmap[y,x] < 0.0001:
                    continue 
                dx = np.abs(x - cx)
                dy = np.abs(y - cy)
                d = np.maximum( dx, dy )
                d = np.int64(d/binsize)
                if d >= binnum:
                    d = binnum - 1
                hist[d,0] += probmap[y,x]
        hist = hist /(0.001 + np.sum(hist) )
        return hist
        
    def histogram_on_orientation(self, probmap):
        cx = probmap.shape[1] / 2
        cy = probmap.shape[0] / 2
        binsize = 30
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
        hist = hist / (np.sum(hist) + 0.0001)
        return hist

    def calc_histogram(self,probmap):
        if self.b_speed_mode == 1:
            return self.histogram_on_speed(probmap)
        else:
            return self.histogram_on_orientation(probmap)

    #a method to show monitor inforamtion stored
    def calc_histogram_mean(self):
        if len(self.hists) < self.histcapacity and self.histcapacity > 0:
            return 0.0
        refhist = np.zeros(self.hists[0].shape)
        for h in self.hists:
            refhist += h
        refhist /= len(self.hists)      
        
        s = 0
        for k in range(refhist.shape[0]):
            s += (k + 1) * refhist[k,0]
        s = s * 1.0 / refhist.shape[0]
        return s
             


    def most_likely_and_ambiguity_test(self, hist):
        [y,x] = np.nonzero(hist == hist.max())
        y = y[0]
        x = x[0]
        if type(y) is np.ndarray:
            y = y[0]
            x = x[0]
        yml = y

        if self.b_speed_mode == 0:
            T = 20 #degree
            s = 0
            for k in range(hist.shape[0]):
                s += hist[k,0] * np.abs(k - yml)
            s *= 360.0 / hist.shape[0]
            if s >= T:
                return (0,yml) #bad observation
        else:
            T = 1.5 
            s = 0
            for k in range(hist.shape[0]):
                s += hist[k,0] * np.abs(k - yml)
            s *= self.nbr_radius * 1.0 / (hist.shape[0] - 1)
            if s >= T:
                return (0,yml)

        return (1,yml)


 
    def calc_anomaly_probability(self, queryhist):
        if len(self.hists) < self.histcapacity and self.histcapacity > 0:
             return -1.0

        #most-likely and ambiguity test
        ret,yml = self.most_likely_and_ambiguity_test(queryhist)
        if ret == 0:
            return -1.0

        #get reference histogram
        refhist = np.zeros(queryhist.shape)
        for h in self.hists:
            refhist += h
        refhist /= len(self.hists)         

        return 1 - refhist[yml,0]
     
    def check_add_new_frame(self, f0, f1):

        probmap = self.calc_ssd(f0,f1)
        hist = self.calc_histogram(probmap)
        prob = self.calc_anomaly_probability(hist)

        if prob >= 0:
            if len(self.hists) >= self.histcapacity and self.histcapacity > 0:
                self.hists.pop() #delete the last one      
            self.hists.insert(0, hist) #insert the header

        if prob > 0.8:
            return 1 #alarmed
        elif prob < 0:
            return -1
        else:
            return 0


    def add_frame(self, f0, f1):
        probmap = self.calc_ssd(f0,f1)
        hist = self.calc_histogram(probmap)

        #only insert good observation
        ret, yml = self.most_likely_and_ambiguity_test(hist)
        if ret == 0:
            return 

        if len(self.hists) >= self.histcapacity and self.histcapacity > 0:
            self.hists.pop() #delete the last one      
        self.hists.insert(0, hist) #insert the header


    def check_frame(self, f0, f1):

        if len(self.hists) < 1:
            return -1

        probmap = self.calc_ssd(f0,f1)
        hist = self.calc_histogram(probmap)
        prob = self.calc_anomaly_probability(hist)

        if prob > 0.9:
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
    ssd_radius = 8
    frameshape = img.shape
    b_speed_mode  = 0
    for y in range(nbr_radius + ssd_radius, img.shape[0] - nbr_radius - ssd_radius, 2 * nbr_radius):
        for x in range(nbr_radius + ssd_radius, img.shape[1] - nbr_radius - ssd_radius, 2 * nbr_radius):
            centerxy = (x,y)
            monitor = MONITOR(centerxy, frameshape, nbr_radius, ssd_radius,b_speed_mode)
            results.append(monitor)
    return results 

def run_train(traindir, monitors):
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
        for k in range(len(monitors)):
            monitors[k].add_frame(f0,f1)
        f0 = f1 #switch 
        print '.',
    print '\r\n'
    return monitors

def run_online_train(imgdir,outdir):
    filenames = scan_dir_for(imgdir, '.tif')
    monitors = [] 
    for idx in range(len(filenames)):
        sname, fname = filenames[idx]
        f1 = cv2.imread(fname, 0)
        if len(monitors) < 1:
            monitors = setup_monitors(f1)
            print 'setup ', len(monitors)
        if idx == 0:
            f0 = f1
            continue

        alarms = [0 for k in range(len(monitors))]
        for k in range(len(monitors)):
            alarms[k] = monitors[k].check_add_new_frame(f0,f1)

        if 0:
            img = np.zeros(f1.shape)
            for mts in monitors:
                v = mts.calc_histogram_peak()
                left,top,right,bottom = mts.get_region()
                img[top:bottom, left:right] = v
            cv2.imwrite('out/%s.1.jpg'%sname, img)

        f0 = f1 #switch 

       
        img = cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB)
        maskcolor = np.array([0,0,255])
        for k in range(len(alarms)):
            if alarms[k] <= 0:
                continue
            cx,cy = monitors[k].get_centerxy()
            radius = 8
            w0 = 0.4
            w1 = 1 - w0
            for y in range(cy - radius, cy + radius,1):
                for x in range(cx - radius, cx + radius,1):
                    img[y,x,:] = np.uint8(img[y,x,:] * w0 + maskcolor * w1)
        outfilename = outdir + sname + ".jpg"
        cv2.imwrite(outfilename, img)
        print 'online train ',sname
    return monitors




def run_predict(traindir, outdir, monitors):

    filenames = scan_dir_for(traindir, '.tif')
    
    for idx in range(len(filenames)):
        sname, fname = filenames[idx]
        f1 = cv2.imread(fname, 0)
        if idx == 0:
            f0 = f1
            continue

        if 0:
            img = cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB)
            for k in range(len(monitors)):
                x,y = monitors[k].get_centerxy()
                cv2.putText(img, '%d'%k ,(x,y), cv2.FONT_HERSHEY_COMPLEX,0.2,(255,0,0))
            cv2.imwrite('test.2.jpg', img)

        alarmed = [0 for k in range(len(monitors))]
        for k in range(len(monitors)):
            alarmed[k] = monitors[k].check_frame(f0,f1)
   
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
            w0 = 0.4
            w1 = 1 - w0
            for y in range(cy - radius, cy + radius,1):
                for x in range(cx - radius, cx + radius,1):
                    img[y,x,:] = np.uint8(img[y,x,:] * w0 + maskcolor * w1)
        outfilename = outdir + sname + ".jpg"
        print 'predict',sname,' %d/%d'%(quiet,total)
        cv2.imwrite(outfilename, img)
    return monitors


if __name__ == "__main__":
    with open('config.txt', 'r') as f:
        rootdir = f.readline().strip()
    if 0:
        monitors = run_train(rootdir+'Train/Train001/', [])
        for k in range(2, 34):
            traindir = rootdir + 'Train/Train%.3d/'%k
            monitors = run_train(traindir,monitors)
            with open('model%d.txt'%k, 'w') as f:
                pickle.dump(monitors, f)
        run_predict(rootdir+'Test/Test001/', 'out/', monitors)
    elif 1:
        with open('model9.txt', 'r') as f:
            monitors = pickle.load(f)

        if 1:
            monitor_infos = []
            m1 = 0
            for mts in monitors:
                m = mts.calc_histogram_mean()
                left,top,right,bottom = mts.get_region()
                monitor_infos.append((m, left, top, right, bottom))
                if m > m1:
                    m1 = m

            img = np.zeros((158,238))
            for item in monitor_infos:
                m, left, top, right, bottom = item
                img[top:bottom, left:right] = np.uint8(m * 255.0 / m1)
            cv2.imwrite('test.1.jpg', img)

        run_predict(rootdir+'Test/Test031/', 'out/', monitors)

    else:
        monitors = run_online_train(rootdir+'Test/Test025/', 'out/')


    
                 



