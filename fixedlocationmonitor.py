import os,sys,pdb,cv2,math
import numpy as np
"""
Robust Real-Time Unusual Event Detection Using Multiple Fixed-Location Monitors"
by Amit Adam, Ehud Rivlin, llan Shimshoni, David Reinitz
"""

class MONITOR:
    def __self__(self, centerxy, frameshape, nbr_radius, ssd_radius):
        self.x = centerxy[0]
        self.y = centerxy[1]
        height = frameshape[0]
        width = frameshape[1]
        self.left = np.maximum(self.x - nbr_radius,ssd_radius)
        self.top = np.maximum(self.y - nbr_radius,ssd_radius)
        self.right = np.minimum(self.x + nbr_radius, width - ssd_radius)
        self.bottom = np.minimum(self.y + nbr_radius, height - ssd_radius)
        self.sdd_radius = ssd_radius
        self.ssd_a = 1 / 100.0
        self.ssd_k = 1.0
        self.hists = []

    def calc_ssd(self, f0, f1):
        winsize = (2 * self.ssd_radius + 1) * (2 * self.ssd_radius + 1)) * 1.0
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
        cy = prbomap.shape[0] / 2
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
         hist = hist / np.sum(hist) #normalization
         return hist

    def calc_anomaly_probability(self, queryhist):
         if len(self.hists) < 1:
             return 0.0

        #most-likely and ambiguity test
        T = 20 #degree
        [y,x] = np.find(queryhist == queryhist.max())
        if len(y) > 0:
            y = y[0]
            x = x[0]
        yml = y
        s = 0
        for k in queryhist.shape[0]:
            s += queryhist[k] * np.abs(k - yml)
        s *= 360 / queryhist.shape[0]
        if s >= T
            return 0.0

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
        else:
            return 0





