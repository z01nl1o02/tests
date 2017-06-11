import os,sys,pdb
import numpy as np
import cv2
from collections import defaultdict
import argparse

class MUCT_FOR_ASMLIB:
    def __init__(self, rootdir):
        self._landmarkpath = os.path.join(rootdir,'muct-landmarks/muct76-opencv.csv')
        self._jpgdir = os.path.join(rootdir, 'jpg')
        self._ptsfdir = self._jpgdir
    def load_landmark(self):
        self._landmarkdict = defaultdict(list)
        with open(os.path.join(self._landmarkpath),'rb') as f:
            line = f.readline()
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                data = line.split(',')
                name = data[0]
                tag = data[1]
                coord = []
                for k in range( 2, len(data), 2):
                    x = int(float(data[k]))
                    y = int(float(data[k + 1]))
                    coord.append((x,y))
                self._landmarkdict[name] = coord
        print '%d shapes loaded'%len(self._landmarkdict)
        return
    def get_image_path(self,name):
        return os.path.join(self._jpgdir, name + '.jpg')

    def show_shape(self):
        cv2.namedWindow("show shape")
        for name in self._landmarkdict.keys():
            coord = self._landmarkdict[name]
            img = cv2.imread(get_image_path(name),1)
            if img is None:
                print 'miss ',name
                continue
            for x,y in coord:
                cv2.circle(img, (x,y), 3,(0,255,0))
            cv2.imshow("show shape", img)
            c = cv2.waitKey(-1)
            if int(c) == 27:
                break
        cv2.destroyWindow('show shape')
        return
    def output_pts(self,setid): #a,d,e only due to b/c contains miss points
        total = 0
        for name in self._landmarkdict.keys():
            if setid != name.split('-')[0][-1]:
                continue
            if not os.path.exists(self.get_image_path(name)):
                print 'miss ',name,'\t',
                continue
            total += 1
            coord = self._landmarkdict[name]
            lines = ['%d'%len(coord)]
            for x, y in coord:
                lines.append( '%d %d'%(x,y) )
            with open( self.get_image_path(name) + '.pts', 'wb') as f:
                f.writelines('\r\n'.join(lines) )
        print '',total, ' pts done!'
        return 
    def output_list(self):
        lines = []
        objs = os.listdir( self._jpgdir)
        for obj in objs:
            sname,ext = os.path.splitext(obj)
            if '.pts' != ext:
                continue
            lines.append( os.path.join(self._jpgdir, obj) )
        print 'write %d pts files in list file'%len(lines)
        with open('list.txt', 'wb') as f:
            f.writelines('\r\n'.join(lines))
        return 
    def output_modeldef(self):
        lines = []
        lines.append("#number of landmark points")
        lines.append('76')
        lines.append('#number of paths')
        lines.append('10') #num of path
        #check ShapeInfo::loadFromShapeDescFile() for detail
        lines.append('#contour')
        lines.append('17 0') #path point index is 1-based !!
        lines.append('22 0')
        lines.append('28 0')
        lines.append('33 0')
        lines.append('38 0')
        lines.append('49 1')
        lines.append('61 0')
        lines.append('69 1')
        lines.append('73 1')
        lines.append('76 1')

        lines.append('#initial positions')
        lines.append('# r.y -= r.height*?')
        lines.append('0.1')
        lines.append('# r.x -= r.width*?')
        lines.append('0.1')
        lines.append('# r.width *= ?')
        lines.append('1.2')
        lines.append('# r.height *= ?')
        lines.append('1.45')
        
        lines.append('#???')
        lines.append('#step?*100/sqrt(area)')
        lines.append('1.0')


        lines.append('#init scale ratio when searching')
        lines.append('0.65')
        lines.append('#init X offset when searching')
        lines.append('0')
        lines.append('#init Y offset when searching')
        lines.append('0.25')
        with open('modeldef.txt', 'wb') as f:
            f.writelines('\r\n'.join(lines))
    def output(self):
        self.output_pts('a')
        self.output_pts('d')
        self.output_pts('e')
        self.output_list()
        self.output_modeldef()
        return

def run(rootdir):
    CLS = MUCT_FOR_ASMLIB(rootdir)
    CLS.load_landmark()
    #CLS.show_shape()
    CLS.output()
    return 

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('muctroot',help='root path of muct dataset')
    args = ap.parse_args()
    run(args.muctroot)



