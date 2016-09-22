import os,sys,pdb,cPickle
import hashlib
import argparse

"""
used to find mapping between two image set. 
image name updated e.g.
"""

def run(md2jpg_pkl_from,md2jpg_pkl_to,outfile):
    with open(md2jpg_pkl_from,'rb') as f:
        md2jpgFrom = cPickle.load(f)
    with open(md2jpg_pkl_to,'rb') as f:
        md2jpgTo = cPickle.load(f)
        
    theMap = {}
    for md in md2jpgFrom.keys():
        if md not in md2jpgTo.keys():
            print 'miss %s'%(md2jpgFrom[md][0])
            continue
        jpgFrom = md2jpgFrom[md][0].split('\\')[-1]
        jpgTo = md2jpgTo[md][0].split('\\')[-1]
        theMap[jpgFrom] = jpgTo
    sortMap = sorted(theMap.iteritems(), key = lambda X:X[0])
    lineList = ['oldname|newname']
    for jpgFrom,jpgTo in sortMap:
        lineList.append('%s|%s'%(jpgFrom,jpgTo))
    lines = '\r\n'.join(lineList)    
    with open(outfile, 'wb') as f:
        f.writelines(lines)

if  __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('mdDictFrom',help='md2jpg as source')
    ap.add_argument('mdDictTo',help='md2jpg as objective')
    ap.add_argument("outcsv", help='output csv')
    args = ap.parse_args()
    run(args.mdDictFrom,args.mdDictTo,args.outcsv)

