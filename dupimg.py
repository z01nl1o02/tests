import os,sys,pdb,pickle
import hashlib
import shutil
md = {}
for root,dirs,names in os.walk('.'):
    for name in names:
        sname,ext = os.path.splitext(name)
        if 0 != cmp('.jpg', ext):
            continue
        fname = os.path.join(root, name)
        code = hashlib.md5(open(fname,"rb").read()).hexdigest()
        print name,'->',code
        if code not in md:
            md[code] = [fname]
        else:
            md[code].append(fname)
os.mkdir('result')
for code in md.keys():
    fname = md[code][0] #only the first one
    shutil.move(fname,'result')
print 'done!'