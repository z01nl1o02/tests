import os,sys,pickle,shutil,pdb

class PYREF(object):
    def build_tree(self, rootdir, outdir, dirtree):
        subdirs = dirtree.replace(rootdir, '').strip().strip('\\').split('\\')
        try:
            if 0:
                line = outdir
                for item in subdirs:
                    line = os.path.join(line, item)
                    os.mkdir(line)
            else:
                subdirs = "\\".join(subdirs)
                os.makedirs(os.path.join(outdir,subdirs))
        except Exception, e:
            x = None            
        return
       
      
    def search_by_ext(self,rootdir, extdict, resultdir):
        pathlist = []
        unkext = {}
        for rdirs,pdirs,names in os.walk(rootdir):
            for name in names:
                sname,ext = os.path.splitext(name)
                if ext not in extdict:
                    if ext not in unkext:
                        unkext[ext] = 1
                        print 'unknown ext ', ext
                    continue
                fname = os.path.join(rdirs, name)
                pathlist.append((fname,name))


        if 1:
            num = 0
            misslist = []
            for fname, name in pathlist:
                rname,ext = os.path.splitext(fname)
                if os.path.exists(rname):
                    num += 1
                    thedir = '\\'.join(rname.split('\\')[0:-1])
                    self.build_tree(rootdir, resultdir, thedir)
                    thedir = thedir.replace(rootdir, resultdir)
                    shutil.copy(fname, thedir)
                    shutil.copy(rname, thedir)
                elif os.path.exists(rname+'.pdf'):
                    num += 1
                    rname = rname + '.pdf'
                    thedir = '\\'.join(rname.split('\\')[0:-1])
                    self.build_tree(rootdir, resultdir, thedir)
                    thedir = thedir.replace(rootdir, resultdir)
                    shutil.copy(fname, thedir)
                    shutil.copy(rname, thedir)
                else:
                    misslist.append((fname,name))
            pathlist = misslist
            print 'matched ', num
 

        if resultdir is not None:
            print 'start to write log '
            filenum = 0
            line = ""
            for k in range(len(pathlist)):
                fname, name = pathlist[k]
                if (k + 1)%10000 == 0:
                    with open(os.path.join(resultdir,'search_result_'+str(filenum)+'.txt'),'w') as f:
                        f.writelines(line)
                        line = ""
                        filenum += 1
                line += name + ',' + fname + '\n'

            if len(line) > 0:
                with open(os.path.join(resultdir,'search_result_'+str(filenum)+'.txt'),'w') as f:
                        f.writelines(line)

            
        return pathlist

    def get_duplicated_files(self,pathlist, resultdir):
        tmpdict = {}
        for fname, name in pathlist:
            if name not in tmpdict:
                tmpdict[name] = [fname]
            else:
                tmpdict[name].append(fname)
        for name in tmpdict.keys():
            if len(tmpdict[name]) < 2:
                tmpdict.pop(name)
        if resultdir is not None:
            print 'start to write log '
            line = ""
            for name in tmpdict.keys():
                line += name + '\n'
                for item in tmpdict[name]:
                    line += '\t' + item + '\n'
                line += '==============================\n'
            with open(os.path.join(resultdir, 'dup.txt'), 'w') as f:
                f.writelines(line)
        return tmpdict

if __name__=="__main__":
    rootdir = sys.argv[1]
    pyref = PYREF()
    pathlist = pyref.search_by_ext(rootdir, {'.lms':1, '.vim':1, '.vi':1}, 'out')
    with open('pathlist.dat', 'wb') as f:
        pickle.dump(pathlist, f)
    dupdict = pyref.get_duplicated_files(pathlist,'out')
    with open('dupdict.dat', 'wb') as f:
        pickle.dump(dupdict, f)
     




