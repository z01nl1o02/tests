import os,sys,pdb

def gen_dir_list(rootdir):
    try:
        os.remove('list.txt')
    except Exception as e:
        print 'exception : ',e
    line = ""
    srcdepth = len(rootdir.strip('\\').split('\\'))
    for rdir,pdir, names in os.walk(rootdir):
        curdepth = len(rdir.strip('\\').split('\\'))
        if curdepth != srcdepth:
            continue
        if len(pdir) == 0:
            continue
        for dn in pdir:
            line += dn + '\n'
    with open('list.txt', 'w') as f:
        f.writelines(line)        

if __name__=="__main__":
    gen_dir_list(sys.argv[1])
