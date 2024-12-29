import numpy as np
import Sphere.sphere as sp
points3 = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/point3.dat',delimiter=',')
bs = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Bvalues3.dat',delimiter=',')
dbs = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/del_b3.dat',delimiter=',')
bdata = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Bvalues3.dat',delimiter=',')
adata = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Avalues3.dat',delimiter=',')
Ns = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/NeedleStart3.dat',delimiter=',')

f1 = open('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Indexdel3.dat')
indexlist = f1.readlines()
f1.close()

f2 = open('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Index3w.dat')
newindexlist = f2.readlines()
f2.close()


def Diffset(id1,id2):
    wholeIDset = set()
    needIDset = set()
    for i in range(len(id1)):
        sti1 = str(id1[i])
        sti1 = sti1.replace('\n','')
        # print(len(sti))
        # print(type(sti)) #str
        ss = sti1.split(',')
        for j in range(len(ss)):
            wholeIDset.add(int(ss[j]))  

    for i in range(len(id2)):
        sti2 = str(id2[i])
        sti2 = sti2.replace('\n','')
        # print(len(sti))
        # print(type(sti)) #str
        ss = sti2.split(',')
        for j in range(len(ss)):
            needIDset.add(int(ss[j]))    
    #print(newIDset)
    Diffset = wholeIDset - needIDset
    return Diffset

#sp.tsl4
def pfromcb(nb,NeedleS,l = 22.5):
    #cb\\方向
    #保留针头位置，调整针尾方向
    ps = NeedleS
    pe = NeedleS + [l*nb[0],l*nb[1],l*nb[2]]       
    return ps,pe
 
def minerrdir(pts,indexlist,c_bs,idx,itsflag = 0):#每条交了的针道换向一次，indexlist是索引全列表，idx是{2,12,7}，则应换三次 TODO :ns
    #包含粒子号：2: {0, 1, 2, 3, 4, 5, 6, 7}, 7: {72, 73, 74, 75, 76, 77, 78, 79}, 12: {16, 17, 18, 19, 20, 21, 22, 23}
    #point的index和hough的idx是不同的
    # c_bs是待选的方向表，sphere类中生成的，是一堆b的list,函数pfromcb直接算出p1,p2的list,,idx是(相交)被删除的针道号索引{2,12,7}，
    #sphere调用idx,localsubdevide生成cbs,从中选择err最小的为index
    #用于计算err的pts应该是当前idx对应的，实际需要换向的远少于霍夫求出来的，需要进行反向判断,ok
    #给出始末两点p1,p2，便于直接画图
    #TODO：NS和a不一样，用NS得了,用a需要用t
    """ // distance computation after IPOL paper Eq. (7)
    t = (b * (points[i] - a));
    Vector3d d = (points[i] - (a + (t*b))); """
    p1 = []
    p2 = []
    me = []
    for j in range(len(idx)):
        sti = str(indexlist[idx[j]])
        sti = sti.replace('\n','')
        ss = sti.split(',')
        points = []
        for l in range(len(ss)):
            points.append(pts[int(ss[l])])#索引的是粒子号,这应该是比如说第一个针道{0, 1, 2, 3, 4, 5, 6, 7}

        minerr = 10
        ndls = Ns[idx[j]]#{2,12,7} #array([1.67097, 3.41552, 1.55988])
        for indx in range(np.shape(c_bs)[0]):
            b = c_bs[indx]
            t = b * (points - ndls)
            err = []
            for ll in range(np.shape(points)[0]):
                err.append(np.linalg.norm(points[ll] - (ndls + t[ll]*b))) 
            terr = np.sum(err)/np.shape(points)[0] #2.6437491481355067
            if (itsflag == 0 and minerr > terr): 
                minerr = terr
                pp1,pp2 = pfromcb(b,ndls)
                p1.append(pp1)
                p2.append(pp2)
                me.append(minerr)
    return p1,p2,me
    
def invcheck(dset,idx,allindex):
   """从总差集dset和删除idx,贪婪法求最小覆盖，返回最小覆盖id，"""
   idtable = {}
   id = set()

   for i in range(np.shape(idx)[1]):
       tempid = []
       sti = str(allindex[idx[0][i]])
       sti = sti.replace('\n','')
       ss = sti.split(',')
       for j in range(len(ss)):
           tempid.append(int(ss[j])) 

       idtable[idx[0][i]] = set(tempid)

   #print(idtable)
   
   while dset:
    mostcovered = None # 覆盖了最多的未覆盖州的广播台
    seeds_covered = set() # 包含该广播台覆盖的所有未覆盖的州
    for idnum, seedsinnum in idtable.items():
        covered = dset & seedsinnum  # 计算交集
        if len(covered) > len(seeds_covered):
            mostcovered = idnum
            seeds_covered = covered
 
    dset -= seeds_covered
    id.add(mostcovered)

   return id

def findseedid(bs,dbs):
    """从b列表和剔除后的b列表求得删掉的id"""
    res = np.array(list(set(map(tuple,bs)) - set(map(tuple, dbs))))
    boolin = []
    for i in range(np.shape(bs)[0]):
        boolin.append(bs[i] in res)
    idx = np.argwhere(boolin)
    return idx

difset = Diffset(indexlist,newindexlist)
print(difset)#相差粒子号
#print(type(difset)) #<class 'set'>
indx =findseedid(bs,dbs)
print(indx.T)#相差针道号
#print(type(indx)) #<class 'numpy.ndarray'>
#print(indx.T[0][0])#2
#print(np.shape(indx.T)[1])#10
woyaodeid = list(sp.invcheck(difset,indx.T,indexlist))
# Needlestat = Ns[woyaodeid]
print(woyaodeid) #4
iid = sp.findsphereid(sp.tsl3,bs,woyaodeid[0])
_,_,nv = sp.localSubDivide(sp.tsl3,iid)
print(nv)
pp1,pp2,me = minerrdir(points3,indexlist,nv,woyaodeid)
print(pp1)
print(pp2)
print(me)
