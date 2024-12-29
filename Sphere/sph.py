from matplotlib import markers
from Sphere.adjGraph import Graph
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TAU = 1.61803399
NR = math.sqrt(1 + TAU * TAU)
V = 1 / NR
TAU = TAU / NR

def buildicosahedron():
    pl20 = Graph()  

    v = pl20.addVertex('1')
    p1 = [-V,TAU,0]
    v.setPosition(p1)
        
    v = pl20.addVertex('2')
    p2 = [V,TAU,0]
    v.setPosition(p2)

    v = pl20.addVertex('3')
    p3 = [0,V,-TAU]
    v.setPosition(p3)

    v = pl20.addVertex('4')
    p4 = [0,V,TAU]
    v.setPosition(p4)

    v = pl20.addVertex('5')
    p5 = [-TAU,0,-V]
    v.setPosition(p5)
    
    v = pl20.addVertex('6')
    p6 = [TAU,0,-V]
    v.setPosition(p6)

    v = pl20.addVertex('7')
    p7 = [-TAU,0,V]
    v.setPosition(p7)

    v = pl20.addVertex('8')
    p8 = [TAU,0,V]
    v.setPosition(p8)

    v = pl20.addVertex('9')
    p9 = [0,-V,-TAU]
    v.setPosition(p9)

    v = pl20.addVertex('10')
    p10 = [0,-V,TAU]
    v.setPosition(p10)

    v = pl20.addVertex('11')
    p11 = [-V,-TAU,0]
    v.setPosition(p11)

    v = pl20.addVertex('12')
    p12 = [V,-TAU,0]
    v.setPosition(p12)

    #按triangle的顺序addedge ,id+1
    pl20.addTriangle(pl20.vertices['1'],pl20.vertices['2'],pl20.vertices['3'])
    pl20.addTriangle(pl20.vertices['1'],pl20.vertices['2'],pl20.vertices['4'])
    pl20.addTriangle(pl20.vertices['1'],pl20.vertices['3'],pl20.vertices['5'])
    pl20.addTriangle(pl20.vertices['1'],pl20.vertices['5'],pl20.vertices['7'])
    pl20.addTriangle(pl20.vertices['1'],pl20.vertices['4'],pl20.vertices['7'])
    pl20.addTriangle(pl20.vertices['2'],pl20.vertices['3'],pl20.vertices['6'])
    pl20.addTriangle(pl20.vertices['2'],pl20.vertices['4'],pl20.vertices['8'])
    pl20.addTriangle(pl20.vertices['2'],pl20.vertices['6'],pl20.vertices['8'])
    pl20.addTriangle(pl20.vertices['3'],pl20.vertices['5'],pl20.vertices['9'])
    pl20.addTriangle(pl20.vertices['3'],pl20.vertices['6'],pl20.vertices['9'])
    pl20.addTriangle(pl20.vertices['4'],pl20.vertices['7'],pl20.vertices['10'])
    pl20.addTriangle(pl20.vertices['4'],pl20.vertices['8'],pl20.vertices['10'])
    pl20.addTriangle(pl20.vertices['5'],pl20.vertices['9'],pl20.vertices['11'])
    pl20.addTriangle(pl20.vertices['9'],pl20.vertices['11'],pl20.vertices['12'])
    pl20.addTriangle(pl20.vertices['6'],pl20.vertices['9'],pl20.vertices['12'])
    pl20.addTriangle(pl20.vertices['6'],pl20.vertices['8'],pl20.vertices['12'])
    pl20.addTriangle(pl20.vertices['8'],pl20.vertices['10'],pl20.vertices['12'])
    pl20.addTriangle(pl20.vertices['10'],pl20.vertices['11'],pl20.vertices['12'])
    pl20.addTriangle(pl20.vertices['7'],pl20.vertices['10'],pl20.vertices['11'])
    pl20.addTriangle(pl20.vertices['5'],pl20.vertices['7'],pl20.vertices['11'])

    pl20.numtriangles += 1 #添加最后一个面，三边连接上了

    return pl20

def localSubDivide(sphere,sid):
    """sid 用findsphereid函数找"""
    sphere.initcolor()
    wholelist = []
    newbv = []
    #TODO: NEWBV，不从sphere里调用，直接做一个新的localb list
    ssid = str(sid)
    while(not(sphere.triangles.isEmpty())):
        wholelist.append(sphere.triangles.removeRear())

    locallist = [] #包含sid的表
    newlocallist = []

    for i in range(len(wholelist)):
       if ssid in wholelist[i]:
           tri = wholelist[i]
           locallist.append(tri)   

    print(locallist)

    for i in range(len(locallist)):
        wholelist.remove(locallist[i])

    for j in range(len(locallist)):
        tri = locallist[j]

        va = sphere.vertices[tri[0]]
        ap = np.array(va.position)
        vb = sphere.vertices[tri[1]]
        bp = np.array(vb.position)
        vc = sphere.vertices[tri[2]]
        cp = np.array(vc.position)

        dp = ap + bp
        dp /= np.linalg.norm(dp)
        ep = ap + cp
        ep /= np.linalg.norm(ep)
        fp = bp + cp
        fp /= np.linalg.norm(fp)

    
        if (va.color >= 1)and(vb.color >= 1):
            if (va.id == ssid) or (vb.id == ssid): #是内边
             vd = va.getChild(vb)
        else:
            vd = sphere.addVertex(str(sphere.numVertices + 1))
            vd.setPosition(dp.tolist())
            newbv.append(dp.tolist())
            va.setChild(vb,vd)
            vb.setChild(va,vd)
            
        if (va.color >= 1)and(vc.color >= 1):
            if (va.id == ssid) or (vc.id == ssid):
                ve = va.getChild(vc)
        else:
            ve = sphere.addVertex(str(sphere.numVertices + 1))
            ve.setPosition(ep.tolist())
            newbv.append(ep.tolist())
            va.setChild(vc,ve)
            vc.setChild(va,ve)
        
        if (vb.color >= 1)and(vc.color >= 1):
            if (vc.id == ssid) or (vb.id == ssid):
             vf = vb.getChild(vc)
        else:
            vf = sphere.addVertex(str(sphere.numVertices + 1))
            vf.setPosition(fp.tolist())
            newbv.append(fp.tolist())
            vb.setChild(vc,vf)
            vc.setChild(vb,vf)

        va.color += 1
        vb.color += 1
        vc.color += 1
        
        #[['1', '2', '3'], ['1', '3', '5'], ['2', '3', '6'], ['3', '5', '9'], ['3', '6', '9']]第二个三角形的时候应该只加两个点
        sphere.delEdge(va,vb)
        sphere.delEdge(va,vc)
        sphere.delEdge(vb,vc)
        sphere.addTriangle(va,vd,vf)
        sphere.addTriangle(vd,vb,ve)
        sphere.addTriangle(vf,ve,vc)
        sphere.addTriangle(vf,vd,ve)

        #i -= 2 #这样并不能改变顺序，-2之后是4——》7
        newlocallist.append([va.id,vd.id,vf.id])
        newlocallist.append([vd.id,vb.id,ve.id])
        newlocallist.append([vf.id,ve.id,vc.id])
        newlocallist.append([vf.id,vd.id,ve.id]) #删1加四

    wholelist.append(newlocallist)
    

    return [sphere,wholelist,newbv]

def localcd(sphere,id): 
    """ 这部分需要用slicer判断相交性在slicer里写， 这部分返回local的b列表就行,
    script：1.生成带粒子号索引的相交的列表， dset
            2.相交判断
            3.比err,挑最优修改方向的函数"""
    pass 
    
def SubDivide(sphere): #现在由于没有去重，和huogh—sphere中的方向是对不上的，想用方向id必须使4层之后是1281，否则是不能找到的。。算了太恶心了，用b找
    num = sphere.triangles.size()
    v_num = sphere.numVertices
    for i in range(num):#啊这不就是传说中的死循环！！
        tri = sphere.triangles.removeRear()
        ai = tri[0]
        bi = tri[1]
        ci = tri[2]
        va = sphere.vertices[ai]
        ap = np.array(va.position)
        vb = sphere.vertices[bi]
        bp = np.array(vb.position)
        vc = sphere.vertices[ci]
        cp = np.array(vc.position)
        dp = ap + bp
        dp /= np.linalg.norm(dp)
        ep = ap + cp
        ep /= np.linalg.norm(ep)
        fp = bp + cp
        fp /= np.linalg.norm(fp)

        findd = False
        finde = False
        findf = False

        for j in range(1,sphere.numVertices):
            vj= sphere.vertices[str(j)]
            jp= np.array(vj.position)
            if(all(dp == jp)):
                findd = True
                vd = vj

            if(all(ep == jp)):
                finde = True
                ve = vj

            if(all(fp == jp)):
                findf = True
                vf = vj

        if not findd:
            vd = sphere.addVertex(str(sphere.numVertices + 1))
            vd.setPosition(dp.tolist())
            di = str(sphere.numVertices + 1)

        if not finde:
            ve = sphere.addVertex(str(sphere.numVertices + 1))
            ve.setPosition(ep.tolist())
            ei = str(sphere.numVertices + 1)

        if not findf:
            vf = sphere.addVertex(str(sphere.numVertices + 1))
            vf.setPosition(fp.tolist())
            fi = str(sphere.numVertices + 1)

        sphere.delEdge(va,vb)
        sphere.delEdge(va,vc)
        sphere.delEdge(vb,vc)
        sphere.addTriangle(va,vd,vf) #
        sphere.addTriangle(vd,vb,ve)
        sphere.addTriangle(vf,ve,vc)
        sphere.addTriangle(vf,vd,ve)

    return sphere

def findsphereid(sphere,bdata):
    #id 在Hough里可以打出来，方向对应的球体的id,可以通过needleid{2,7,12}索引用b找id
    #现在是找的sphere里误差最小的方向
    #*b = sphere->vertices[index];
    mind = 10 
    b = bdata
    #b = bdata[needleid]
    for v in sphere.vertices:
        vp =  np.array(sphere.vertices[v].position)
        deltad =  np.linalg.norm(b - vp)
        if mind > deltad:
            mind = deltad
            sidx = int(sphere.vertices[v].id)
            fb = vp      
    #print(sidx)   
    #print(type(idx)) 
    #print(mind) 
    #print(fb)    
    return sidx

def Genemodifytable(sphere,bv,n):
    #id 在Hough里可以打出来，方向对应的球体的id,可以通过needleid{2,7,12}索引用b找id
    #现在是找的sphere里误差最小的方向
    #*b = sphere->vertices[index];
    mind = 10 
    bt = np.array(bv)
    sid = str(findsphereid(sphere,bt))
    aim = sphere.vertices[sid]
    tabld = []
    tablv = []
    for v in sphere.vertices:
        aimp =aim.position 
        vp =  np.array(sphere.vertices[v].position)
        deltad =  np.linalg.norm(aimp - vp)
        tabld.append(deltad)
        
    ind = np.argsort(tabld)[0:n]
    ind = ind + 1
    tabld = np.sort(tabld)

    for i in range(len(ind)):
        vpp =  np.array(sphere.vertices[str(ind[i])].position)
        tablv.append(vpp)
      
    return tablv,tabld[0:n]


def LayerDivide(sphere,layer):
    for i in range(layer):
        SubDivide(sphere)
    
    return sphere


def drawpic(sphere,newdirection = None):
    x = []
    y = []
    z = []
    id = []
    for v in sphere.vertices:
        vp =  np.array(sphere.vertices[v].position)
        x.append(vp[0])
        y.append(vp[1])
        z.append(vp[2])
        id.append(sphere.vertices[v].id)
    fig = plt.figure()  # 设置画布大小
    ax = Axes3D(fig)  # 设置三维轴
    ax.scatter3D(x,y,z)  # 三个数组对应三个维度（三个数组中的数一一对应）
    ax.grid(False)
    #plt.savefig('3D.jpg', bbox_inches='tight', dpi=2400)  # 保存图片，如果不设置 bbox_inches='tight'，保存的图片有可能显示不全
    for i in range(len(x)):
        ax.text(x[i],y[i],z[i],id[i], zorder=1,  color='k')
    
    
    if(newdirection!=None):
        xv = []
        yv = []
        zv = []
        for l in range(len(newdirection)):
            nd = newdirection[l]
            xv.append(nd[0])
            yv.append(nd[1])
            zv.append(nd[2])
        ax.scatter3D(xv,yv,zv,c='g')
        for i in range(len(xv)):
            xsi = [0, xv[i]]
            #xs.append(xsi)
            ysi = [0, yv[i]]
            #ys.append(ysi)
            zsi = [0, zv[i]]
            #zs.append(zsi)
            ax.plot3D(xsi,ysi,zsi)
    plt.show()

def pfromcb(nb,NeedleS,l = 22.5):
    #cb\\方向
    #保留针头位置，调整针尾方向
    ps = NeedleS
    pe = NeedleS + [l*nb[0],l*nb[1],l*nb[2]]       
    return ps,pe

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

def findseedid(bs,dbs):
    """从b列表和剔除后的b列表求得删掉的id"""
    res = np.array(list(set(map(tuple,bs)) - set(map(tuple, dbs))))
    boolin = []
    for i in range(np.shape(bs)[0]):
        boolin.append(bs[i] in res)
    idx = np.argwhere(boolin)
    return idx

# def findseedid(fpts):
#     """用相交了的b找,返回针道索引"""
#在sphere里
#     pass

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
# def dirSubdivision(sphere,id = 3,dir):
#     return  sphere

# print(TAU)
# print(NR)
# l1 = np.linalg.norm([-V,TAU,0])#0.9999999999999999
# l2 = np.linalg.norm([0.8506508085336908,0,0.5257311118252163])#也是0.9999999999999999
#bb = [0.5257311118252163,  0.8506508085336908, 0]
#tsl1 = buildicosahedron()
#tsl5 = LayerDivide(tsl1,5)

# mtbl,dlt = Genemodifytable(tsl5,bb,110)
# #print(mtbl)
# mtbl = mtbl-np.array(bb)
# np.savetxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/ModifyTable5100.dat', mtbl, delimiter=',',fmt='%.8f')
# np.savetxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Modifydist.dat', dlt, delimiter=',',fmt='%.8f')
# for v in vl:
#     print(tsl2.vertices[v].getPosition())

#[0.5257311118252163, 0.8506508085336908, 0]
#tsl4 = LayerDivide(tsl1,4)
#bs = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Bvalues3.dat',delimiter=',')
#tsl4 = LayerDivide(tsl1,4)
#drawpic(tsl2)
#b1 = np.array([0.96870700,0.24301800,-0.05047710])
#iid = findsphereid(tsl3,bs,0)
#[ldtsl,wl,nbv] = localSubDivide(tsl3,iid)

# print(len(nbv))
# print(nbv)
#drawpic(tsl3,newdirection=nbv)##不太对
#Id = findsphereid(tsl4,b1)
# drawpic(tsl1)
#tsl4 = SubDivide(tsl1)
#drawpic(tsl4)
# print(bv)
#vv = tsl1.triangles.removeRear()
# v0 = tsl1.vertices[str(1)]
# print(v0)
#print(vv)#1 in['1', '2', '3']
#print(type(vv[0]))# str
#print(list(tsl1.vertices['3'].getConnections())) #单向不行，改成双向，双向可以了，
#print(tsl2.numEdge)#30,
#print(list(tsl4.triangles.))#20
#print(tsl2.numVertices)
# [ldtsl,wl,nbv] = localSubDivide(tsl1,3)
# print(nbv)
# print(wl)
#drawpic(tsl4)