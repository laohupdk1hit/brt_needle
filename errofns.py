import numpy as np
import Sphere.sph as sp
import scipy.optimize as opt
def chuizu(askingpoint,ndle,c_ndls): #itspoint (有俩，返回最小距离的垂足坐标和距离）
    #return垂足位置 1*3
    Ne = np.array(ndle)
    CNs = np.array(c_ndls)
    Its = np.array(askingpoint)
    
    es = Ne - CNs
    ed = Ne - Its
    k= np.dot(es,ed) / ((np.linalg.norm(es))**2)
    en = k * es

    tcz = ndle - en #垂足坐标
    tl = np.linalg.norm(Its - tcz) 
    return tcz,tl

def transfer(ptstotf):
    #transform
    # transformNode = slicer.vtkMRMLTransformNode()
    # slicer.mrmlScene.AddNode(transformNode)
    # mrHead.SetAndObserveTransformNodeID(transformNode.GetID())
    Transf = np.array([[0.89611647,   0.42734952,   0.11978173,  64.09397724],
                       [0.14055984,  -0.52926923,   0.83672995, -21.32523201],
                       [0.42097293,  -0.73297099,  -0.53435505,  45.93846823],
                       [0.        ,   0.        ,   0.        ,   1. ]])
    Trf = np.reshape(Transf,(4,4))
    Ps = np.array([ptstotf[0],ptstotf[1],ptstotf[2],1])
    #p = np.matmul(Trf,Ps)
    psintumor = np.matmul(Trf,Ps)[0:3]
    return psintumor

def errofns(x,target,pts,ids): 
    '''换向针道所属粒子，对新针道的平均误差，'''
    points = []
    for l in range(len(ids)):
        pttobetf = pts[int(ids[l])]
        pt = transfer(pttobetf)#需要换坐标系
        points.append(pt)#索引的是粒子号,这应该是比如说第一个针道{0, 1, 2, 3, 4, 5, 6, 7}
    err = 0
    for n in range(len(points)):
        _,dist = chuizu(points[n],target,x)
        err += dist
    meanerr = np.sum(err)/len(points)#排除粒子数影响
    return meanerr    
       

points3 = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/point3.dat',delimiter=',')
bs = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/Bvalues3.dat',delimiter=',')
dbs = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/del_b3.dat',delimiter=',')
f1 = open('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/Indexdel3.dat')
indexlist = f1.readlines()
f1.close()
f2 = open('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/Index3w.dat')
newindexlist = f2.readlines()
f2.close()
difset = sp.Diffset(indexlist,newindexlist)
print(difset)#相差粒子号
indx =sp.findseedid(bs,dbs)
print(indx.T)#相差针道号
woyaodeid = list(sp.invcheck(difset,indx.T,indexlist))
print(woyaodeid) #4最小覆盖针道号
sti = str(indexlist[woyaodeid[0]]) #indexlist是str
sti = sti.replace('\n','')
ss = sti.split(',')
print('ss:'+str(ss))  #ss:['16', '24', '25', '26', '27', '28', '29', '30', '31']
c_Needlestart = np.array([[148.331,-79.432,9.406],[108.7,-20,30]])
Needleend =np.array( [ 67.23782694, -21.59289206,  43.30489454])
Needlestart = np.array([117.90319446,  31.14845052,  26.68041745])

ini = [-0.16912585,  0.92058731, -0.62637263]
xget = [115.71727403,  33.26253954,  27.00689363]

print(errofns(xget,Needleend,points3,ss))

