# f = open('E:\Program Files\Slicer 4.11.20210226\slicerscripts\Index.dat')

# ID = f.readlines()
# f.close()
# del ID[0]
# print(ID)
# print(type(ID)) #str
# print(ID[0])

# f1 = open('E:\Program Files\Slicer 4.11.20210226\slicerscripts\Indexw.dat','w')
# f1.writelines(ID)
# f1.close()
# a=[1,2,3,2,3,4]
# b=[1,2,3,2,3,4]
# c=np.zeros((2,6),dtype=float)
# c[0]=a
# c[1]=b
# print(c)
# c1=c[:,0:3]
# print(c1)
# c2=c[:,3:6]
# print(c2)

import numpy as np
# # b=np.random.rand(10,3)

# a=np.random.randn(10,3) #高斯分布，均值为0
# print(a)
# def anglecheck(p1,p2,nb,v_max):
#     na = p2 - p1
#     cos = np.dot(na,nb)/(np.linalg.norm(na)*np.linalg.norm(nb))
#     acos = np.arccos(cos)
#     angle = np.abs(np.rad2deg(acos))
#     print(angle)
#     if(angle >= -v_max and angle <= v_max):
#         return True
#     else:
#         return False
# p1 =np.array([52,-9,61])
# p2 =np.array([110,10,30])
# nb =np.array([0.28980499505996704, 0.7861059904098511, -0.545939028263092])
# vm =50
# cr=anglecheck(p1,p2,nb,vm)
# print(cr)

startdata = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/NeedleStart4.dat',delimiter=',')
enddata  = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/NeedleEnds4.dat',delimiter=',')
adata  = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Avalues4.dat',delimiter=',')
bdata  = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Bvalues4.dat',delimiter=',')
nsdata  = np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/NeedleEnds_spatial4.dat',delimiter=',')

""" index不能这么整……和别的维度不一致。。。使用去重返回的id构造 """
f = open('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Index4.dat') 
Id = f.readlines()
f.close()

Pos = []
aa = []
bb = []
nss = []
id=[]
for i in range(np.shape(startdata)[0]):
    ps = np.concatenate((startdata[i],  enddata[i]), axis=None)
    a = adata[i]
    b = bdata[i]
    ns = nsdata[i]
    id.append(Id[i])
    Pos.append(ps)
    aa.append(a)
    bb.append(b)
    nss.append(ns)

Pos=np.array(Pos)
aa=np.array(aa)
bb=np.array(bb)
nss=np.array(nss)
Id = np.array(Id)
_,idp = np.unique(Pos,return_index=True,axis=0)
_,ida = np.unique(aa,return_index=True,axis=0)
_,idb = np.unique(bb,return_index=True,axis=0)
_,idn = np.unique(nss,return_index=True,axis=0)
newpos=Pos[np.sort(idp)]
newid=Id[np.sort(idp)]
newaa=aa[np.sort(ida)]
newbb=bb[np.sort(idb)]
newnss=nss[np.sort(idn)]

newps=newpos[:,0:3]
newpe=newpos[:,3:6]

np.savetxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/NeedleStart4.dat',newps, delimiter=',',fmt='%.8f') 
np.savetxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/NeedleEnds4.dat',newpe, delimiter=',',fmt='%.8f') 
np.savetxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Avalues4.dat',newaa, delimiter=',',fmt='%.8f') 
np.savetxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Bvalues4.dat',newbb, delimiter=',',fmt='%.8f') 
np.savetxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/NeedleEnds_spatial4.dat',newnss, delimiter=',',fmt='%.8f') 
f1 = open('E:/Program Files/Slicer 4.11.20210226/slicerscripts/Indexdel4.dat','w')
f1.writelines(newid)
f1.close()


