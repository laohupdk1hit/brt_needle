import numpy as np
import Sphere.sph as sp
from __main__ import vtk,slicer

def anglecheck(p1,p2,nb,v_max):
    na = p2 - p1
    cos = np.dot(na,nb)/(np.linalg.norm(na)*np.linalg.norm(nb))
    acos = np.arccos(cos)
    angle = np.abs(np.rad2deg(acos))
    angle -= 90
    #print(angle)
    if(angle >= -v_max and angle <= v_max):
        return (True,angle)
    else:
        return (False,angle)

def buildbsp(obstacle):
    #obb=getNode('Model_2_bone.vtk')
    bspTree = vtk.vtkModifiedBSPTree()
    bspTree.SetDataSet(obstacle.GetPolyData())
    bspTree.BuildLocator()
    return bspTree

def itsectcheck(bsp,p1, p2,tolerance=0.001, tb=vtk.mutable(0.0), xb=[0.0, 0.0, 0.0], pcoordsb=[0.0, 0.0, 0.0], subIdb=vtk.mutable(0)):
    checkintsct = bsp.IntersectWithLine(p1, p2, tolerance, tb, xb, pcoordsb, subIdb)
    return checkintsct

def transfer(sdata,edata,tf = getNode('TargettoRAS')):
    #transform
    # transformNode = slicer.vtkMRMLTransformNode()
    # slicer.mrmlScene.AddNode(transformNode)
    # mrHead.SetAndObserveTransformNodeID(transformNode.GetID())
    Transf = slicer.util.arrayFromTransformMatrix(tf)
    # array([[  0.89611647,   0.42734952,   0.11978173,  64.09397724],
    #        [  0.14055984,  -0.52926923,   0.83672995, -21.32523201],
    #        [  0.42097293,  -0.73297099,  -0.53435505,  45.93846823],
    #        [  0.        ,   0.        ,   0.        ,   1.        ]]
    Trf = np.reshape(Transf,(4,4))
    Ps = np.array([sdata[0],sdata[1],sdata[2],1])
    Pe = np.array([edata[0],edata[1],edata[2],1])
    #p = np.matmul(Trf,Ps)
    PositionStart = np.matmul(Trf,Ps)[0:3]
    #print(PositionStart)
    PositionEnd= np.matmul(Trf,Pe)[0:3]
    #print(PositionEnd)
    return PositionStart,PositionEnd

def minerrdir(pts,ids,Ns,idx,c_bs,obs): #p1 array([ 67.23782694, -21.59289206,  43.30489454]),,p1经过transf
    points = []
    for l in range(len(ids)):
        points.append(pts[int(ids[l])])#索引的是粒子号,这应该是比如说第一个针道{0, 1, 2, 3, 4, 5, 6, 7}
    minerr= 1000
    pm1= np.array([0,0,0])
    pm2 = np.array([1,0,0])
    ndls = Ns[idx]
    for indx in range(np.shape(c_bs)[0]):
        b = c_bs[indx]
        t = b * (points - ndls)
        err = []
        for ll in range(np.shape(points)[0]):
            err.append(np.linalg.norm(points[ll] - (ndls + t[ll]*b))) 
        terr = np.sum(err)/np.shape(points)[0]
        pp1,pp2 = sp.pfromcb(b,ndls)#这里是原坐标
        trsp1,trsp2 = transfer(pp1,pp2)
        localflag = itsectcheck(obs,trsp1,trsp2)#需要transf！！！！
        if (localflag == 0 and minerr > terr): 
            minerr = terr
            pm1 = pp1
            pm2 = pp2
    return pm1,pm2,minerr            

def makepath(path, approachablePoints, color, modelName):
    # Create an array for all approachable points 
    # 这个有问题，是polyline，原来的代码只有一个target，因此可以用
    # 需要定义connnectivity
    # 每条针道单独画
    scene = slicer.mrmlScene   
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    if approachablePoints != 0:
      for i in range(int(len(path)/2)):
        pointst = path[2*i].tolist()
        pointed = path[2*i+1].tolist()
        points.InsertNextPoint(pointst)
        points.InsertNextPoint(pointed)
        line  = vtk.vtkLine()
        line.GetPointIds().SetId(0,2*i)
        line.GetPointIds().SetId(1,2*i+1)
        lines.InsertNextCell(line) 
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    # Create model node
    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetName(scene.GenerateUniqueName(modelName))
    model.SetAndObservePolyData(polyData)
    # Create display node
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetColor(color[0], color[1], color[2])
    modelDisplay.SetScene(scene)
    modelDisplay.SetVisibility2D(True) # Show in slice view
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    # Add to scene
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    scene.AddNode(model) #显示在slicer
    return (model, modelDisplay, points) 

slicer.util.loadMarkupsFiducialList('E:/Program Files/Slicer 5.2.2/slicerscripts/data/pointcloud/point3.fcsv')
# tipdata = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/NeedleStart3.dat',delimiter=',')
# shaftdata = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/NeedleEnds3.dat',delimiter=',')
needlepath = np.loadtxt('E:/Program Files/Slicer 5.2.2/Slicerscripts/houghdata/NeedleP3.dat',delimiter=',')
# bdata = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/Bvalues3.dat',delimiter=',')
# adata = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/Avalues3.dat',delimiter=',')
# bs = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/Bvalues3.dat',delimiter=',')
# dbs = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/del_b3.dat',delimiter=',')
points3 = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/point3.dat',delimiter=',')
# f1 = open('E:/Program Files/Slicer 5.2.2/licerscripts/houghdata/IndexP3.dat')
# indexlist = f1.readlines()
# f1.close()
# f2 = open('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/Index3w.dat')
# newindexlist = f2.readlines()
# f2.close()
# difset = sp.Diffset(indexlist,newindexlist)
# print(difset)#相差粒子号
# indx =sp.findseedid(bs,dbs)
# print(indx.T)#相差针道号
# woyaodeid = list(sp.invcheck(difset,indx.T,indexlist))
# print(woyaodeid) #4最小覆盖针道号
# np.reshape(shaftdata, (-1,3) )
# np.reshape(tipdata , (-1,3) )
nodecount = np.shape(needlepath)[0]
PositionShaft = []
PositionTip = []
for i in range(nodecount):
    ps,pt = transfer(needlepath[i][0:3],needlepath[i][3:6])
    PositionShaft.append(ps)
    PositionTip.append(pt)

yellow = [1, 1, 0]
red = [1, 0, 0]
green = [0, 1, 0]
blue = [0, 0, 1]
#build bsptree of bones
xb = [0.0, 0.0, 0.0]
obb=getNode('Model_2_bone.vtk')
bspTreebone = buildbsp(obb)
#build bsptree of skin
xs = [0.0, 0.0, 0.0]
obs=getNode('Model_135_right_external_oblique_muscle.vtk')
bspTreeskin = buildbsp(obs)
#build bsptree of liver
# xl = [0.0, 0.0, 0.0]
# obl=getNode('Model_3_liver.vtk')
# bspTreeliver = buildbsp(obl)
# #build liver normal vector filter
# """ 肝脏交点法向量计算 """
# normFilter = vtk.vtkPolyDataNormals()
# normFilter.SetInputData(obl.GetPolyData())
# normFilter.SetComputePointNormals(0)    # 开启点法向量计算
# normFilter.SetComputeCellNormals(1)    # 关闭单元法向量计算
# normFilter.SetAutoOrientNormals(1)
# normFilter.SetSplitting(0)
# # normFilter.Update()
# """ 这时候已经算出肝脏上所有三角片的normal了，需要提取相交的cell/point
# vtkArrayDownCast<vtkFloatArray>(output->GetPointData()->GetNormals())
# or with vtkArrayDownCast<vtkFloatArray>(output->GetPointData()->GetArray("Normals"))
# 在主循环中进行交点方向的判断和提取 """
# """ 存储为.dat """
# lpd=obl.GetPolyData()
# Ln=lpd.GetPointData().GetNormals()
# #liver locator
# cellId = vtk.reference(0)
# c = [0.0, 0.0, 0.0]
# subId = vtk.reference(0)
# d = vtk.reference(0.0)
# my_cl = vtk.vtkCellLocator()
# my_cl.SetDataSet(obl.GetPolyData())  # reverse.GetOutput() --> vtkPolyData
# my_cl.BuildLocator()
allpath=[]
for i in range(nodecount):
    p1 = PositionShaft[i]
    p2 = PositionTip[i]    
    allpath.append(p1)
    allpath.append(p2)
    #modelReceived, allPaths, allPathPoints =makepath(path,2,1, blue, "allPaths%d"%(i)) #不行，会有一堆model

modelReceived, allPaths, allPathPoints =makepath(allpath, nodecount, blue, "allPaths")
#iD = []
# itspos = []
# bvalues=[]
# avalues=[]
# del_ps = []
# del_pe = []
# del_pos = []
# del_poe = []
# PoS =np.array(PositionStart)
# PoE =np.array(PositionEnd)
okpath = []
itspath = []
OKid = []
ITSid = []
# citspath = []
# normb = []
# me  = []
# tempps = []
for i in range(nodecount):
    p1 = PositionShaft[i]
    p2 = PositionTip[i] 
    checkintsctb = itsectcheck(bspTreebone,p1,p2)#骨头，血管等OaR的并集,先用骨头
    checkintscts = itsectcheck(bspTreeskin,p1,p2)
    # checkintsctl = itsectcheck(bspTreeliver,p1,p2)
    # my_cl.FindClosestPoint(xl, c, cellId, subId, d)
    # bn = np.array(Ln.GetTuple3(cellId))
    #print('i'+str(i))
    #anglecheck(np.array(p1),np.array(p2),bn,70)
        # """ #get到的是13967！！！！！！！！！gettuple从0开始，数组从1开始 >>> LN.GetTuple3(40715)
        # Traceback (most recent call last):
        # File "<console>", line 1, in <module>
        # ValueError: expects 0 <= tupleIdx && tupleIdx < GetNumberOfTuples() """
    if (checkintsctb == 0 and checkintscts == 1):#可达并且不和OaR相交
        #path = []
        okpath.append(p1)
        okpath.append(p2)
        OKid.append(i)
        #iD.append(indexlist[j])
        # del_ps.append(p1)
        # del_pe.append(p2)
        # del_pos.append(startdata[j])
        # del_poe.append(enddata[j])
        # bvalues.append(bdata[j])
        # avalues.append(adata[j])
        # itspos.append(xxs) #for循环陷阱
        # if anglecheck(np.array(p1),np.array(p2),bn,70)[0]:
        #     normb.append(anglecheck(np.array(p1),np.array(p2),bn,70)[1])
        #     #modelReceived, cPaths, cPathPoints =makeonepath(path, nodecount,1,green, "readyPaths%d"%(j))
        # else :
        #     normb.append(anglecheck(np.array(p1),np.array(p2),bn,70)[1])
            #modelReceived, cPaths, cPathPoints =makeonepath(path, nodecount,1,yellow, "readyPaths%d"%(j))
    elif(checkintsctb == 1 and checkintscts == 1):#可达，但和OaR相交，进行换向：##不应该对所有相交的进行换向，只对我要的换向！！
        # print('its')
        itspath.append(p1)
        itspath.append(p2)
        ITSid.append(i)

""" >>> print(OKid)
[1, 2, 4, 5, 7, 8]
>>> print(ITSid)
[0, 3, 6, 9] """

# for j in range(len(woyaodeid)):#不用放主循环吧。。。。。
#     #localflag = itsectcheck(bspTreebone,p1,p2)
#     tsl1 = sp.buildicosahedron()
#     tsl3 = sp.LayerDivide(tsl1,3) #每次重新生成sp
#     iid = sp.findsphereid(tsl3,bs)
#     print(iid)
#     _,_,nbv = sp.localSubDivide(tsl3,iid) #后面循环没有生成nbv,导致只能算一次,#现在不用局部重分，找到iid就可以？
#     sti = str(indexlist[woyaodeid[j]])
#     sti = sti.replace('\n','')
#     ss = sti.split(',')
#     #print('ss:'+str(ss))  ss:['72', '73', '74', '75', '76', '77', '78', '79']
#     ppp1,ppp2,mer = minerrdir(points3,ss,shaftdata,woyaodeid[j],nbv,bspTreebone)#p1也和主循环无关，因为需要invcheck..还是从NS读
#     #print('mer'+str(j)+':'+str(mer))
#     citspath.append(ppp1)
#     citspath.append(ppp2)
#     me.append(mer)
            
#print(me) #len(me) = 36 ,不对，应该只有四个      
modelReceived, cPaths, cPathPoints =makepath(okpath, nodecount,blue , "readyPaths")
modelReceived, dPaths, dPathPoints =makepath(itspath, nodecount,red , "itsPaths")
modelReceived, ePaths, ePathPoints =makepath(citspath, nodecount,green , "citsPaths")

# iD = polygon.IntersectWithLine(p1, p2, tolerance, t, x, pcoords, subId)

#     print('intersected? ', 'Yes' if iD == 1 else 'No')
#     print('intersection: ', x)
# x ：直线与平面相交于哪点。RAS绝对坐标
# p1、p2所在直线 与 多边形所在平面 的相交点的坐标，若平行则无限大：世界RAS坐标
# -9.25596e+061 -9.25596e+061 -9.25596e+061。

# pcoords ：线段与面相交与哪点。
# p1、p2所构成的 线段 与 多边形所在平面 的相交点的坐标，无则 0 0 0

# p1=[52,-9,61]
# p2=[110,10,30]
# tolerance = 0.001
# t = vtk.mutable(0.0)
# x = [0.0, 0.0, 0.0] # The coordinate of the intersection 
# pcoords = [0.0, 0.0, 0.0]
# subId = vtk.mutable(0)
# #ob=getNode('Model_3_liver.vtk')
# ob=getNode('Model_135_right_external_oblique_muscle.vtk')
# bspTree = vtk.vtkModifiedBSPTree()
# bspTree.SetDataSet(ob.GetPolyData())
# bspTree.BuildLocator()
# checkintsct = bspTree.IntersectWithLine(p1, p2, tolerance, t, x, pcoords, subId)
# checkintsct
# 1
# x
# [92.85923507006808, 4.384921833298167, 39.16144332461878]


# # Create a 4x4 transformation matrix as numpy array
# transformNode = ...
# transformMatrixNP = np.array(
#   [[0.92979,-0.26946,-0.25075,52.64097],
#   [0.03835, 0.74845, -0.66209, -46.12696],
#   [0.36608, 0.60599, 0.70623, -0.48185],
#   [0, 0, 0, 1]])

# # Update matrix in transform node
# transformNode.SetAndObserveMatrixTransformToParent(slicer.util.vtkMatrixFromArray(transformMatrixNP))

#    transformMatrix = vtk.vtkMatrix4x4()
#    transformNode.GetMatrixTransformToWorld(transformMatrix)
#    print("Position: [{0}, {1}, {2}]".format(transformMatrix.GetElement(0,3), transformMatrix.GetElement(1,3), transformMatrix.GetElement(2,3)))

# tf = getNode('TargettoRAS')

#polydata->GetCellData()->GetArray("Normals"))

# # get line cell ids as 1d array
# line_ids = vtk_to_numpy(fb.GetPolyData().GetLines().GetData())

# vtkSmartPointer<vtkIdList> cellPointIds =
#   vtkSmartPointer<vtkIdList>::New();
# //根据cellID获取顶点索引信息
# polydata->GetCellPoints(cellid, cellPointIds);
# for(vtkIdType i = 0; i < cellPointIds->GetNumberOfIds(); i++){
#   mesh->GetCellData()->GetNormals()->GetNormal(cellId)

# lines = vtk.vtkCellArray()
# >>> line1  = vtk.vtkline
# Traceback (most recent call last):
#   File "<console>", line 1, in <module>
# AttributeError: module 'vtkmodules.all' has no attribute 'vtkline'
# >>> line1  = vtk.vtkline()
# Traceback (most recent call last):
#   File "<console>", line 1, in <module>
# AttributeError: module 'vtkmodules.all' has no attribute 'vtkline'
# >>> line1  = vtk.vtkLine()
# >>> p1=[0,1,0]
# >>> p2=[1,0,0]
# >>> p3 = [0,0,1]
# >>> points = vtk.vtkPoints()
# >>> points.InsertNextPoint(p1)
# 0
# >>> points.InsertNextPoint(p2)
# 1
# >>> points.InsertNextPoint(p3)
# 2
# >>> c
# >>> lines = vtk.vtkCellArray()
# lines.InsertNextCell(line1)
# 0
# >>> line1.GetPointIds().SetId(0,1)
# >>> polydata = vtk.vtkPolyData()
# >>> polydata.SetPoints(points)
# >>> polydata.SetLines(lines)
# >>> scene = slicer.mrmlScene 
# >>> model = slicer.vtkMRMLModelNode()
# modelDisplay
# (MRMLCorePython.vtkMRMLModelDisplayNode)000001E088C26D08
# >>> vtk.VTK_MAJOR_VERSION
# 8
# >>> slicer.util.loadMarkupsFiducialList("E:/Program Files/Slicer 5.2.2/slicerscripts/addf2.fcsv")
# [True, (vtkSlicerMarkupsModuleMRMLPython.vtkMRMLMarkupsFiducialNode)000001E088C26BE8]
# >>> scene.AddNode(modelDisplay)
# (MRMLCorePython.vtkMRMLModelDisplayNode)000001E088C26D08
# >>> model.SetAndObservePolyData(polydata)
# >>> model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
# >>> scene.AddNode(model)
# (MRMLCorePython.vtkMRMLModelNode)000001E088C26588
# >>> 