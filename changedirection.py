import scipy.optimize as opt
import numpy as np
from __main__ import vtk,slicer
from scipy.spatial import distance_matrix
import time
import copy
def buildbsp(obstacle):
    #obb=getNode('Model_2_bone.vtk')
    bspTree = vtk.vtkModifiedBSPTree()
    bspTree.SetDataSet(obstacle.GetPolyData())
    bspTree.BuildLocator()
    return bspTree

def itsectcheck1(bsp,p1, p2,tolerance=0.001, tb=vtk.mutable(0.0), xb=[0.0, 0.0, 0.0], pcoordsb=[0.0, 0.0, 0.0], subIdb=vtk.mutable(0)):
    checkintsct = bsp.IntersectWithLine(p1, p2, tolerance, tb, xb, pcoordsb, subIdb)
    return checkintsct

def makepath(path, approachablePoints, color, modelName):
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
    modelDisplay.SetLineWidth(3)
    modelDisplay.SetScene(scene)
    modelDisplay.SetVisibility2D(True) # Show in slice view
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    # Add to scene
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    scene.AddNode(model) #显示在slicer
    return (model, modelDisplay, points) 

def eefcheck(dm,eefconst =40):
    len = dm.shape[0]
    checkmatrix = np.zeros([len, len])
    for i in range(0,len):
        for j in range(0,len):
            if (dm[i,j]<eefconst):
                checkmatrix[i,j] = dm[i,j]

    rows, cols = np.where(checkmatrix != 0.)
    repeatedpairs = np.array(list(zip(rows, cols)))
    truepairnum = int(0.5 * repeatedpairs.shape[0])
    pairs = repeatedpairs[0:truepairnum,:]
    return list(pairs.flatten())

yellow = [1, 1, 0]
red = [1, 0, 0]
green = [0, 1, 0]
blue = [0, 0, 1]
purple = [0.5,0,0.5]


obbone = getNode('Bones')
obskin = getNode('surface')
oboar = getNode('OAR')
needlepath = np.loadtxt('D:/Project/Needle_Path/scene2024experiment/Scene20241204_4/hough/Needle_v1.dat',delimiter=',')
nodecount = np.shape(needlepath)[0]
PositionShaft = []
PositionTip = []
for i in range(nodecount):
    # PositionShaft.append(np.array([-needlepath[i][0],-needlepath[i][1],needlepath[i][2]]))
    # PositionTip.append(np.array([-needlepath[i][3],-needlepath[i][4],needlepath[i][5]]))
    PositionShaft.append(np.array([-needlepath[i][0],-needlepath[i][1],needlepath[i][2]]))
    PositionTip.append(np.array([-needlepath[i][3],-needlepath[i][4],needlepath[i][5]]))

allpath=[]
for i in range(nodecount):
    p1 = PositionShaft[i]
    p2 = PositionTip[i]    
    allpath.append(p1)
    allpath.append(p2)
    #modelReceived, allPaths, allPathPoints =makepath(path,2,1, blue, "allPaths%d"%(i)) #不行，会有一堆model

modelReceived, allPaths, allPathPoints =makepath(allpath, nodecount, green, "N1_allPaths")

#TODO:replace 换了的针道
#PositionShaft[4] = np.array([115.35356637, 14.18558705, 46.53810282])
dm = distance_matrix(PositionShaft,PositionShaft)
bspTreebone = buildbsp(obbone)
bspTreeskin = buildbsp(obskin)
bspTreeoar = buildbsp(oboar)
#obscure detection
okpath = []
itspath = []
eefpath = []
OKid = []
ITSid = []
EEFid = eefcheck(dm)
for i in range(nodecount):
    p1 = PositionShaft[i]
    p2 = PositionTip[i] #i是针道 
    checkintsctb = itsectcheck1(bspTreebone,p1,p2)#骨头，血管等OaR的并集,先用骨头
    checkintscts = itsectcheck1(bspTreeskin,p1,p2)
    checkintscto = itsectcheck1(bspTreeoar,p1,p2)
    #if (checkintsctb == 0 and checkintscts == 1):#可达并且不和OaR相交
    if (checkintsctb == 0):
        #path = []
        okpath.append(p1)
        okpath.append(p2)
        OKid.append(i)
    #elif(checkintsctb == 1 and checkintscts == 1):#可达，但和OaR相交，进行换向：##不应该对所有相交的进行换向，只对我要的换向！！
    if (checkintsctb == 1 or checkintscto == 1):
        itspath.append(p1)
        itspath.append(p2)
        ITSid.append(i)

eefshaft=[]

for id in EEFid:
    eefshaft.append(PositionShaft[id]) 
    eefpath.append(PositionShaft[id])
    eefpath.append(PositionTip[id])
eefshaft = np.array(eefshaft)

modelReceived, cPaths, cPathPoints =makepath(okpath, nodecount,blue , "readyPaths")
modelReceived, dPaths, dPathPoints =makepath(itspath, nodecount,red , "itsPaths")
modelReceived, ePaths, ePathPoints =makepath(eefpath, nodecount,yellow , "eefPaths")
modelReceived, gPaths, gPathPoints =makepath(eefpath, nodecount,green , "goodPaths")

""">>> ITSid
[2, 4, 6, 8, 9] """

Needletip=[]
Needleshaft=[]
fix_Needleshaft = []
#ITSid
for id in ITSid:
    Needletip.append(PositionTip[id])
Needletip = np.array(Needletip)

for id in ITSid:
    Needleshaft.append(PositionShaft[id]) 
Needleshaft = np.array(Needleshaft)
for id in OKid:
    fix_Needleshaft.append(PositionShaft[id]) 
fix_Needleshaft = np.array(fix_Needleshaft)


Needletip = np.array([[  24.73313195,   33.06845741, -978.24197484]])
Needleshaft = np.array([[  -81.27006092,  -101.69274641, -1033.03924709]])
fix_Needleshaft = np.array([[  -73.68772388,    -9.40279073, -1115.46545538],
       [  -81.27006092,  -101.69274641, -1033.03924709],
       [  -55.55360433,  -109.92073444, -1049.48183798],
       [   78.58989343,  -145.45767625,  -927.02230311],
       [  -97.04324468,   -78.03343803,  -945.2647606 ],
       [   16.26965463,  -126.36170372, -1062.65138822],
       [  -21.66081501,   -77.64821204, -1098.90605554],
       [   30.08937495,  -133.29613922, -1049.23312781]])#固定针参数，应该有多个/1000
 #单位是mm

# fix_add = list(fix_Needleshaft)
# fix_add.append(Needletip[0])
# fix_add = np.array(fix_add)
# fix_Needleshaft = fix_add

#ITSid = [0, 1, 2, 7]
EEFid = [1]
tumorcenter2ac = np.array([-3.33249420e-02, -2.20050357e-04, -1.06035503e+00])
ctimagescale = 0.5

def vtktransfer1(pdata,tf = getNode('vtk')):
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
    Ps = np.array([pdata[0],pdata[1],pdata[2],1])
    #p = np.matmul(Trf,Ps)
    PositionStart = np.matmul(Trf,Ps)[0:3]
    #print(PositionStart)
    #print(PositionEnd)
    return PositionStart

def unitchange(vin):
    v  = copy.copy(vin)
    v /= 1000
    return v

# def roborinit(basepos,baseori,toolset):
#     myrobot = rtb.models.UR5() 
#     myrobot.base = basepos * baseori
#     myrobot.tool = SE3.Trans(toolset)
#     return myrobot

def buildcellfinder(obstacle):
    cf = vtk.vtkCellLocator()
    cf.SetDataSet(obstacle.GetPolyData())  # reverse.GetOutput() --> vtkPolyData
    cf.BuildLocator()
    return cf

def itsectcheck(bspobs,p1, p2,tolerance=0.001, points = vtk.vtkPoints(),cell_ids = vtk.vtkIdList() ):
    #points直线与bsp相交于哪些点。RAS绝对坐标
    tumorcenter = np.array([-3.33249420e-02, -2.20050357e-04, -1.06035503e+00])
    checkintsct = 0
    checkintsct = bspobs.IntersectWithLine(p1, p2, tolerance, points,cell_ids)
    n =  points.GetNumberOfPoints()
    pt = []
    d0 = 0.1
    for i in range(n):
        ptvs = np.array(points.GetPoint(i))
        dist = np.linalg.norm(ptvs-tumorcenter)
        if dist>d0:
            d0 = dist
            pt = points.GetPoint(i)
    return checkintsct ,pt

def localsphere(itsp,obsdata,scope = 10):
    box = vtk.vtkBox() 
    #scope窗口，需要包含一根肋骨，但不能包含两根
    box.SetBounds(itsp[0]-scope,itsp[0]+scope,itsp[1]-scope,itsp[1]+scope,itsp[2]-scope,itsp[2]+scope)
    model = obsdata.GetPolyData()
    # 创建vtkClipPolyData对象并设置截取
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(model)
    clipper.SetClipFunction(box)
    clipper.InsideOutOn()#取内部
    clipper.Update()
    clippedModel = clipper.GetOutput()
    localobb = vtk.vtkOBBTree()
    localobb.SetDataSet(clippedModel)
    localobb.BuildLocator()
    corner = [0.0]*3  # 存储OBB包围盒的角点坐标，相当于OOB包围盒的“原点”位置。
    max_point = [0.0]*3  # 存储OBB包围盒的最大点坐标，主轴的方向向量。
    mid_point = [0.0]*3  # 存储OBB包围盒的中间点坐标，次轴的方向向量。
    min_point = [0.0]*3  # 存储OBB包围盒的最小点坐标，最次轴的方向向量。
    size = [0.0]*3  # 存储OBB包围盒相对于轴的排序列表，点max的位置= corner + max，依次类推。
    localobb.ComputeOBB(clippedModel, corner, max_point, mid_point, min_point, size)
    p0 = np.array(corner)
    p1 = np.array(max_point)
    p2 = np.array(mid_point)
    p3 = np.array(min_point)
    radius = np.linalg.norm(p1)/2
    #spnum = int(np.ceil(np.linalg.norm(p1)/np.linalg.norm(p2)))
    spnum = 6
    centers = []
    for i in range(0,spnum):
        ratio = (i+1)/spnum
        cnp = p0 + ratio*p1 +0.5*p2 +0.5*p3 
        centers.append(cnp)   
    return radius,centers


def add_sphere_alternative(name="Sphere", center=(0, 0, 0), radius=10.0):
    """
    替代方法添加球体到场景。
    """
    # 创建模型节点
    sphere_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
    if sphere_model_node is None:
        raise RuntimeError("Failed to create sphere model node.")

    # 创建球体几何
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(center)
    sphere_source.SetRadius(radius)
    sphere_source.Update()

    # 将几何设置到模型节点
    sphere_model_node.SetAndObservePolyData(sphere_source.GetOutput())

    # 创建显示节点
    display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
    display_node.SetColor(1, 0, 0)  # 红色
    display_node.SetOpacity(0.7)   # 半透明
    slicer.mrmlScene.AddNode(display_node)
    sphere_model_node.SetAndObserveDisplayNodeID(display_node.GetID())

    print(f"Added alternative sphere '{name}' at {center} with radius {radius}.")
    return sphere_model_node

def ccellfind(cf,ns,cp = [0.0, 0.0, 0.0],cell = vtk.vtkGenericCell(), cellId = vtk.mutable(0),subId = vtk.mutable(0), d = vtk.reference(0.0)):
    #cf cellfinder locator,ne:target,ns：优化变量，针尾位置
    #从ne到ns取20个点 #现在的问题是，对直线距离最近的不一定是同一块骨头
    cf.FindClosestPoint(ns, cp,cellId, subId, d)  
    return cp,d

# cfp,_ = ccellfind(cfbone,Needleshaft[0])
# rd,cts = localsphere(cfp,obbone)
# print(rd)
# print(cts)

def chuizu(askingpoint,ndle,c_ndls): #返回垂足坐标和垂距
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



def errofns(x,target,pts,ids): #ss:['0', '16', '20', '33', '38', '39', '']
    '''换向针道所属粒子，对新针道的平均误差，'''
    points = []
    for l in range(len(ids)):
        pttobetf = pts[int(ids[l])-1]
        #pttobetf = pts[ids[l]]
        #pt = transfer1(pttobetf)
        pt = vtktransfer1(pttobetf)
        points.append(pt)##索引的是粒子号,这应该是比如说第一个针道{0, 1, 2, 3, 4, 5, 6, 7}
    err = 0
    for n in range(len(points)):
        _,dist = chuizu(points[n],target,x)
        err += dist
    meanerr = np.sum(err)/len(points)#排除粒子数影响

    return meanerr


#uc
dlta =0.1
# myur5 = roborinit(mybasepos,mybaseori,mytoolset)
# rbtreadypose = np.array([-90,-90,90,-90,-90,0])*np.pi/180 #
# initialse3 = myur5.fkine(rbtreadypose)
# initial_guess =np.array(initialse3.t) 
# initial_guess.reshape(1,3) 
const_length = 190
const_eef = 40
const_bone = 40 #const_l 肋骨参数，15
bsptreebone = buildbsp(obbone)
cfbone = buildcellfinder(obbone)

def cons1(x,target):     #变量得是cns坐标,针长约束 target是固定针尖,eq
    return np.linalg.norm(x - target) - const_length
    #return const_l-np.linalg.norm(x - target)

def cons2(x,target,localc): #ineq,不干涉约束,cf现在是对一条线了 ############
    # cfp,dr = ccellfind(cfbone,target,x)
    # _,mindist = closest(cfp,dr)
    #return  delta- mindist
    _,dist = chuizu(localc,target,x)
    return  dist - rd - dlta 

def cons3(x,target,itsp): #ineq,局部约束,不跨骨头约束 
    # cfp,dr = ccellfind(cfbone,target,x)
    #mincp,_ = closest(cfp,dr)
    #return np.linalg.norm(mincp - ini_itsp)- const_singlebone_dist
    _,dist = chuizu(itsp,target,x)
    return const_bone - dlta - dist 

def cons4(x,fix_ns_poss):#针尾距离约束
    fps = np.array(fix_ns_poss)#初始固定针尾
    xx = np.array(x)
    ds = []
    dx = xx - fps
    for i in range(0,dx.shape[0]):
        ds.append(np.linalg.norm(dx[i])) 
    ds = np.array(ds)
    #return const_nsfix_dist - ds.min() 
    return ds.min() - const_eef

def cons5(xin,targetin,ini_nsin,rbt): #uc
    x = unitchange(xin)
    target = unitchange(targetin)
    ini_ns= unitchange(ini_nsin)
    #可操作度梯度约束
    dmx = dmaniplty2x(x,target,rbt)
    #return -np.dot(x-ini_ns,dmx[0]) 
    return np.dot(x-ini_ns,dmx[0]) 

# cons = ()
# cons = ({'type': 'eq', 'fun': lambda x: cons1(x,Needleend)},
#         {'type': 'ineq', 'fun': lambda x: cons3(x,Needleend,itsp_ini)},
#         {'type': 'ineq', 'fun': lambda x: cons4(x,c_Needlestart)},
#         {'type': 'ineq', 'fun': lambda x: cons5(x,Needleend,Needlestart,myur5)})
                                                
# for localcenter in cts:
#     cons = cons + ({'type': 'ineq', 'fun': lambda x: cons2(x,Needleend,localcenter)},)

#mm
###################
rdpoints1 = np.loadtxt('D:/Project/Needle_Path/scene2024experiment/Scene20241204_4/hough/fit_shape_seed4.dat',delimiter=',')
# bs = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/Bvalues3.dat',delimiter=',')
# dbs = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/functest/del_b3.dat',delimiter=',')
#########
f1 = open('D:/Project/Needle_Path/scene2024experiment/Scene20241204_4/hough/Index.dat')
indexlist = f1.readlines()
f1.close()

f,itsp_ini= itsectcheck(bspTreeoar,Needletip[1],Needleshaft[1])
rd,cts = localsphere(itsp_ini,oboar,scope=10)
print(rd)
print(cts)
#drawspheres(rd,cts)
for ct in cts:
    sphere = add_sphere_alternative(str(i), ct, rd)


bons =  opt.Bounds([Needleshaft[1][0]-25,Needleshaft[1][1]-25,Needleshaft[1][2]-25],[Needleshaft[1][0]+25,Needleshaft[1][1]+25,Needleshaft[1][2]+25] )

##TODO#####TODO 这个每次都要换的吧####
sti = str(indexlist[3]) #indexlist是str
sti = sti.replace('\n','')
ss = sti.split(',')
ss = ss[:-1]
print('ss:'+str(ss))  #ss:['0', '16', '20', '33', '38', '39', '']

arguments_of_obj  = (Needletip[1],rdpoints1,ss)

def constraint_func(x):
    cons = []
    c1 = cons1(x, Needletip[1])
    c3 = cons3(x, Needletip[1], itsp_ini ) #如果不交，找最近点，然后球包络
    c4 = cons4(x, fix_Needleshaft )
    #c5 = cons5(x, Needletip[3], Needleshaft[3], myur5)
    #cons = [c1, c3, c4, c5]
    #cons = [c1, c4]
    cons = [c1,c3,c4]
    for localcenter in cts:
        cons.append(cons2(x,Needletip[1],localcenter))
    return cons
 
#有c5时是10个，无c5是9个
lb = np.array([0,0,0,0,0,0,0,0,0])
ub = np.array([0,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
nlc = opt.NonlinearConstraint(constraint_func, lb, ub )
#修正末尾不和骨头相交时是8个
# lb = np.array([0,0,0,0,0,0,0,0])
# ub = np.array([0,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
# nlc = opt.NonlinearConstraint(constraint_func, lb, ub )

starttime = time.time()
#result = opt.minimize(errofns,args = arguments_of_obj,x0=Needleshaft[1], method = 'trust-constr',  constraints = nlc, bounds =bons)
result = opt.minimize(errofns,args = arguments_of_obj,x0=Needleshaft[1]+np.array([0,0,10]), method = 'SLSQP',  constraints = nlc, bounds =bons)
endtime= time.time()
print(result)
print(endtime - starttime, 's')
'''cons2每次从图形中搜索,并且有可能跨肋骨搜索，肋骨的形状必然是非凸的，这就导致整个问题非凸。。 尝试缩小求解范围，使得局部是凸的，
如果不行，在局部用圆柱包络吧'''
# res = opt.differential_evolution(errofns,bounds=bons,args=arguments_of_obj,constraints = nlc,maxiter = 100)
# print(res)
'''求解时间过长，maxiter改为100仍然不行'''
'''用6个球做局部包络行了'''

def modify(Ntip,xx,pts,ids): #ss:['0', '16', '20', '33', '38', '39', '']
    '''换向针道所属粒子，对新针道的平均误差，'''
    points = []
    mdf_seed = []
    for l in range(len(ids)):
        pttobetf = pts[int(ids[l])-1]
        #pttobetf = pts[ids[l]]
        #pt = transfer1(pttobetf)
        pt = vtktransfer1(pttobetf)
        points.append(pt)##索引的是粒子号,这应该是比如说第一个针道{0, 1, 2, 3, 4, 5, 6, 7}
    for n in range(len(points)):
        modified_sd,_ = chuizu(points[n],Ntip,xx)
        mdf_seed.append(modified_sd)
    return mdf_seed


x00 = np.array([132.51014471,  27.56838713,  46.19396012])
modified_seed = modify(Needletip[3],x00,rdpoints1,ss)
print(modified_seed)
nonmpt = []
for i in range(len(rdpoints1) ):
    pt = vtktransfer1(rdpoints1[i])
    nonmpt.append(list(pt))

nonmpt = np.array(nonmpt)

np.savetxt('E:/Program Files/Slicer 5.2.2/1/bign5tf.dat',nonmpt)
for i in range(len(ss) ):
    idx = int(ss[i]) - 1
    nonmpt[idx] = modified_seed[i]

np.savetxt('E:/Program Files/Slicer 5.2.2/1/S5mdf.dat',nonmpt)

# def transfer(sdata,edata,tf = getNode('tumor3')):
#     #transform
#     # transformNode = slicer.vtkMRMLTransformNode()
#     # slicer.mrmlScene.AddNode(transformNode)
#     # mrHead.SetAndObserveTransformNodeID(transformNode.GetID())
#     Transf = slicer.util.arrayFromTransformMatrix(tf)
#     # array([[  0.89611647,   0.42734952,   0.11978173,  64.09397724],
#     #        [  0.14055984,  -0.52926923,   0.83672995, -21.32523201],
#     #        [  0.42097293,  -0.73297099,  -0.53435505,  45.93846823],
#     #        [  0.        ,   0.        ,   0.        ,   1.        ]]
#     Trf = np.reshape(Transf,(4,4))
#     Ps = np.array([sdata[0],sdata[1],sdata[2],1])
#     Pe = np.array([edata[0],edata[1],edata[2],1])
#     #p = np.matmul(Trf,Ps)
#     PositionStart = np.matmul(Trf,Ps)[0:3]
#     #print(PositionStart)
#     PositionEnd= np.matmul(Trf,Pe)[0:3]
#     #print(PositionEnd)
#     return PositionStart,PositionEnd

# def vtktransfer(sdata,edata,tf = getNode('vtkonly')):
#     Transf = slicer.util.arrayFromTransformMatrix(tf)  
#     Trf = np.reshape(Transf,(4,4))
#     Ps = np.array([sdata[0],sdata[1],sdata[2],1])
#     Pe = np.array([edata[0],edata[1],edata[2],1])
#     #p = np.matmul(Trf,Ps)
#     PositionStart = np.matmul(Trf,Ps)[0:3]
#     #print(PositionStart)
#     PositionEnd= np.matmul(Trf,Pe)[0:3]
#     #print(PositionEnd)
#     return PositionStart,PositionEnd
# ##base坐标系和eef坐标系和等待位置的q0，q0用作求逆解的初始值
# mybasepos = SE3.Trans([0,0.7,0]) #单位是m
# mybaseori = SE3.Ry(0) *SE3.Rz(0) #TODO
# mytoolset = [0,0,0.26]  #TODObone
# q0 = np.array([-90,-90,90,-90,-90,0])*np.pi/180
# def ikine_q(rbt,nsin,nein): 
#     ns = unitchange(nsin)
#     ne = unitchange(nein)
#     approach = ns - ne
#     orientation = ns - tumorcenter2ac
#     oa = approach / np.linalg.norm(approach)
#     oo = orientation / np.linalg.norm(orientation)
#     Rot = SE3.OA(oo,oa)
#     ns  /= 1000
#     Tep = SE3.Trans(ns) * Rot
#     qn = rbt.ikine_LM(Tep)#最好指定q0
#     return qn.q

# def dmaniplty2x(ns,ne,rbt): #uc在cons5换过了
#     q = ikine_q(rbt,ns,ne)#q0
#     mdq = rbt.jacobm(q)
#     j6 = rbt.jacob0(q)
#     pinvj =np.linalg.pinv(j6[0:3,...]) #6*3
#     dm =  np.transpose(mdq) @ pinvj   
#     return dm


# def transfer1(pdata,tf = getNode('tumor5')):
#     #transform
#     # transformNode = slicer.vtkMRMLTransformNode()
#     # slicer.mrmlScene.AddNode(transformNode)
#     # mrHead.SetAndObserveTransformNodeID(transformNode.GetID())
#     Transf = slicer.util.arrayFromTransformMatrix(tf)
#     # array([[  0.89611647,   0.42734952,   0.11978173,  64.09397724],
#     #        [  0.14055984,  -0.52926923,   0.83672995, -21.32523201],
#     #        [  0.42097293,  -0.73297099,  -0.53435505,  45.93846823],
#     #        [  0.        ,   0.        ,   0.        ,   1.        ]]
#     Trf = np.reshape(Transf,(4,4))
#     Ps = np.array([pdata[0],pdata[1],pdata[2],1])
#     #p = np.matmul(Trf,Ps)
#     PositionStart = np.matmul(Trf,Ps)[0:3]
#     #print(PositionStart)
#     #print(PositionEnd)
#     return PositionStart

# Postf = []
# for i in range(count):
#     tps = transfer1(points1[i][0:3])
#     Postf.append(list(tps))