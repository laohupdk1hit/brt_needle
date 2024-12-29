import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import Sphere.sph as sp #计算err用
import copy
import vtk
#换向针道最小覆盖组第0个
Needleend = np.array([[ 67.23782694, -21.59289206,  43.30489454],[ 66.86595653, -15.34293999, 43.70515054],[62.43773683, -22.49528295, 46.76663665],[68.33674106, -26.32004947, 48.5269012]])  #单位是mm
Needlestart = np.array([[117.90319446,  31.14845052,  26.68041745],[127.10302931, 7.80049057,  5.48371466],[133.00574554 ,-11.04281264 , 69.43816066],[127.33775619 , 19.35873456 , 40.95590647]])
c_Needlestart = np.array([148.331,-79.432,9.406])#固定针参数，应该有多个/1000
tumorcenter2ac = np.array([64.09397724, -21.32523201, 45.93846823])/1000

##base坐标系和eef坐标系和等待位置的q0，q0用作求逆解的初始值
mybasepos = SE3.Trans([0,0.7,0]) #单位是m
mybaseori = SE3.Ry(0) *SE3.Rz(0) #TODO
mytoolset = [0,0,0.26]  #TODObone
def unitchange(vin):
    v  = copy.copy(vin)
    v /= 1000
    return v

def roborinit(basepos,baseori,toolset):
    myrobot = rtb.models.UR5() 
    myrobot.base = basepos * baseori
    myrobot.tool = SE3.Trans(toolset)
    return myrobot

def ikine_q(rbt,nsin,nein): 
    ns = unitchange(nsin)
    ne = unitchange(nein)
    approach = ns - ne
    orientation = ns - tumorcenter2ac
    oa = approach / np.linalg.norm(approach)
    oo = orientation / np.linalg.norm(orientation)
    Rot = SE3.OA(oo,oa)
    ns  /= 1000
    Tep = SE3.Trans(ns) * Rot
    qn = rbt.ikine_LM(Tep)#最好指定q0
    return qn.q

def maniplty(ns,ne,rbt): #uc在ikine换过了
    q = ikine_q(rbt,ns,ne)#q0
    m = rbt.manipulability(q,axes = 'trans')
    return m

bons = [Needlestart[3][0]-5,Needlestart[3][0]+5],[Needlestart[3][1]-5,Needlestart[3][1]+5],[Needlestart[3][2]-5,Needlestart[3][2]+5]
print(bons)
print(type(bons))
myur5 = roborinit(mybasepos,mybaseori,mytoolset)
rbtreadypose = np.array([-90,-90,90,-90,-90,0])*np.pi/180 #
initialse3 = myur5.fkine(rbtreadypose)
initial_guess =np.array(initialse3.t) 
initial_guess.reshape(1,3) 
step = 0.1

x = np.linspace(bons[0][0],bons[0][1],num = 10)
y = np.linspace(bons[1][0],bons[1][1],num = 10)
z = np.linspace(bons[2][0],bons[2][1],num = 10)

X,Y,Z = np.meshgrid(x,y,z)
ns_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
np.savetxt('pts',ns_points,delimiter=',')
m = []
#result = np.array([maniplty(point, Needleend[0], myur5) for point in ns_points])
for point in ns_points:
    v = maniplty(point, Needleend[0], myur5)
    print(v)
    m.append(v)

np.savetxt('res',m,delimiter=',')