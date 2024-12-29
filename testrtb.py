from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np

#myur5.ik_LM(qwitht,end=myur5.links.)
# def dmaniplty2q(q):
#     delt = 0.0001
#     Eli = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     mdq = []
#     # q = myur5.ikine_LM(qtest)#算出来是不同的逆解
#     # print(q)
#     for i in range (0,6):
#         Eli[i] = 1.0
#         dmi = myur5.manipulability(q+delt*Eli,axes='rot')-myur5.manipulability(q-delt*Eli,axes='rot')
#         #dmi = myur5.manipulability(q+delt*Eli)-myur5.manipulability(q-delt*Eli)
#         mdq.append(dmi/(2*delt))  #1*6
#         Eli = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     return mdq

#print(dmaniplty2q(qtest))

#print(myur5.jacobm([0.1, -0.3, 0.5, -0.2, 0.2, 1],axes = 'rot'))

# def dmaniplty2x():
#     #q = ikine_q(ns,ne)#q0
#     mdq = myur5.jacobm(qtest)
#     j6 = myur5.jacob0(qtest)
#     print(mdq)
#     pinvj =np.linalg.pinv(j6[0:3,...]) #6*3
#     dm =  np.transpose(mdq) @ pinvj   #维度不对 应该是个1*3 
#     return dm

# print(dmaniplty2x())

#JACOBM（） 'all'与第一关节无关，'trans'是一样的，'rot'是一样的

""" geometric jacobian 和 analyticjacobian end-effector velocity is described in terms of translational and angular velocity, 
not a velocity twist as per the text by Lynch & Park 
 geometric jacobian 是矢量积,ana和描述方式有关，represent trtansform到base
 analytic jacobian 是相对base坐标系的旋量, 
 末关节的角速度描述为相对末关节坐标系的z轴旋转时，选择geo，用激光跟踪仪时用ana???"""

mybasepos = SE3.Trans([0,0.8,0]) 
mybaseori = SE3.Ry(-90) *SE3.Rz(-90)
mytoolset = [0,0,0.26]
q0 = np.array([-90,-90,90,-90,-90,0])*np.pi/180
def roborinit(basepos,baseori,toolset):
    myrobot = rtb.models.UR5() 
    myrobot.base = basepos *baseori
    myrobot.tool = SE3.Trans(toolset)
    return myrobot

myur5 = roborinit(mybasepos,mybaseori,mytoolset)
myur5.plot(q0,backend='pyplot')