import numpy as np
import math
#import torch
from spatialmath import SE3
import roboticstoolbox as rtb
import Sphere.sph as sp

#x.grad为Dy/dx(假设Dy为最后一个节点)
def gradient_descent(x0, k, f, eta): #迭代k次,包括x0在内共k+1个数
    # 初始化计算图参数
    x = torch.tensor(x0, requires_grad=True)
    for i in range(1, k+1):
        y = f(x)
        y.backward() 
        with torch.no_grad(): 
            x -= eta*x.grad
        x.grad.zero_()  #这里的梯度必须要清0，否则计算是错的
    x_star = x.detach().numpy()
    return f(x_star), x_star 

# 多元函数，但非向量函数（指返回值为向量）
def f(x):
    return 5*x[0]**4 + 4*x[0]**2*x[1] - x[0]*x[1]**3 + 4*x[1]**4 - x[0]

if __name__ == '__main__':
    x0 = np.array([1.0, -1.0])
    k = 25 # k为迭代次数
    eta = 0.01 # ita为迭代步长
    minimum, x_star = gradient_descent(x0, k, f, eta)
    print("the minimum is %.5f, the x_star is: ( %.5f, %.5f)"\
        % (minimum, x_star[0], x_star[1]))

# my_ur5_related = ur5.initialize()
# qnIn = matlab.double([0.1, 0.1, 0.2, 0.2, 0.2, 0.2], size=(1, 6))

# jinv_nOut = my_ur5_related.jinv_ur5(qnIn)
# print('jinv:',jinv_nOut,sep='\n')

# pdm_nOut = my_ur5_related.mpdq_ur5(qnIn)
# print('pdm:',pdm_nOut,sep='\n')

# mplty_nOut = my_ur5_related.mplty_ur5(qnIn)
# print('mplty:',mplty_nOut,sep='\n')

# my_ur5_related.terminate()

pts = np.array([(97.89268493652344, 10.318024635314941, 33.2463264465332), (94.51429748535156, 6.801211357116699, 34.354854583740234)])
Needleend =np.array( [ 67.23782694, -21.59289206,  43.30489454])
Needlestart = np.array([117.90319446,  31.14845052,  26.68041745])
c_Needlestart = np.array([148.331,-79.432,9.406])#测试换向
tumorcenter = np.array([64.09397724, -21.32523201, 45.93846823])
rbtUR5 = rtb.models.UR5()
# Ne = np.array(Needleend)
# CNs = np.array(Needlestart)+np.array([5,5,5])
# Its = np.array(pts)
# es = Ne - CNs
# ed = Ne - Its
# en1 = np.dot(es,ed) / (np.linalg.norm(es)) ^2
# print(en1)
# en = np.dot(en1,es)
# print(en)

# panda = rbt.models.Panda()
# pandaets = rbt.models.Panda().ets()
# print(pandaets.eval(panda.qr))


def findmin(ne,ns,nt):
    #cf cellfinder locator,ne:target,ns：优化变量，针尾位置
    #从ne到ns取20个点
    pts = np.linspace(ne,ns,20)
    mind = 100000
    cpt = []
    for i in range(0,20): 
        pt = pts[i]
        d = np.linalg.norm(nt - pt)
        if d < mind:
            mind = d #不循环
            cpt = pt
            print(mind)
    return cpt,mind

# tpt,dt = findmin(Needleend,c_Needlestart,Needlestart)
# print(tpt)
# print(dt)

def cons4(x,fix_ns_poss,const_nsfix_dist):#针尾距离约束
    fps = np.array(fix_ns_poss)
    xx = np.array(x)
    ds = np.linalg.norm((xx - fps),axis=1)
    return ds.min() - const_nsfix_dist

#print(cons4(c_Needlestart,[Needleend,Needlestart],100))

def ori_z0(ns,ne,tcenter):
    #计算对于肿瘤中心的旋转矩阵
    approach = ns - ne
    orientation = ns - tcenter
    oa = approach / np.linalg.norm(approach)
    oo = orientation / np.linalg.norm(orientation)
    Rot = SE3.OA(oo,oa)
    return Rot

def ikine_q(ns,ne): 
    approach = ns - ne
    orientation = ns - tumorcenter
    oa = approach / np.linalg.norm(approach)
    oo = orientation / np.linalg.norm(orientation)
    Rot = SE3.OA(oo,oa)
    ns /= 1000
    Tep = SE3.Trans(ns) * Rot
    qn = rbtUR5.ikine_LM(Tep)
    return qn.q
# print(ori_z0(Needlestart,Needleend,tumorcenter))
# Needlestart  /= 1000
# Tep = SE3.Trans(Needlestart) * ori_z0(Needlestart,Needleend,tumorcenter)
# print(Tep)
# #Tep需要变换到base坐标
# rbtUR5 = rtb.models.UR5()
# qm = rbtUR5.ikine_LM(Tep)
#print(type(ikine_q(Needlestart,Needleend))) 这里是隐患，用copy()
# mplty = rbtUR5.manipulability(qm.q,axes='trans')
# print(mplty)
# print(type(np.linalg.inv(rbtUR5.jacob0(qm.q))))
#SE3.inv(rbtUR5.jacob0())
def dmaniplty2x(ns,ne):
    delt = 0.0001
    Eli = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mdq = []
    q = ikine_q(ns,ne)#q0
    for i in range (0,6):
        Eli[i] = 1.0
        dmi = rbtUR5.manipulability(q+delt*Eli,axes='trans')-rbtUR5.manipulability(q-delt*Eli,axes='trans')
        mdq.append(dmi/(2*delt))  #1*6
        Eli = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    j6 = rbtUR5.jacob0(q)
    pinvj =np.linalg.pinv(j6[0:3,...]) #6*3
    dm =  np.transpose(mdq) @ pinvj   #维度不对 应该是个1*3 
    return mdq,dm
#q=[-0.99267596,  1.37768191,  2.89132527,  1.13374428,  1.25311028,  1.59689903]
#print(rbtUR5.manipulability(q,axes='all'))
manipu ,manipu2x = dmaniplty2x(Needlestart,Needleend)
print(manipu) 
print(manipu2x)

