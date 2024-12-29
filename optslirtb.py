from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np
import scipy.optimize as opt

myur5 = rtb.models.UR5()
mybasepos = SE3.Trans([0,0.8,0]) 
mybaseori = SE3.Ry(-90) *SE3.Rz(-90)
mytoolset = SE3.Trans([0,0,0.26])
q00 = np.array([-90,-90,90,-90,-90,0])*np.pi/180
tumorcenter = np.array([64.09397724, -21.32523201, 45.93846823])
Needleend =np.array( [ 67.23782694, -21.59289206,  43.30489454])/1000
Needlestart = np.array([117.90319446,  31.14845052,  26.68041745])/1000 #[117.90319446,  31.14845052,  26.68041745]外部
c_Needlestart =  np.array([148.331,-79.432,9.406])/1000#测试换向
itsp_ini = [94.514, 6.801, 34.355]
myur5.base = mybasepos *mybaseori
myur5.tool = mytoolset
pts = np.linspace(Needleend,Needlestart,20)
print(pts[0])
def ikine_q(ns,ne): 
    approach = ns - ne
    orientation = ns - tumorcenter
    oa = approach / np.linalg.norm(approach)
    oo = orientation / np.linalg.norm(orientation)
    Rot = SE3.OA(oo,oa)
    ns  /= 1000
    Tep = SE3.Trans(ns) * Rot
    qn = myur5.ikine_LM(Tep)#最好指定q0
    return qn.q

def dmaniplty2x(ns,ne):
    q = ikine_q(ns,ne)
    mdq = myur5.jacobm(q)
    j6 = myur5.jacob0(q)
    print(mdq)
    pinvj =np.linalg.pinv(j6[0:3,...]) #6*3
    dm =  np.transpose(mdq) @ pinvj   #维度不对 应该是个1*3 
    np.reshape(dm,(1,3))
    return dm


def ccellfind(a,b):
    return 0.5 *(a-b), 0.3 *(a-b)

def err(x,argo):
    return np.linalg.norm(x - argo)

def cons1(x,target,const_l):     #变量得是cns坐标,针长约束 target是固定针尖,eq
    return np.linalg.norm(x - target) - const_l

def cons2(x,target,delta): #ineq,不干涉约束,cf现在是对一条线了
    _,mindist = ccellfind(x,target)
    return mindist - delta

def cons3(x,target,ini_itsp,const_singlebone_dist): #ineq,局部约束,不跨骨头约束
    mincp,_ = ccellfind(x,target)
    return const_singlebone_dist - np.linalg.norm(mincp - ini_itsp)

def cons4(x,fix_ns_poss,const_nsfix_dist):#针尾距离x约束
    ds = np.sqrt((x[0]-fix_ns_poss[0])**2 + (x[1]-fix_ns_poss[1])**2+(x[2]-fix_ns_poss[2])**2)
    return ds- const_nsfix_dist

def cons5(x,target,ini_ns):
    #可操作度梯度约束,单位
    dmx = dmaniplty2x(x,target)
    return np.dot(x-ini_ns,dmx[0]) 

Needleend =np.array( [ 67.23782694, -21.59289206,  43.30489454])
Needlestart = np.array([117.90319446,  31.14845052,  26.68041745]) #[117.90319446,  31.14845052,  26.68041745]外部
c_Needlestart = [148.331,-79.432,9.406]#测试换向
itsp_ini = [94.514, 6.801, 34.355]

const_length = 75
const_eef = 60
const_bone = 15 #const_l 肋骨参数，15
dlta = 0.01

cons = ({'type': 'eq', 'fun': lambda x: cons1(x,Needleend,const_length)},
        {'type': 'ineq', 'fun': lambda x: cons2(x,Needleend,dlta)},
        {'type': 'ineq', 'fun': lambda x: cons3(x,Needleend,itsp_ini,const_bone)},
        {'type': 'ineq', 'fun': lambda x: cons4(x,c_Needlestart,const_eef)},
        {'type': 'ineq', 'fun': lambda x: cons5(x,Needleend,Needlestart)})

initialse3 = myur5.fkine(q00)
initial_guess = initialse3.t

arguments_of_obj  = 0.3

result = opt.minimize(err,args = arguments_of_obj,x0=initial_guess, method = 'SLSQP',  constraints =cons)
#err的变量为x,1*3
print(result)




# x0 = np.array([0.5, 0])
# res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
#                constraints=[linear_constraint, nonlinear_constraint],
#                options={'verbose': 1}, bounds=bounds)