import numpy as np
import pandas as pd
from scipy import interpolate
#from tg43.Iridium192 import Ir_192
from tg43.I125 import I_125
import json

def Dose_Rate2D(Position,calc_point,fuente,PlanDate):
    deltaPlanCal = (PlanDate-fuente.CalDate).days + (PlanDate-fuente.CalDate).seconds/(24*3600)

    Sk = fuente.RAKR*np.exp(-np.log(2)*deltaPlanCal/fuente.MeanLife)
    r = calc_point-Position
    a = pd.DataFrame([Position.loc[i,['x','y','z']] - Position.loc[i+1,['x','y','z']] if i!=len(Position)-1 else Position.loc[i-1,['x','y','z']] - Position.loc[i,['x','y','z']] for i in range(len(Position))]) 
    r_dot_a=[r.iloc[i].dot(a.iloc[i]) for i in range(len(r))]
    r['modulo_r'] = r.apply(lambda x:np.linalg.norm(x), axis='columns')
    a['modulo_a'] = a.apply(lambda x:np.linalg.norm(x), axis='columns')

    theta = [np.degrees(np.arccos(r_dot_a[i]/(r.modulo_r[i]*a.modulo_a[i]))) for i in range(len(r_dot_a))]

    a_norm = a[['x','y','z']].apply(lambda x:x/np.linalg.norm(x), axis='columns')

    r1 = calc_point - (Position - a_norm*fuente.length/20)
    r1_dot_a=[r1.iloc[i].dot(a[['x','y','z']].iloc[i]) for i in range(len(r1))]
    r1['modulo_r1'] = r1.apply(lambda x:np.linalg.norm(x), axis='columns')
    theta_1 = np.array([np.degrees(np.arccos(r1_dot_a[i]/(r1.modulo_r1[i]*a.modulo_a[i]))) for i in range(len(r1_dot_a))])

    r2 = calc_point - (Position + a_norm*fuente.length/20)
    r2_dot_a=[r2.iloc[i].dot(a[['x','y','z']].iloc[i]) for i in range(len(r2))]
    r2['modulo_r2'] = r2.apply(lambda x:np.linalg.norm(x), axis='columns')
    theta_2 = np.array([np.degrees(np.arccos(r2_dot_a[i]/(r2.modulo_r2[i]*a.modulo_a[i]))) for i in range(len(r2_dot_a))])

    beta = np.radians(theta_2 - theta_1)

    GL0 = 2*np.arctan(fuente.length/20)/(fuente.length/10)
    GL_r_th = np.array([1/(r.modulo_r[i]**2-(fuente.length/20)**2) if (theta[i]==0 or theta[i]==180) else beta[i]/((fuente.length/10)*r.modulo_r[i]*np.sin(np.radians(theta[i]))) for i in range(len(beta))])

    g_r = np.interp(r.modulo_r*10,fuente.RadialDoseFuntion['r(mm)'],fuente.RadialDoseFuntion['g(r)'])

    x,y = np.meshgrid(np.linspace(0,180,37),np.linspace(0,50,11))
    # Anisotropy2D.drop('r(mm)\\theta(°)',axis=1)
    f = interpolate.interp2d(x,y,np.array(fuente.Anisotropy2D),kind='cubic')

    F_r_th = np.array([(f(theta[i],r.modulo_r[i]*10))[0] for i in range(len(r))])
    
    return Sk*fuente.DoseRateConstant*(GL_r_th/GL0)*g_r*F_r_th.T

def Dose_Rate1D(Position,xrange,yrange,zrange,fuente):
    #unit mm
    X, Y, Z = np.meshgrid(xrange, yrange, zrange)
    Rs = np.sqrt((X-Position[0])**2 + (Y-Position[1])**2 + (Z-Position[2])**2)
    Rs = Rs / 10 # unit change 2 cm
    Sk = fuente.Sk
    it = np.nditer(Rs)
    Geo_fun = []
    for r in it:
        if r >= 1:
            geo_fun = 1/r**2
        else:
            geo_fun = (r**2-0.25*fuente.length**2)/(1-0.25*fuente.length**2)
        Geo_fun.append(geo_fun)
    g_f = np.array(Geo_fun).reshape((np.shape(xrange)[0],np.shape(yrange)[0],np.shape(zrange)[0]))
    g_r = np.interp(Rs,fuente.RadialDoseFuntion['r(cm)'],fuente.RadialDoseFuntion['g(r)'])
    phi_r = np.interp(Rs,fuente.Phyani['r(cm)'],fuente.Phyani['phi(r)'])
    dd = Sk*fuente.DoseRateConstant*g_f*g_r*phi_r
    point_cloud = np.stack((X, Y, Z, dd), axis=3)
    #（XYZ,DOSE）
    return point_cloud ,dd

def act(time,A0 = 0.5):
    LAMDA = -0.01168 #fuente.MeanLife/np.log(2)
    return 1.270 * A0 * np.exp(LAMDA * time) #Sk = 1.27

def cumulative_dose_3d(dose_rate_3d, t_end):
    '''疗程内总规划剂量cGY t_end:day'''
    
    # Define the time array
    time_array = np.linspace(0,t_end * 24,num = 100)
    
    # Calculate the time step  = 1hr
    dt = 1
    
    # Initialize 3D array for cumulative dose
    cumulative_dose_3d = np.zeros_like(dose_rate_3d)
    
    # Iterate over all spatial points
    for x in range(dose_rate_3d.shape[0]):
        for y in range(dose_rate_3d.shape[1]):
            for z in range(dose_rate_3d.shape[2]):
                for i in range(len(time_array)-1):
                    t = time_array[i]
                    t1 = time_array[i+1]
                    # Perform numerical integration 
                    step_dose = dose_rate_3d[x,y,z] * (act(t) + act(t1)) * (t1 - t) / 2
                    cumulative_dose_3d[x,y,z] += step_dose
    return cumulative_dose_3d
   

def Dose_Distribution(Catheters, Calc_Matrix, fuente, PlanDate):
    """
    This funtion return a matrix of dose in the space
    """
    DoseperMatrix=[]
    for calc_point in Calc_Matrix:
        DoseperCatheter=[]
        for Position in Catheters:
            DoseRate=Dose_Rate2D(Position[['x','y','z']],calc_point,fuente,PlanDate)
            DoseperDwell = DoseRate*np.array(Position['time']/3600)
            DoseperCatheter.append(round(DoseperDwell.sum(),2))
        DoseperCatheter = np.array(DoseperCatheter)
        DoseperMatrix.append(DoseperCatheter.sum())
    return DoseperMatrix

def Dose_Distribution1D(Seedpos, xrange,yrange,zrange, fuente):
    """
    This funtion return a matrix of dose in the space 
    time:h
    """
    Doserate_pc = np.zeros((np.shape(xrange)[0],np.shape(yrange)[0],np.shape(zrange)[0],4))
    DoseRate = np.zeros((np.shape(xrange)[0],np.shape(yrange)[0],np.shape(zrange)[0]))

    for Position in Seedpos:
        dpcdose , ddose =Dose_Rate1D(Position,xrange,yrange,zrange,fuente)
        Doserate_pc += dpcdose
        DoseRate += ddose
    
    return Doserate_pc,DoseRate

if __name__ == '__main__':
    I125seed = I_125(1,Sk=0.635)
    # 读取JSON文件
    with open('hn_plan.json', 'r') as file:
        data = json.load(file)

    seedpos = []

    for needle_path in data["needle_paths"]:
        seeds_n = needle_path["seed position"]
        for sdpos in seeds_n:
          seedpos.append(sdpos)

    seedpos = np.array(seedpos)
    #74.07, -10.46, 46.34
    #+1
    # x = np.linspace(62-30, 62+50, 33)
    # y = np.linspace(-17-40, -17+40, 33)
    # z = np.linspace(36-40, 36+40, 33) ##hero的lesion position
    # 输出结果
    x_max = max(seedpos[:,0]) 
    x_min = min(seedpos[:,0]) 
    y_max = max(seedpos[:,1]) 
    y_min = min(seedpos[:,1]) 
    z_max = max(seedpos[:,2]) 
    z_min = min(seedpos[:,2]) 

    print(x_min,x_max,y_min,y_max,z_min,z_max) 
     
    # x2 = np.linspace(-101.756, 147.755591, 32)
    # y2 = np.linspace(124.756, 374.267591, 32)
    # z2= np.linspace(-1, 188, 12) 
    x2 = np.linspace(351.267591, 101.756, 64)
    y2 = np.linspace(-124.756, -374.267591, 64)
    z2 = np.linspace(-190, -1, 24) 
    #np.linspace()
    #seedpos = np.loadtxt('point_data/bign1tf.dat',delimiter=' ')
    #dose1d = Dose_Rate1D(np.array([0,0,0]),x,y,z,I125seed)
    #print(seedpos)
    #dspc,dronly = Dose_Distribution1D(seedpos,x1,y1,z1,I125seed)
    #_,dr1  =  Dose_Distribution1D(seedpos,x1,y1,z1,I125seed)
    drpc2,dr2  =  Dose_Distribution1D(seedpos,x2,y2,z2,I125seed)
    #print(np.max(dr1))
    print(np.max(dr2))
    #print(dd1d)# unit = cGy/h 
    #print(np.shape(dspc))
    #print(np.shape(dronly))

    #dose_cumul = cumulative_dose_3d(dr1,t_end=7)
    dose_cumu2 = cumulative_dose_3d(dr2,t_end=7)
    #print(np.max(dose_cumul))
    print(np.max(dose_cumu2))

    np.save('beihangplandspc.npy',drpc2)
    #np.save('point_data/bign1tfdos.npy',dronly)
    np.save('beihangplan.npy',dose_cumu2)

    # for i in range(points.shape[0]):
    #     for j in range(points.shape[1]):
    #         for k in range(points.shape[2]):
    #             point = points[i, j, k]
    #             image_data.SetScalarComponentFromDouble(i, j, k, 0, point[3])

    # for i in range(dronly.shape[0]):
    #     for j in range(dronly.shape[1]):
    #         for k in range(dronly.shape[1]):
    #             dv = dronly[i, j, k]

    #print(np.load('point_data\p3dos.npy'))





