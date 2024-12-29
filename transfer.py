import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3,SO3
from spatialmath.base import *

ras2bot = np.array([[-0.999652, 0.018743, -0.0185558, 623.852],
       [ 0.0184689, -0.00480143, -0.999818, -1060.87],
       [-0.0188287, -0.999813, 0.0044536, 25.5172],
       [0, 0, 0, 1]])

def transf(path, center):
    transformations = []
    rotation_vectors = []

    for i in range(path.shape[0]):
        a = [0,0,0,1]
        b = [0,0,0,1]
        c = [0,0,0,1]
        a[:3] = [-path[i][0],-path[i][1],path[i][2]]
        b[:3] = [-path[i][3],-path[i][4],path[i][5]]
        c[:3] = center
    
        # 平移变换
        tf_a = ras2bot @ a  # SE3 对象
        tf_b = ras2bot @ b  # SE3 对象
        tf_c = ras2bot @ c  # SE3 对象

        # 计算方向向量
        approach = tf_b[:3] - tf_a[:3]  # 取平移部分 (3,)
        orientation = tf_c[:3] - tf_a[:3] # 取平移部分 (3,)
  

        # 归一化方向向量
        oa = approach / np.linalg.norm(approach)  # (3,)
        oo = orientation / np.linalg.norm(orientation)  # (3,)

        # 构造旋转矩阵
        Needle_rot = SO3.OA(oo, oa)  # 构造旋转部分
        adjust_rot = SO3.Rz(170,unit='deg')
        Needle_ad = Needle_rot @ adjust_rot
        Needle_trans_i = tf_a[:3]  # 平移部分保留 tf_a 的平移


        # 提取旋转轴和旋转角度
        rotation_axis, rotation_angle = Needle_ad.angvec()
        rotation_vector = rotation_axis * rotation_angle  # 旋转矢量
        # 保存每个变换的移动分量和旋转矢量
        transformations.append(Needle_trans_i)
        rotation_vectors.append(rotation_vector)

    return transformations, rotation_vectors

# needle_file = 'D:/Project/Needle_Path/scene2024experiment/Scene20241204_2/hough/Needle.dat'
# points_file = 'D:/code/TPS/fit_shape_seed2.dat'

needle_file = 'D:/Project/Needle_Path/scene2024experiment/Scene20241204_2/hough/Needles_mdf.dat'
points_file = 'D:/Project/Needle_Path/scene2024experiment/Scene20241204_2/hough/Points_mdf.dat'
#index_file = 'D:/Project/Needle_Path/scene2024experiment/Scene20241204_2/hough/Index.dat'
# 读取针道位置
try:
    needle_positions = np.loadtxt(needle_file, delimiter=',')
except Exception as e:
    raise FileNotFoundError(f"Error reading needle file: {e}")
print(needle_positions.shape)
# 读取粒子位置
try:
    points = np.loadtxt(points_file, delimiter=' ')
except Exception as e:
    raise FileNotFoundError(f"Error reading points file: {e}")

tumorcenter2ac = np.mean(points,axis=0) / 1000
print(tumorcenter2ac)

needle_positions = np.reshape(needle_positions, (-1, 6))
Needle_t, rotation_vector = transf(needle_positions, tumorcenter2ac)

# 输出结果
print("最终 SE3 姿态:")
print(Needle_t)
print("\n旋转矢量 (轴角表示):")
print(rotation_vector)

# 将结果保存为文件
output_file = "needle_transformations_and_rotations1_ad.txt"

with open(output_file, "w") as f:
    for i in range(len(Needle_t)):
        # 将平移和旋转矢量保存到文件，每行一个
        line = f"Translation: {Needle_t[i]} Rotation: {rotation_vector[i]}\n"
        f.write(line)

print(f"Results saved to {output_file}")

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


# def transf(path, center):
#     transformations = []
#     rotation_vectors = []

#     for i in range(path.shape[0]):
#         a = path[i,:3]
#         b = path[i,3:]
        
#         # 平移变换
#         tf_a = Tf * SE3.Trans(a)  # SE3 对象
#         tf_b = Tf * SE3.Trans(b)  # SE3 对象
#         tf_c = Tf * SE3.Trans(center)  # SE3 对象

#         # 计算方向向量
#         approach = tf_a.t - tf_b.t  # 取平移部分 (3,)
#         orientation = tf_a.t - tf_c.t  # 取平移部分 (3,)

#         # 归一化方向向量
#         oa = approach / np.linalg.norm(approach)  # (3,)
#         oo = orientation / np.linalg.norm(orientation)  # (3,)

#         # 构造旋转矩阵
#         Needle_rot = SO3.OA(oo, oa)  # 构造旋转部分
#         Needle_trans_i = tf_a.t.flatten()  # 平移部分保留 tf_a 的平移


#         # 提取旋转轴和旋转角度
#         rotation_axis, rotation_angle = Needle_rot.angvec()
#         rotation_vector = rotation_axis * rotation_angle  # 旋转矢量
#         # 保存每个变换的移动分量和旋转矢量
#         transformations.append(Needle_trans_i)
#         rotation_vectors.append(rotation_vector)

#     return transformations, rotation_vectors







