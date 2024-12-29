import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull,Delaunay
import open3d

# 读取 STL 文件
stl_mesh = mesh.Mesh.from_file('tumorshr4.stl')
vertices = stl_mesh.vectors.reshape(-1, 3)  # 展平为一个 N x 3 的点云
voxel_size=5

min_bounds = np.min(vertices, axis=0)
max_bounds = np.max(vertices, axis=0)

# 生成体素网格
x_range = np.arange(min_bounds[0], max_bounds[0], voxel_size)
y_range = np.arange(min_bounds[1], max_bounds[1], voxel_size)
z_range = np.arange(min_bounds[2], max_bounds[2], voxel_size)
grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
voxel_grid = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

center = 0.5*(min_bounds+max_bounds)

# 2. 平移点云到原点
points_centered = voxel_grid - center

def rotate(angle_degrees, axis='z'): 
    angle = np.radians(angle_degrees)
    if axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    return  rotation_matrix

# 旋转 50 度绕 Y 轴
roty = rotate(-100, axis='y')
rotx = rotate(-30, axis='x')
rotz = rotate(60, axis='z')

# 3. 旋转点云
rotated_points = points_centered @ rotx.T @ roty.T @ rotz.T
 
# 4. 移回
points_rotated = rotated_points + center

# 提取 STL 中的所有顶点
hull = ConvexHull(vertices)

delaunay = Delaunay(vertices[hull.vertices])

# 判断哪些体素点在Delaunay三角剖分中
inside_voxels = points_rotated[delaunay.find_simplex(points_rotated) >= 0]


# 加载点云 (使用之前的体素点云 inside_voxels)
points = inside_voxels

# 将点云转换为 Open3D 点云对象
pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(points)


# 执行泊松盘采样
sampled_cloud = pcd.farthest_point_down_sample(num_samples=20)

# 将采样后的点云转回 numpy 数组
sampled_voxels = np.asarray(sampled_cloud.points)

np.savetxt('fit_shape_seed4.dat',sampled_voxels,fmt='%1.6f',delimiter=',')


# 可视化 
# 原始体素点
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(inside_voxels[:, 0], inside_voxels[:, 1], inside_voxels[:, 2], s=1, alpha=0.5, label="Original Voxels")
ax1.set_title("Original Voxels")
ax1.legend()

# 均匀采样点
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter(sampled_voxels_uniform[:, 0], sampled_voxels_uniform[:, 1], sampled_voxels_uniform[:, 2], c='b', s=5, label="uni Voxels")
# ax2.set_title("Random Uniform Sampling")
# ax2.legend()
# 均匀采样点
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(sampled_voxels[:, 0], sampled_voxels[:, 1], sampled_voxels[:, 2], c='b', s=5, label="posson Voxels")
ax2.set_title("Random Uniform Sampling")
ax2.legend()

plt.show() 

# def nonlinear_transform(points):
#     transformed_points = np.empty_like(points)
#     transformed_points[:, 0] = points[:, 0] + 0.1 * points[:, 1]**2
#     transformed_points[:, 1] = points[:, 1]
#     transformed_points[:, 2] = points[:, 2] + 0.05 * points[:, 0]**2
#     return transformed_points

# nonlinear_points = nonlinear_transform(inside_voxels)

# plot_point_cloud(nonlinear_points, "Nonlinear Transformed Point Cloud")

# def polynomial_transform(points):
#     transformed_points = np.empty_like(points)
#     transformed_points[:, 0] = points[:, 0] + 0.05 * points[:, 0]**3
#     transformed_points[:, 1] = points[:, 1] - 0.02 * points[:, 1]**3
#     transformed_points[:, 2] = points[:, 2] + 0.03 * points[:, 2]**3
#     return transformed_points

# polynomial_points = polynomial_transform(inside_voxels)

# plot_point_cloud(polynomial_points, "Polynomial Transformed Point Cloud")

# from scipy.stats import gaussian_kde

# def density_transform(points, density_func):
#     # 使用密度函数进行变换
#     density = density_func(points.T)
#     scaled_points = points * density[:, np.newaxis]
#     return scaled_points

# # # 使用KDE
# # kde = gaussian_kde(inside_voxels.T)
# # density_transformed_points = density_transform(inside_voxels, kde)

# # plot_point_cloud(density_transformed_points, "Density Transformed Point Cloud")


# from scipy.optimize import minimize

# # 定义目标标量场 F(x)
# def target_scalar_field(x):
#     # 示例：假设目标场是一个高斯分布
#     center = np.array([5, 5, 5])
#     sigma = 2.0
#     return np.exp(-np.sum((x - center)**2) / (2 * sigma**2))

# # 定义标量场 G(x)，需要与目标场拟合
# def current_scalar_field(x):
#     # 假设 G(x) 由点云位置决定，这里可以使用插值或其他近似方法
#     return np.exp(-np.sum((x - np.array([0, 0, 0]))**2) / 4.0)

# def error_function(points):
#     total_error = 0
#     for point in points:
#         total_error += (current_scalar_field(point) - target_scalar_field(point)) ** 2
#     return total_error

# # 随机初始化点云
# initial_points = np.random.rand(100, 3) * 10

# # 最小化误差函数
# result = minimize(error_function, initial_points.flatten(), method='L-BFGS-B')

# # 获取拟合后的点云
# fitted_points = result.x.reshape(-1, 3)

# plot_point_cloud(fitted_points, "fittest Point Cloud")

# def plot_scalar_field(scalar_field_func, bounds, resolution=50, mode='3d'):
   
#     # 在定义的边界上生成网格
#     x = np.linspace(bounds[0][0], bounds[0][1], resolution)
#     y = np.linspace(bounds[1][0], bounds[1][1], resolution)
#     X, Y = np.meshgrid(x, y)
    
#     if mode == '3d':
#         z = np.linspace(bounds[2][0], bounds[2][1], resolution)
#         X, Y, Z = np.meshgrid(x, y, z)
#         F_values = scalar_field_func(X, Y, Z)
        
#         # 三维等值面显示
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.contour3D(X, Y, Z, F_values, levels=20, cmap='viridis')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
        
#     elif mode == '2d':
#         # 二维热度图显示
#         F_values = scalar_field_func(X, Y)
#         plt.imshow(F_values, extent=(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]), origin='lower', cmap='viridis')
#         plt.colorbar(label='Scalar Field Value')
#         plt.xlabel('X')
#         plt.ylabel('Y')
        
#     plt.title("Scalar Field Visualization")
#     plt.show()

# import pandas as pd
# from scipy import interpolate
# from tg43.I125 import I_125

# def Dose_Rate1D(Position,xrange,yrange,zrange,fuente):
#     #unit mm
#     X, Y, Z = np.meshgrid(xrange, yrange, zrange)
#     Rs = np.sqrt((X-Position[0])**2 + (Y-Position[1])**2 + (Z-Position[2])**2)
#     Rs = Rs / 10 # unit change 2 cm
#     Sk = fuente.Sk
#     it = np.nditer(Rs)
#     Geo_fun = []
#     for r in it:
#         if r >= 1:
#             geo_fun = 1/r**2
#         else:
#             geo_fun = (r**2-0.25*fuente.length**2)/(1-0.25*fuente.length**2)
#         Geo_fun.append(geo_fun)
#     g_f = np.array(Geo_fun).reshape((np.shape(xrange)[0],np.shape(yrange)[0],np.shape(zrange)[0]))
#     g_r = np.interp(Rs,fuente.RadialDoseFuntion['r(cm)'],fuente.RadialDoseFuntion['g(r)'])
#     phi_r = np.interp(Rs,fuente.Phyani['r(cm)'],fuente.Phyani['phi(r)'])
#     dd = Sk*fuente.DoseRateConstant*g_f*g_r*phi_r
#     point_cloud = np.stack((X, Y, Z, dd), axis=3)
#     #（XYZ,DOSE）
#     return point_cloud ,dd

# def Dose_Distribution1D(Seedpos, xrange,yrange,zrange, fuente):
#     """
#     This funtion return a matrix of dose in the space 
#     time:h
#     """
#     Doserate_pc = np.zeros((np.shape(xrange)[0],np.shape(yrange)[0],np.shape(zrange)[0],4))
#     DoseRate = np.zeros((np.shape(xrange)[0],np.shape(yrange)[0],np.shape(zrange)[0]))

#     for Position in Seedpos:
#         dpcdose , ddose =Dose_Rate1D(Position,xrange,yrange,zrange,fuente)
#         Doserate_pc += dpcdose
#         DoseRate += ddose
    
#     return Doserate_pc,DoseRate