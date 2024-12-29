import vtk
import numpy as np

pts =  np.loadtxt('data/input/2024-10-15-9.dat',delimiter=',')
points = vtk.vtkPoints()

for pt in pts:
    points.InsertNextPoint(pt[0],pt[1],pt[2])

# 创建点云数据对象
point_cloud = vtk.vtkPolyData()
point_cloud.SetPoints(points)

# 创建球体的源
sphere = vtk.vtkSphereSource()
sphere.SetRadius(4.5)  # 设置球体半径

# # 创建逻辑算符
# boolean_operation = vtk.vtkBooleanOperationPolyDataFilter()
# boolean_operation.SetOperationToUnion()

# # 创建一个空的Polydata用于存储并集结果
# Polydata = vtk.vtkPolyData()
# pt0 = point_cloud.GetPoint(0)
# sphere.SetCenter(pt0)
# sphere.Update()
# Polydata.DeepCopy(sphere.GetOutput())  # 将第一个球体作为初始输入

# 创建vtkAppendPolyData类
append_filter = vtk.vtkAppendPolyData()

#遍历点云数据，以每个点为球心绘制球体并添加到vtkAppendPolyData中
for i in range(point_cloud.GetNumberOfPoints()):
    point = point_cloud.GetPoint(i)
    sphere.SetCenter(point)
    sphere.Update()

    # 获取球体的多边形数据
    poly_data = vtk.vtkPolyData()
    poly_data.ShallowCopy(sphere.GetOutput())

    # 将球体添加到vtkAppendPolyData中
    append_filter.AddInputData(poly_data)
    append_filter.Update()

# 遍历点云数据，以每个点为球心绘制球体并进行并集操作
# for i in range(1, point_cloud.GetNumberOfPoints()):
#     point = point_cloud.GetPoint(i)
#     sphere.SetCenter(point)
#     sphere.Update()

#     # 获取球体的多边形数据
#     poly_data = vtk.vtkPolyData()
#     poly_data.ShallowCopy(sphere.GetOutput())

#     boolean_operation.SetOperationToUnion()

#     boolean_operation.SetInputData(0, Polydata)  # 设置索引0的输入为上一次的结果
#     boolean_operation.SetInputData(1, poly_data)  # 设置索引1的输入为当前球体
#     boolean_operation.Update() #不可行，在换针道后，新的点和原来的彼此分离，无法进行求并运算，之后的一直报错。

#     Polydata.ShallowCopy(boolean_operation.GetOutput())


# 获取合并后的多体数据集
Polydata = vtk.vtkPolyData()
Polydata.ShallowCopy(append_filter.GetOutput())

# 创建vtkDelaunay3D滤波器并设置输入数据
delaunay = vtk.vtkDelaunay3D()
delaunay.SetInputData(Polydata)
delaunay.Update()

# 获取输出数据
output = delaunay.GetOutput()

# 创建vtkGeometryFilter并设置输入数据
geometry_filter = vtk.vtkGeometryFilter()
geometry_filter.SetInputData(output)
geometry_filter.Update()

# 获取转换后的输出数据
output_polydata = geometry_filter.GetOutput()

# 创建vtkPolyDataNormals对象
normals = vtk.vtkPolyDataNormals()
normals.SetInputData(output_polydata)
normals.ComputePointNormalsOn()
normals.ComputeCellNormalsOff()
normals.Update()

# 获取法线化后的输出数据
output_normals = normals.GetOutput()

# 创建vtkSTLWriter并设置输入数据
stl_writer = vtk.vtkSTLWriter()
stl_writer.SetFileName("2024-10-15-9.stl")
stl_writer.SetInputData(output_normals)
stl_writer.Write()

# 加载STL模型
reader = vtk.vtkSTLReader()
reader.SetFileName("2024-10-15-9.stl")
reader.Update()

# 计算体积
mass = vtk.vtkMassProperties()
mass.SetInputData(reader.GetOutput())
mass.Update()

volume = mass.GetVolume()
print("体积md：", volume)

# 创建球体的映射器
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(Polydata)

# 创建球体的演员
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.0,1.0, 0.0)  # 设置球体的颜色为红色
actor.GetProperty().SetOpacity(0.2)   # 设置球体的颜色为红色

# 创建vtkAxesActor来显示坐标系
axes_actor = vtk.vtkAxesActor()

# 创建渲染器和窗口
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.AddActor(axes_actor)  # 添加坐标系

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# 创建交互器
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)


# 更新渲染窗口
render_window.Render()
# 将结果输出为模型文件
# writer = vtk.vtkPLYWriter()
# writer.SetFileName("output.ply")
# writer.SetInputData(poly_data)
# writer.Write()

# 启动交互式窗口
interactor.Start()

#slicer.util.loadMarkupsFiducialList("E:/Program Files/Slicer 5.2.2/slicerscripts/data/scatter/Tablep3.fcsv")

#np.loadtxt('C:/Users/laohupdk/Desktop/tablep3.txt')