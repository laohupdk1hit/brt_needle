# from matplotlib import cm
# import matplotlib.pyplot as plt

# magma = cm.get_cmap('magma', 8)
# print(magma(0.56))

import numpy as np
import vtk
import matplotlib.pyplot as plt

# 自定义点云数据 [x, y, z, s]
points = np.array([[1, 2, 3, 0.5],
                   [4, 5, 6, 0.8],
                   [7, 8, 9, 0.3]])

# 创建vtkPoints对象，并添加点云数据
vtk_points = vtk.vtkPoints()
for point in points:
    vtk_points.InsertNextPoint(point[0], point[1], point[2])

# 创建vtkPolyData对象，并将vtkPoints对象添加到其中
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)

# 创建vtkFloatArray对象，并将标量值s添加到其中
s_array = vtk.vtkFloatArray()
s_array.SetNumberOfComponents(1)
for point in points:
    s_array.InsertNextValue(point[3])

# 将vtkFloatArray对象作为标量数据添加到vtkPolyData对象中
polydata.GetPointData().SetScalars(s_array)

# 创建vtkSphereSource对象，用于可视化点云数据
sphere_source = vtk.vtkSphereSource()
sphere_source.SetPhiResolution(10)
sphere_source.SetThetaResolution(10)

# 创建vtkGlyph3D对象，并将vtkPolyData对象和vtkSphereSource对象添加到其中
glyph = vtk.vtkGlyph3D()
glyph.SetInputData(polydata)
glyph.SetSourceConnection(sphere_source.GetOutputPort())
glyph.SetScaleModeToDataScalingOff()
glyph.Update()

# 创建vtkPolyDataMapper对象，并将vtkGlyph3D对象添加到其中
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyph.GetOutputPort())
mapper.SetScalarRange(0, 1)  # 设置标量值的范围

# 创建vtkColorTransferFunction对象，并定义颜色映射
color_func = vtk.vtkColorTransferFunction()

# 使用matplotlib中的cmap定义颜色映射
cmap = plt.get_cmap('magma')
for i in range(cmap.N):
    r, g, b, _ = cmap(i)
    color_func.AddRGBPoint(i / float(cmap.N - 1), r, g, b)

# 将vtkColorTransferFunction对象设置为vtkPolyDataMapper的颜色映射
mapper.SetLookupTable(color_func)
mapper.ScalarVisibilityOn()

# 创建vtkActor对象，并将vtkPolyDataMapper对象添加到其中
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 创建vtkRenderer对象，并将vtkActor对象添加到其中
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

# 创建vtkRenderWindow对象，并将vtkRenderer对象添加到其中
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# 创建vtkRenderWindowInteractor对象，并将vtkRenderWindow对象添加到其中
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 渲染点云数据
render_window.Render()

# 启动交互式窗口
interactor.Start()
