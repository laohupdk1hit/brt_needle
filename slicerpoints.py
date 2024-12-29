import slicer
import vtk
import numpy as np
# 创建一个VTK点云数据对象
point_cloud = vtk.vtkPolyData()

# 创建一个VTK点数据数组来存储点的坐标
points = vtk.vtkPoints()

# 创建一个VTK浮点型数据数组来存储s值
s_values = vtk.vtkFloatArray()
s_values.SetName("s_values")

# 从文件中读取点坐标数据
mypoints = np.loadtxt('E:\Program Files\Slicer 5.2.2\slicerscripts\pts', delimiter=',')
for point in mypoints:
    points.InsertNextPoint(point[0], point[1], point[2])

# 从文件中读取s值数据
myvalues = np.loadtxt('E:\Program Files\Slicer 5.2.2\slicerscripts\mvs', delimiter=',')
for v in myvalues:
    s_values.InsertNextValue(v)

# 将点数据和s值数据设置到点云对象中
point_cloud.SetPoints(points)
point_cloud.GetPointData().SetScalars(s_values)

# Create model node
scene = slicer.mrmlScene
model = slicer.vtkMRMLModelNode()
model.SetScene(scene)
model.SetName(scene.GenerateUniqueName('Point Cloud'))
model.SetAndObservePolyData(point_cloud)

modelDisplay = slicer.vtkMRMLModelDisplayNode()
modelDisplay.SetScene(scene)
modelDisplay.SetVisibility2D(True) # Show in slice view
scene.AddNode(modelDisplay)
model.SetAndObserveDisplayNodeID(modelDisplay.GetID())

# 将模型显示节点与点渲染器关联
modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())

modelDisplay.SetScalarRange(s_values.GetRange())
scene = slicer.mrmlScene
scene.AddNode(model) #显示在slicer

#不行妈的

# # 将点渲染器包装成vtkMRMLModelNode对象
# point_model = slicer.vtkMRMLModelNode()
# point_model.SetAndObservePolyData(point_cloud)
# point_model.SetName("Point Cloud")






