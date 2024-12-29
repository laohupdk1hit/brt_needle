# import numpy as np
# import vtk
# import slicer

# # 从文件加载点云数据
# points = np.load('E:\code\mycode\dose_DVH\point_data\S2mdfdos.npy')

# # 创建vtkPoints对象
# vtk_points = vtk.vtkPoints()

# # 创建标量数组
# dos_values = vtk.vtkFloatArray()
# dos_values.SetNumberOfComponents(1)

# # 将点云数据添加到vtkPoints对象中，并将标量值添加到数组中
# for i in range(8):
#     for j in range(8):
#         for k in range(8):
#             point = points[i, j, k]
#             vtk_points.InsertNextPoint(point[0], point[1], point[2])
#             dos = point[3]
#             dos_values.InsertNextValue(dos)

# # 创建vtkPolyData对象
# poly_data = vtk.vtkPolyData()
# poly_data.SetPoints(vtk_points)
# poly_data.GetPointData().SetScalars(dos_values)

# # 创建vtkSphereSource对象，用于可视化点云数据
# sphere_source = vtk.vtkSphereSource()
# sphere_source.SetPhiResolution(10)
# sphere_source.SetThetaResolution(10)
# sphere_source.SetRadius(0.2)

# # 创建vtkGlyph3D对象，并将vtkPolyData对象和vtkSphereSource对象添加到其中
# glyph = vtk.vtkGlyph3D()
# glyph.SetInputData(poly_data)
# glyph.SetSourceConnection(sphere_source.GetOutputPort())
# glyph.Update()

# # 创建vtkPolyDataMapper对象
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputConnection(glyph.GetOutputPort())

# # 创建vtkActor对象
# actor = vtk.vtkActor()
# actor.SetMapper(mapper)

# # 创建vtkMRMLModelNode
# model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
# model_node.SetAndObservePolyData(glyph.GetOutput())
# model_node.SetName("PointCloud")

# # 创建显示节点
# display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
# slicer.mrmlScene.AddNode(display_node)

# # 将显示节点添加到模型节点中
# model_node.SetAndObserveDisplayNodeID(display_node.GetID())

# # 设置显示节点属性
# display_node.SetVisibility3D(True)
# display_node.SetOpacity(1.0)  # 设置点云不透明度
# display_node.SetScalarVisibility(True)  # 设置显示标量值

# # 创建一个3D视图窗口
# view_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLViewNode')
# if not view_node:
#     view_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLViewNode')
# slicer.app.layoutManager().threeDWidget(0).mrmlViewNode().SetViewNode(view_node)

# # 在3D视图窗口中显示模型节点
# slicer.mrmlScene.AddNode(model_node)

import numpy as np
import vtk
import slicer

# 从文件加载点云数据
points = np.load('D:/code/needepath/slicerscripts/dose_DVH/beihangplan.npy')
#slicer.util.loadMarkupsFiducialList('E:/Program Files/Slicer 5.2.2/scene2024experiment/Scatter3.fcsv'point_data\)

# 创建vtkImageData对象
image_data = vtk.vtkImageData()
image_data.SetDimensions(points.shape[0], points.shape[1], points.shape[2])
#image_data.SetSpacing(2.5,2.5,2.5)
image_data.AllocateScalars(vtk.VTK_DOUBLE, 1)

# # 将点云数据添加到vtkImageData对象中
# for i in range(points.shape[2]):
#     for j in range(points.shape[1]):
#         for k in range(points.shape[0]):
#             image_data.SetScalarComponentFromDouble(i, j, k, 0, points[j,i,k]) #坐标顺序问题

# 将点云数据添加到vtkImageData对象中
for k in range(points.shape[2]):
    for j in range(points.shape[1]):
        for i in range(points.shape[0]):
            image_data.SetScalarComponentFromDouble(i, j, k, 0, points[i,j,k]) #坐标顺序问题

# 创建一个空的vtkMRMLVolumeNode
volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
volume_node.SetName("DoseVolumeBH")
volume_node.SetSpacing(image_data.GetSpacing())
volume_node.SetOrigin(image_data.GetOrigin())

# 添加vtkImageData到vtkMRMLScalarVolumeNode
volume_node.SetAndObserveImageData(image_data)

display_node = volume_node.GetDisplayNode()
if display_node is None:
    display_node = slicer.vtkMRMLScalarVolumeDisplayNode()
    slicer.mrmlScene.AddNode(display_node)
    volume_node.SetAndObserveDisplayNodeID(display_node.GetID())

ct_volume_node = slicer.util.getNode('ct')  # 将'CTVolume'替换为你CT volume的实际名称

# 获取 CT volume 的中心、IJK 到 RAS 的转换矩阵和体素大小
ct_center = np.array(ct_volume_node.GetOrigin())
ct_spacing = np.array(ct_volume_node.GetSpacing())
ct_ijk_to_ras_matrix = vtk.vtkMatrix4x4()
ct_volume_node.GetIJKToRASDirectionMatrix(ct_ijk_to_ras_matrix)
# 获取体积维度 (单位为体素数)
dimensions = ct_volume_node.GetImageData().GetDimensions()

# 获取体素大小
spacing = np.array(ct_volume_node.GetSpacing())

# 获取体积起点（左下角的位置）
origin = np.array(ct_volume_node.GetOrigin())

# 计算边界值
# min_bounds = origin
# min_tf = origin * np.array([-1,1,-1])
# max_tf = min_tf + spacing * (np.array(dimensions) - 1)
# max_bounds = max_tf * np.array([-1,1,-1])

# ###得到x,y再取负
# min_bounds = min_bounds * np.array([-1,-1,1])
# max_bounds = max_bounds * np.array([-1,-1,1])

#设置dosevolume
volume_node.SetOrigin(ct_center)

# 设置体素大小 (Spacing)
volume_node.SetSpacing(8 * ct_spacing )

# 设置 IJK 到 RAS 的转换矩阵
volume_node.SetIJKToRASDirectionMatrix(ct_ijk_to_ras_matrix)

# 设置显示节点属性
min_val = np.min(points)
max_val = np.max(points)
display_node.SetAutoWindowLevel(0)
display_node.SetWindowLevel(max_val - min_val, min_val)

###########设置volume属性

# 在Slicer中显示体积渲染结果
slicer.util.setSliceViewerLayers(background=volume_node)

# 在3D视图中显示体积渲染结果
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

def drawspheres(r,centers):
    for ct in centers:
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(ct[0], ct[1], ct[2])
        sphereSource.SetRadius(r)
        sphereSource.Update()

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Set actor's color to pink
        actor.GetProperty().SetColor(0.5, 0.0, 0.5)  # RGB color values for pink
        
        # Set actor's opacity to 0.5
        actor.GetProperty().SetOpacity(0.5)

        # Add actor to the scene
        renderer = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderers().GetFirstRenderer()
        renderer.AddActor(actor)
        renderer.ResetCamera()

cts = np.loadtxt('E:/slicerscripts/dose_DVH/point_data/S3nonmdf.dat',delimiter=' ')
#E:\slicerscripts\dose_DVH\point_data\xcord.dat
#'E:/Program Files/Slicer 5.2.2/scene2024experiment/scene_big3/S3nonmdf.dat'
rs = 3