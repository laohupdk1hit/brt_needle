import vtk

# 创建球体源
sphere = vtk.vtkSphereSource()
sphere.SetRadius(0.6)  # 设置球体半径为0.1
sphere.SetThetaResolution(50)  # 设置水平分辨率
sphere.SetPhiResolution(50)  # 设置垂直分辨率

# 更新源以生成几何体
sphere.Update()

# 创建STL写入器
stl_writer = vtk.vtkSTLWriter()
stl_writer.SetFileName("STLsphere.stl")  # 保存为sphere.stl
stl_writer.SetInputConnection(sphere.GetOutputPort())

# 写入STL文件
stl_writer.Write()

print("STLsphere.stl")