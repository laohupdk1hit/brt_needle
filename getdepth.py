import numpy as np
import vtk

pt2 = np.loadtxt('E:/Program Files/Slicer 5.2.2/slicerscripts/data/grid/gridp2.txt')
np.savetxt('data\grid\gridpoints2.dat',pt2,fmt='%.6f',delimiter=',')

def buildbsp(obstacle):
    #obb=getNode('Model_2_bone.vtk')
    bspTree = vtk.vtkModifiedBSPTree()
    bspTree.SetDataSet(obstacle.GetPolyData())
    bspTree.BuildLocator()
    return bspTree

obb=getNode('Model_2_bone.vtk')
bspTreebone = buildbsp(obb)

depth = []
for point in pt2:
    dp = bspTreebone.FindClosestPoint(point)
    depth.append(dp)