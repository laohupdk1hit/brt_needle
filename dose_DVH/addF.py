import numpy as np
#vtkMRMLMarkupsFiducialNode_0,51.4869,-9.87066,62,0,0,0,1,1,1,0,Target,,vtkMRMLScalarVolumeNode1
#columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID
#label,l,p,s,defined,selected,visible,locked,description
seeddata  = np.loadtxt('data/pointcloud/point3.dat',delimiter=',')
# print(np.size(seeddata,0))
# print(seeddata[38][1])

f = open('data/pointcloud/point3.dat.fcsv', 'w')
for i in range(np.size(seeddata,0)):
    f.writelines(' ,%f,%f,%f,0,1,0,1,1,1,0,F_%d,,\n' %(seeddata[i][0],seeddata[i][1],seeddata[i][2],i+1))
f.close()
#slicer.util.loadMarkupsFiducialList("E:/Program Files/Slicer 5.2.2/scene2024experiment/scene_big2/Scatter2.fcsv")
#E:\Program Files\Slicer 5.2.2\slicerscripts\functest\addf3.fcsv
#data\grid\gridpoint2.dat
#E:\Program Files\Slicer 5.2.2\scene2024experiment\scene_big2