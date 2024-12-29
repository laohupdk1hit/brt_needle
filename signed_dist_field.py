""" First, load or create the sources that provide the polygonal mesh and data set on which you want to add the signed distance field. 
Next, select these sources in the Pipeline Browser. select the polygonal mesh first, followed by the data set that should have the 
signed distance field added. 
Now add a Programmable Filter and set the Script property to the following """

mesh = self.GetInputDataObject(0, 0)

import paraview.vtk as vtk

pdd = vtk.vtkImplicitPolyDataDistance()##用这个找模型到直线最短距离
pdd.SetInput(mesh)

dataset = self.GetInputDataObject(0, 1)
output.CopyStructure(dataset)

numPts = dataset.GetNumberOfPoints()
distArray = vtk.vtkDoubleArray()
distArray.SetName("Distance")
distArray.SetNumberOfComponents(1)
distArray.SetNumberOfTuples(numPts)
for i in xrange(numPts): #呵呵也是循环的。。。妈的自己写吧
  pt = dataset.GetPoint(i)
  distArray.SetValue(i, pdd.EvaluateFunction(pt))

output.GetPointData().AddArray(distArray)
