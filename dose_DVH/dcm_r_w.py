import pydicom
import numpy as np
from pydicom.dataset import FileMetaDataset
from pydicom.filewriter import write_file_meta_info

# # 读取RT Structure Set文件
# rt_structure = pydicom.dcmread('PL697807079649742-hero/20231205/1-AutoSS/000000.dcm')

# # 设置RT Dose数据
# rt_dose_data = np.load('point_data/p3dos.npy')  # 替换为您的RT Dose数据
# rt_dose_rows, rt_dose_columns, rt_dose_numberOfFrames = rt_dose_data.shape

# # 创建新的数据集对象
# rt_dose = pydicom.Dataset()

# # 设置文件元信息
# file_meta = FileMetaDataset()
# file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
# file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'  # RT Dose SOP Class UID
# file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.1.2"  # 设置新的SOP实例UID
# file_meta.ImplementationClassUID = '1.2.246.352.70.2.1.7'
# file_meta.FileMetaInformationVersion = b'\x00\x01'
# #file_meta.FileMetaInformationGroupLength = len(file_meta_bytes)

# # 将文件元信息添加到数据集
# rt_dose.file_meta = file_meta

# # 设置RT Dose属性
# rt_dose.PatientName = rt_structure.PatientName
# rt_dose.PatientID = rt_structure.PatientID
# rt_dose.StudyInstanceUID = rt_structure.StudyInstanceUID
# #rt_dose.SeriesInstanceUID = rt_structure.SeriesInstanceUID
# rt_dose.SOPInstanceUID = "1.2.3.4.5.1.2"  # 设置新的SOP实例UID
# rt_dose.Modality = "RTDOSE"
# rt_dose.DoseGridScaling = 1.0
# rt_dose.is_little_endian = True
# rt_dose.BitsAllocated = 32
# rt_dose.Rows = rt_dose_rows
# rt_dose.Columns = rt_dose_columns
# rt_dose.NumberOfFrames = rt_dose_numberOfFrames
# rt_dose.SamplesPerPixel = 1
# rt_dose.PhotometricInterpretation = 'MONOCHROME2'
# rt_dose.PixelRepresentation = 0
# rt_dose.BitsStored = 32
# # 设置像素数据
# rt_dose.PixelData = rt_dose_data.tobytes()

# # 保存DICOM文件
# # 保存DICOM文件
# with open('PL697807079649742-hero/20231205/1-Rtdose/output_rt_dose.dcm', 'wb') as f:
#     write_file_meta_info(f, file_meta)
#     rt_dose.save_as(f)

dcm_file = 'testdata/rtss.dcm'  # 替换为你的DICOM文件路径
ds = pydicom.dcmread(dcm_file)

# dose_data_int = np.array(ds.pixel_array)  # 获取像素数组
# dose_data_float = dose_data_int.astype(np.float32)

print(ds)

# dcm_file = 'PL697807079649742-hero/20231205/1-Rtdose/output_rt_dose.dcm'  # 替换为你的DICOM文件路径
# ds = pydicom.dcmread(dcm_file, force=True)

# # 设置 SOP 类别和 UID
# ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'  # RT Dose SOP 类别

# # 保存修改后的文件
# ds.save_as('PL697807079649742-hero/20231205/1-Rtdose/rt_dose.dcm')
import numpy as np

# 用您的文件路径替换下面的路径
file_path = "E:\code\mycode\dose_DVH\point_data\p3dos.npy"

# 尝试加载.npy文件
try:
    data = np.load(file_path)
    print("文件加载成功！")
    print("数据形状：", data.shape)
except Exception as e:
    print("加载失败:", e)


import numpy as np
import vtk
import slicer

# 加载您的.npy文件
file_path = "E:/code/mycode/dose_DVH/point_data/p3dos.npy"
data = np.load(file_path)

# 创建vtkImageData对象
image_data = vtk.vtkImageData()
image_data.SetDimensions(data.shape[0], data.shape[1], data.shape[2])
image_data.AllocateScalars(vtk.VTK_DOUBLE, 1)

# 将NumPy数组中的数据复制到vtkImageData中
for z in range(data.shape[2]):
    for y in range(data.shape[1]):
        for x in range(data.shape[0]):
            image_data.SetScalarComponentFromDouble(x, y, z, 0, data[x, y, z])

# 创建vtkMRMLScalarVolumeNode
volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
volume_node.SetAndObserveImageData(image_data)
volume_node.SetName("MyVolume")
volume_node.CreateDefaultDisplayNodes()

# 在场景中添加VolumeNode以在3D Slicer中查看
slicer.mrmlScene.AddNode(volume_node)


