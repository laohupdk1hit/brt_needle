import numpy as np
import os

def dat_to_fcsv(dat_file, fcsv_file):
    seeddata  = np.loadtxt(dat_file,delimiter=',')
    fcsv_header ="""# Markups fiducial file version = 4.11
# CoordinateSystem = 0
# columns = label,x,y,z,sel,vis,lock
"""

    with open(fcsv_file, 'w') as f:
        f.write(fcsv_header)  # 写入头部信息
        for i, (x, y, z) in enumerate(seeddata):  # 遍历每一行坐标数据
            label = f"F_{i + 1}"  # 自动生成标记名称
            f.write(f"{label},{-x:.6f},{-y:.6f},{z:.6f},1,1,0\n")  # 写入格式化数据


# 使用示例
dat_file_path = "fit_shape_seed4.dat"  # 输入 .dat 文件路径
fcsv_file_path = "coordinates4.fcsv"  # 输出 .fcsv 文件路径

# 检查文件是否存在
if os.path.exists(dat_file_path):
    dat_to_fcsv(dat_file_path, fcsv_file_path)
else:
    print(f"Error: {dat_file_path} does not exist.")

# import slicer
#slicer.util.loadMarkups('D:\code\TPS\coordinates1.fcsv')