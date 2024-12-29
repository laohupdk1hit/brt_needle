import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def adjust_length(segments, l):
    new_segments = []
    for segment in segments:
        b = segment[3:]
        a = segment[:3]
        bx, by, bz  = b
        ax, ay, az = a
        # 计算方向向量 (vx, vy, vz)
        vx = bx - ax
        vy = by - ay
        vz = bz - az
        
        # 计算原始线段的长度
        original_length = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        
        direction_x = vx / original_length
        direction_y = vy / original_length
        direction_z = vz / original_length

        moved_ax = bx - direction_x * l
        moved_ay = by - direction_y * l
        moved_az = bz - direction_z * l

        ns = [moved_ax,moved_ay,moved_az,bx,by,bz]
        
        new_segments.append(ns)
    return new_segments

def adjust_line_segments(segments, target_length,move_distance):
    new_segments = []
    for segment in segments:
        b = segment[3:]
        a = segment[:3]
        bx, by, bz  = b
        ax, ay, az = a
        # 计算方向向量 (vx, vy, vz)
        vx = bx - ax
        vy = by - ay
        vz = bz - az
        
        # 计算原始线段的长度
        original_length = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        
        
        # 归一化方向向量，使其长度为1
        if original_length == 0:
            raise ValueError("a 和 b 重合，无法确定方向向量")
        
        direction_x = vx / original_length
        direction_y = vy / original_length
        direction_z = vz / original_length

        # 计算新的终点坐标
        new_x = bx + direction_x * target_length
        new_y = by + direction_y * target_length
        new_z = bz + direction_z * target_length

        moved_bx = bx - direction_x * move_distance
        moved_by = by - direction_y * move_distance
        moved_bz = bz - direction_z * move_distance

        moved_ax = new_x - direction_x * move_distance
        moved_ay = new_y - direction_y * move_distance
        moved_az = new_z - direction_z * move_distance

        ns = [moved_ax,moved_ay,moved_az,moved_bx,moved_by,moved_bz]
        
        new_segments.append(ns)
    return new_segments

segments = np.loadtxt('D:/Project/Needle_Path/scene2024experiment/Scene20241204_4/hough/Needle.dat',delimiter=',')

target_length = 180
moved_distance = 35
adl = 180

# vert_segments = adjust_line_segments(segments, target_length,moved_distance)
# new_segments = np.array(vert_segments)
new_segments = adjust_length(segments,adl)

np.savetxt('D:/Project/Needle_Path/scene2024experiment/Scene20241204_4/hough/Needle_v1.dat',new_segments,delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for idx, ns in enumerate(segments):
    bx, by, bz = ns[3:]
    ax_, ay_, az_ = ns[:3]  # 避免与 ax 重名

    # 绘制线段 (a -> b)
    ax.plot([ax_, bx], [ay_, by], [az_, bz], color='blue', label='Original Segment')
    # 在线段中间位置标注索引
    mid_x = (ax_ + bx) / 2
    mid_y = (ay_ + by) / 2
    mid_z = (az_ + bz) / 2
    ax.text(mid_x, mid_y, mid_z, f"{idx}", color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Direction Vectors Visualization")
plt.show()