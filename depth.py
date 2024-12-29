
import numpy as np

# 文件路径
needle_file = 'D:/Project/Needle_Path/scene2024experiment/Scene20241204_2/hough/Needles_mdf.dat'
points_file = 'D:/Project/Needle_Path/scene2024experiment/Scene20241204_2/hough/Points_mdf.dat'
index_file = 'D:/Project/Needle_Path/scene2024experiment/Scene20241204_2/hough/Index.dat'

# 读取针道位置
try:
    needle_positions = np.loadtxt(needle_file, delimiter=',')
except Exception as e:
    raise FileNotFoundError(f"Error reading needle file: {e}")

# 读取粒子位置
try:
    points = np.loadtxt(points_file, delimiter=' ')
except Exception as e:
    raise FileNotFoundError(f"Error reading points file: {e}")

pttf =  points

for i in range(points.shape[0]):
    pttf[i,:] = np.array([-points[i][0],-points[i][1],points[i][2]])

# 读取粒子和针道对应关系
try:
    with open(index_file, 'r') as f:
        indexlist = f.readlines()
except Exception as e:
    raise FileNotFoundError(f"Error reading index file: {e}")

def parse_index(index_lines, max_index):
    parsed_indices = []
    for line in index_lines:
        indices = list(filter(None, map(lambda x: x.strip(), line.split(','))))  # 去除空字符串
        # 转换为整数并调整为 0-based 索引
        int_indices = [int(i) - 1 for i in indices if i.strip().isdigit()]
        # 检查是否越界
        for idx in int_indices:
            if idx < 0 or idx >= max_index:
                raise ValueError(f"Index {idx + 1} is out of bounds for points with size {max_index}")
        parsed_indices.append(int_indices)
    return parsed_indices

# 计算深度（粒子到针尖的距离，遍历每个针道行）
def calculate_depth(points, indices, needle_positions):
    depths = []
    for particle_indices in indices:
        group_depths = []
        for idx in particle_indices:
            point = points[idx]  # 获取粒子坐标
            
            for needle in needle_positions:  # 遍历针道每一行
                tip = needle[3:]  # 针尖位置为该行的后3列
                sdepth = np.linalg.norm(point - tip)  # 计算欧几里得距离
                
            group_depths.append(sdepth)
        depths.append(group_depths)
    return depths

def save_depth_results(depth_results, output_file):
    try:
        with open(output_file, 'w') as f:
            for i, group in enumerate(depth_results):
                line = f"Group {i + 1}," + ",".join(map(str, group)) + "\n"
                f.write(line)
        print(f"Depth results saved to {output_file}")
    except Exception as e:
        raise IOError(f"Error saving depth results to file: {e}")

# 解析索引关系
indices = parse_index(indexlist,max_index=points.shape[0])

# 计算粒子深度
depth_results = calculate_depth(pttf, indices, needle_positions)

# 输出结果
for i, group in enumerate(depth_results):
    print(f"Group {i + 1}: Depths = {group}")

save_depth_results(depth_results, 'data/hough/needlevert/depth2.dat')
