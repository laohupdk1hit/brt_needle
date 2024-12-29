import json
import numpy as np

# 读取JSON文件
with open('hn_plan.json', 'r') as file:
    data = json.load(file)

seeds = []

for needle_path in data["needle_paths"]:
    seeds_n = needle_path["seed position"]
    for seedpos in seeds_n:
      seeds.append(seedpos)

seeds = np.array(seeds)

np.savetxt('beihangseeds.dat',seeds)

#print(seeds)

# x_max = max(seeds[:,0]) 
# x_min = min(seeds[:,0]) 
# y_max = max(seeds[:,1]) 
# y_min = min(seeds[:,1]) 
# z_max = max(seeds[:,2]) 
# z_min = min(seeds[:,2])

# print(x_max,x_min,y_max,y_min,z_max,z_min)