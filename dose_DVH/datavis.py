import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data = pd.read_csv('alldvh.csv')
print(data.columns)
# 创建图形窗口
fig, ax = plt.subplots(figsize=(16, 12))
#ax.set_title('DVH Comparision for 2 Seed Distributuin Styles')

# 定义颜色和线型列表
colors = ['red', 'blue', 'green', 'orange', 'purple']
line_styles = ['-', '--']

# 绘制5组曲线
for i in range(5):
    # 获取曲线的数据列名
    dose_et_column = f'Dose_et_{i+1}'
    f_et_column = f'Value_et_{i+1}'
    dose_op_column = f'Dose_op_{i+1}'
    f_op_column = f'Value_op_{i+1}'
    
    # 获取曲线的数据
    dose_et = data[dose_et_column]
    f_et = data[f_et_column]
    dose_op = data[dose_op_column]
    f_op = data[f_op_column]
    
    # 绘制第一条曲线（实线）
    ax.plot(dose_et, f_et, label=f'N{i+1}', color=colors[i], linestyle=line_styles[0])
    
    # 绘制第二条曲线（虚线）
    ax.plot(dose_op, f_op, label=f'S{i+1}', color=colors[i], linestyle=line_styles[1])

# 设置图例
ax.legend()
ax.grid(True)
# 设置X轴和Y轴标签
ax.set_xlabel('Dose (Gy)')
ax.set_ylabel('Fractional Volume (%)')

# 显示图像
plt.show()
