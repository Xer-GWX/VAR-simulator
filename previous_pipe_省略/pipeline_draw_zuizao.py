# import matplotlib.pyplot as plt
# import numpy as np

# import matplotlib.pyplot as plt
# import numpy as np

# # 示例数据
# # 每个元组代表 (Tile IDs, 开始周期, 结束周期, 颜色)
# data = [
#     ([1, 2], 0, 10000, 'lightgreen'),  # 任务在Tile 1和2上同时执行
#     ([3, 4], 5000, 20000, 'green'),    # 任务在Tile 3和4上同时执行
#     ([5], 15000, 30000, 'yellow'),     # 任务仅在Tile 5上执行
#     ([6, 7, 8], 25000, 40000, 'orange'), # 任务在Tile 6, 7, 8上同时执行
#     ([9, 10], 35000, 50000, 'red'),    # 任务在Tile 9和10上同时执行
#     ([11, 12], 45000, 60000, 'blue'),  # 任务在Tile 11和12上同时执行
#     ([13], 55000, 70000, 'cyan'),      # 任务仅在Tile 13上执行
#     ([14, 15, 16], 65000, 80000, 'brown'), # 任务在Tile 14, 15, 16上同时执行
#     # 可以为每个任务添加更多数据...
# ]


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
#使用固定的随机种子：通过为随机数生成器设定固定的种子，可以确保每次运行代码时，颜色的生成都是一致的。
random.seed(27)
# 生成随机颜色的函数
def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# 读取Excel文件
file_path = 'new_dict_data.xlsx'
df = pd.read_excel(file_path)

# 初始化数据列表
data = []
task_colors = {}
batchsize=2
num=-1
#for _, row in df.iloc[1:12].iterrows():
for _, row in df.head(12).iterrows(): # 
    num=num+1
    task_id = row['Layer Index']
    total = row['Total']
    stage_id= row['Stage']
    tilenum = row['tile num']
    task_name = row.get('Task Name', f"[{math.floor(num/6)},{int(task_id)},{int(total)}MB/{int(tilenum)}MB]")
    
    # 如果这个Task ID没有颜色，生成一个并存储
    if task_id not in task_colors:
        task_colors[task_id] = generate_random_color()

    # 获取该Task ID的颜色
    color = task_colors[task_id]
    # 自动生成Tile ID列表，从Start Tile到End Tile
    cycle=row['Cycle'] # array np.f64
    start = int(row['start time'])
    end = int(row['end time'])
    tiles = list(range(int(row['start tile']), int(row['end tile']) + 1))
    
    # 如果颜色为空，则生成随机颜色
    #color = row['Color'] if pd.notna(row['Color']) else generate_random_color()

    # 将任务数据添加到列表中
    data.append((tiles, start, end, color,task_name))

# 设置图形和轴
fig, ax = plt.subplots(figsize=(8, 6))#40，35 ;100 50
max_end = df['end time'][1:12].max()
# 循环遍历数据并绘制每个任务
for tiles, start, end, color,task_name in data:
    first_tile = True  # 标记是否为第一个tile
    for tile in tiles:
        height = 1  # 填满整个Tile区域
        bottom = tile - 0.5  # 从Tile的底部开始
        ax.barh(bottom, end-start, left=start, height=height, color=color, edgecolor='none')
    center_y = (tiles[0] + tiles[-1]-1) / 2 
    center_x = start + (end - start) / 2
    if first_tile:
        ax.text(center_x, center_y, task_name,
                ha='center', va='center', color='black', fontsize=10)
        first_tile = False
# 自定义图表
ax.set_xlabel('Time(cycle)',fontsize=12)
ax.set_ylabel('Tile ID',fontsize=12)
ax.set_title('[2,0]_basic',fontsize=12)
ax.set_yticks(np.arange(0, 145, 12))  # 1到16的Tile ID
ax.set_ylim(0, 145)  # 调整Y轴范围以适应完整的Tile区域
ax.set_xlim(0, max_end)  # 时间周期范围
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
# 保存图表
output_filename = 'm[2,0].png'#bs stage
plt.savefig(output_filename, bbox_inches='tight', dpi=600)

# 关闭图表以释放内存
plt.close()

print(f"图表已保存为 {output_filename}")
