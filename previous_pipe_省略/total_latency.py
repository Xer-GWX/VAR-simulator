import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math

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
    