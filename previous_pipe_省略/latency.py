# 把flops/memory等信息存入excel
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimize_base_update import Optimization
# block1=[2.047,1.520,1.474,1.456,2.044,1.520,1.433,1.451,1.469,1.453,1.489,1.469,1.466,1.488,1.696,1.427]
# attn1=[1.159,0.86867,0.836482,0.817345,1.109,0.833090,0.811266,0.810401,0.826722,0.815938,0.845185,0.829537,0.826594,0.841666,0.838785,0.808129]

# block3=[2.158,1.970,1.805,1.576,1.548,1.743,1.572,1.728,1.588,1.771,2.017,1.769,1.641,1.809,1.638,1.792]
# attn3=[1.345,1.19,1.141,0.934978,0.905761,1.098,0.916353,1.097,0.931361,1.122,1.278,1.070,0.984674,1.139,0.960834,1.136]

# block5=[1.741,2.006,1.802,1.643,1.623,1.608,1.594,1.634,1.598,1.663,1.604,1.602,1.610,1.866,1.897,1.724]
# attn5=[1.031,0.96675,1.086,0.962498,0.942753,0.942401,0.938146,0.961953,0.933153,0.954721,0.942466,0.938145,0.959874,1.127,1.133,1.016]
# block7=[1.884,1.690,1.640,1.822,1.875,1.975,1.956,1.656,1.657,1.633,1.646,1.665,1.718,1.650,1.690,1.710]
# attn7=[1.032,0.981378,0.943586,1.078,1.104,1.165,1.167,0.975586,0.967521,0.948161,0.962945,0.965602,1.010,0.963265,0.970402,1.001]
# block10=[3.612,3.432,3.435,3.439,3.436,3.433,3.441,3.441,3.444,3.443,3.449,3.446,3.435,3.431,3.412,3.407]
# attn10=[1.943,1.916,1.921,1.923,1.923,1.918,1.924,1.921,1.924,1.923,1.928,1.925,1.921,1.921,1.912,1.908]
# cumulative_list=[x/y for x,y in zip(attn1,block1)]
# cumulative_list3=[x/y for x,y in zip(attn3,block3)]
# cumulative_list5=[x/y for x,y in zip(attn5,block5)]
# cumulative_list7=[x/y for x,y in zip(attn7,block7)]
# cumulative_list10=[x/y for x,y in zip(attn10,block10)]
# print(cumulative_list3)
# index=list(range(len(block1)))
# plt.plot(index, cumulative_list3,marker='o', markersize=0.5,linestyle='-', color='b')  
# plt.xlabel('Block')
# plt.ylabel('attn(ms)/block(ms)')
# plt.title('attn(ms)/block(ms) stage3 Plot')
# plt.grid(True)
# plt.savefig("./ratio_latency_s3.png")
B=1
L=[1, 4, 9, 16, 25, 36, 64, 100, 169, 256]

print(L)
D=64
E=32
K=3
W=H=16
C=640
ada_flops=0
ada_mem=0
quant_flops=0
quant_mem=0
L_base=0
x_flops=[]
y_flops=[]
x_mem=[]
temp_m=[]
temp_sram=[]
y_mem=[]
num=[]
num_m=[]
y_attn=[]
y_block=[]
count=0
flops_dict={}
parameter_dict={}
KVcache_dict={}
fhat_dict={}
FIFO_dict={}
a_dict={}
b_dict={}
for i in range(10):
    num_m.append(len(y_flops))
    temp=[12*B*L[i]*D**2,4*B*L[i]*(L[i]+L_base)*D,4*B*L[i]*(L[i]+L_base)*D,4*B*L[i]*D**2,16*B*L[i]*D**2,16*B*L[i]*D**2]
    temp = [f"{num:.4e}" for num in temp]
    temp_w=[3*D*(D+1),0,0,D*(D+1),4*D*(D+1),D*(4*D+1)]
    # INT8 乘8，后面B字节除8抵消了
    temp_i=[2*B*L[i]*D,2*B*L[i]*D,2*B*16*L[i]*(L[i]+L_base),2*B*L[i]*D,2*B*L[i]*D,2*B*L[i]*4*D]
    
    temp_kv=[0,2*B*(L[i]+L_base)*D,2*B*(L[i]+L_base)*D,0,0,0]
    temp_sram=[x + y  for x, y  in zip(temp_i,temp_kv)]
    temp_m=[x + y + z for x, y ,z in zip(temp_w, temp_i,temp_kv)]
    print(f'stage{i}:')
    print('DRAM(MB):')
    temp_w=[k /1024/1024 for k in temp_w]
    print(temp_w)
    print('SRAM(KB):')
    temp_sram=[k /1024 for k in temp_sram]
    print(temp_sram)
    print('TOTAL(MB):')
    temp_m=[k /1024/1024 for k in temp_m]
    print(temp_m)
    temp_kv = [f"{num:.4e}" for num in temp_kv]
    temp_i = [f"{num:.4e}" for num in temp_i]
    temp_w = [f"{num:.4e}" for num in temp_w]
    # for j in range(30):
    #     y_flops.extend(temp)
    #     y_mem.extend(temp_m)
    b_dict=[
    ("FLOPs", [temp]),
    ("Parameter", [temp_w]),
    ("KV", [temp_kv]),
    ("FIFO", [temp_i])
]

    # y_attn.append(4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*D**2)
    # y_block.append(24*B*L[i]*D**2+12*B*L[i]*D**2+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*D**2+16*B*L[i]*D**2+16*B*L[i]*D**2)
    

    # temp3=[8*B*L[i]*D**2,16*B*L[i]*D**2]
    # temp_w3=[2*D*(D+1),4*D*(D+1)]
    # temp_i3=[2*B*L[i]*D,2*B*L[i]*D]
    # temp_m3=[x + y for x, y in zip(temp_w3, temp_i3)]
    
    # y_flops.extend(temp3)
    # y_mem.extend(temp_m3)
    # print(len(y_mem))
    # if i<9:
    #     y_flops.append(2*B*L[i+1]*E*D) #word embedding部分
    #     #print(y_flops)
    #     y_mem.append(B*L[i+1]*E+E*D)
    a_dict[i]=[i,b_dict]
    L_base+=L[i]

excel_data = []

for stage, (index, b_data) in a_dict.items():
    # Initialize lists for all categories
    flops_values = [None] * 6
    param_values = [None] * 6
    kv_values = [None] * 6
    fifo_values = [None] * 6

# for stage, (index, b_data) in a_dict.items():
#     for entry in b_data:
#         category, values = entry
#         for j, value in enumerate(values[0]):  # 只处理内部列表的第一个子列表
#             data_entry = {
#                 'Stage': stage,
#                 'Layer Index': j,
#                 'FLOPs': values[0][j] if category == 'FLOPs' else None,
#                 'Parameter': values[0][j] if category == 'Parameter' else None,
#                 'KV': values[0][j] if category == 'KV' else None,
#                 'FIFO': values[0][j] if category == 'FIFO' else None
#             }
#             excel_data.append(data_entry)

    #Fill the lists with the corresponding values for each category
    for entry in b_data:
        category, values = entry
        print(category)
        for j, value in enumerate(values[0]):
            if category == 'FLOPs':
                flops_values[j] = value
            elif category == 'Parameter':
                param_values[j] = value
            elif category == 'KV':
                kv_values[j] = value
            elif category == 'FIFO':
                fifo_values[j] = value
    
    # Append each index and its values for each category
    for j in range(6):
        data_entry = {
            'Stage': stage,
            'Layer Index': j,
            'FLOPs': flops_values[j],
            'Parameter': param_values[j],
            'KV': kv_values[j],
            'FIFO': fifo_values[j]
        }
        excel_data.append(data_entry)

df = pd.DataFrame(excel_data)
excel_filename = 'hh_dict_data.xlsx'
df.to_excel(excel_filename, index=False)

print(f"数据已保存到 {excel_filename}")
#print(a_dict)

print(f"开始优化")
Optimization(excel_filename, data_entry{'FLOPs'},data_entry{'Parameter'},data_entry{'KV'},data_entry{'FIFO'}).optimize()
