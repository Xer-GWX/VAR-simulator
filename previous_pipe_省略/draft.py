import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
#from models import VQVAE, build_vae_var
#import thop
#model = torchvision.models.segmentation.fcn_resnet50()
#x = torch.randn(1,3,224,224)
#tensor = (torch.rand(1, C, H, W),)
#flops, params = profile(model, inputs=tensor)
#flops, params = thop.profile(model,inputs=(x,))
#print('flops: %.2f G, params: %.2f M' % (flops * 1e-9, params * 1e-6))
# import torch
# import torch.nn as nn
# from thop import profile


# # 创建一个 Linear 层
# linear_layer = nn.Linear(in_features=2, out_features=3, bias=False)

# # 创建一个输入张量，假设我们有一个大小为 3 的批次
# input_tensor = torch.randn(4, 2)  # 随机初始化，模拟输入数据


# # 计算 FLOPs 和参数数量
# flops, params = profile(linear_layer, inputs=(input_tensor,))

# # 打印 FLOPs 和参数数量
# print('FLOPs: ', flops)
# print('Parameters: ', params)
# from torch import nn
# import torch

# model = nn.Linear(2, 3) # 输入特征数为2，输出特征数为1


# input = torch.randn(5,7,9,2)
# output = model(input)
# print(output.shape)
# for param in model.parameters():
#     print(param)


# import torch
# import torch.nn as nn
# import numpy as np
 
# feature_array = np.array([[[[9, 3],  [7, 2]],
#                            [[3, 4],  [1, 2]],
#                            [[-2, 9], [7, 5]],
#                            [[2, 3],  [4, 2]]],
 
#                           [[[1, 2],  [-1, 3]],
#                             [[1, 2], [3, 5]],
#                             [[4, 7], [-6, 4]],
#                             [[1, 4], [1, 5]]]], dtype=np.float32)
 
# feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)
# model=nn.GroupNorm(num_groups=2, num_channels=4)
# gn_out = model(feature_tensor)
# print(gn_out.shape)
# print(feature_tensor.shape) #2,4,2,2
# for param in model.parameters():
#     print(param)

# import torch
# import torch.nn as nn
# import numpy as np
 
# feature_array = np.array([[[[1, 0],  [0, 2]],
#                            [[3, 4],  [1, 2]],
#                            [[2, 3],  [4, 2]]],
 
#                           [[[1, 2],  [-1, 0]],
#                             [[1, 2], [3, 5]],
#                             [[1, 4], [1, 5]]]], dtype=np.float32)
 
 
# feature_array = feature_array.reshape((2, 3, -1)).transpose(0, 2, 1)
# feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)
# model=nn.LayerNorm(normalized_shape=3)
# ln_out = nn.LayerNorm(normalized_shape=3)(feature_tensor)
# print(ln_out)
# for param in model.parameters():
#     print(param)
import numpy
import numpy as np
import matplotlib.pyplot as plt

# 在图中从位置(0,0)到位置(6,250)画一条线  
#xpoints = np.array([0, 6])  
#ypoints = np.array([0, 250])  


# # 不指定x轴的点，默认为0到1平均分
# ypoints = np.array([0, 250])
# plt.plot(ypoints)  
# plt.show()


B=8
L1=[1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
L=[L1[i]*256 for i in range(len(L1))]
print(L)
D=1920
E=16
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
temp_m3=[]
y_mem=[]
num=[]
num_m=[]
y_attn=[]
y_block=[]
count=0
attn_flops =0
for i in range(10):
    num_m.append(len(y_flops))
    temp=[24*B*L[i]*D**2,12*B*L[i]*D**2,4*B*L[i]*(L[i]+L_base)*D,4*B*L[i]*(L[i]+L_base)*D,4*B*L[i]*D**2,16*B*L[i]*D**2,16*B*L[i]*D**2]
    
    temp_w=[6*D*(D+1)/4,3*D*(D+1)/4,2*B*(L[i]+L_base)*D/2,2*B*(L[i]+L_base)*D/2,D*(D+1)/4,4*D*(D+1)/4,D*(4*D+1)]
    temp_i=[2*B*L[i]*D/2,2*B*L[i]*D/2,2*B*L[i]*D/2,2*B*16*L[i]*(L[i]+L_base)/2,2*B*L[i]*D/2,2*B*L[i]*D/2,2*B*L[i]*4*D]
    #print(2*B*16*L[i]*(L[i]+L_base))
    temp_m=[x + y for x, y in zip(temp_w, temp_i)]
    for j in range(30):
        attn_flops += 4*B*L[i]*(L[i]+L_base)*D + 4*B*L[i]*(L[i]+L_base)*D
        y_flops.extend(temp)
        y_mem.extend(temp_m)
    y_attn.append(4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*D**2)
    y_block.append(24*B*L[i]*D**2+12*B*L[i]*D**2+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*D**2+16*B*L[i]*D**2+16*B*L[i]*D**2)
    

    temp3=[8*B*L[i]*D**2,16*B*L[i]*D**2]
    temp_w3=[2*D*(D+1)/4,4*D*(D+1)/4]
    temp_i3=[2*B*L[i]*D/2,2*B*L[i]*D/2]
    temp_m3=[x + y for x, y in zip(temp_w3, temp_i3)]
    
    y_flops.extend(temp3)
    y_mem.extend(temp_m3)
    #print(len(y_mem))
    if i<9:
        y_flops.append(2*B*L[i+1]*E*D) #word embedding部分
        #print(y_flops)
        y_mem.append(B*L[i+1]*E+E*D)
    
    L_base+=L[i]

y_mem=[i * 2  for i in y_mem]
y_mem=[sum(y_mem[:i+1]) for i in range(len(y_mem))]
# print(y_flops)
#print(y_flops[310]/y_mem[310]/2)
#y_mem=[i * 2 for i in y_mem]
#cumulative_list = [sum(y_mem[:i+1]) for i in range(len(y_mem))]
#print(max(y_mem)/1024/1024)
#print('hatey')
# print('loveyou')
# print(max(temp_i)*2)
# y_ratio=[x / y for x, y in zip(y_attn, y_block)]
# index=list(range(len(y_ratio)))
# cumulative_list1=[x/y for x,y in zip(y_flops,y_mem)]
# #print(cumulative_list[:330])
# print(num_m)
# cumulative_list = y_flops#[sum(y_mem[:i+1]) for i in range(len(y_mem))]
# indices = list(range(len(cumulative_list1)))
# print(len(indices))
# print(cumulative_list1[len(indices)-1])
# plt.plot(index, y_ratio,marker='o', markersize=0.5,linestyle='-', color='b')  
# plt.grid(True)
# for i in range(1,10):
#     plt.text(num_m[i], cumulative_list[num_m[i]], f'{i+1}')
#     print(np.array(cumulative_list[num_m[i]], dtype=float)*1e-7)
# print(np.array(cumulative_list1[116:125],dtype=float))   
#plt.xlabel('Index')
# plt.ylabel('FLOPs/MEMORY(FLOPs/BYTE)')
# plt.title('FLOPs/MEMORY Ada Plot_stage2')
# plt.savefig("./fm_ada_stage2.png")
#plt.ylabel('attn/block FLOPs')
#plt.title('attn/block FLOPs Plot')
#plt.savefig("./ratio_flops.png")
# y_flops.append(2*B*E**2*K**2*W*H)
# y_mem.append(E*E*K*K+B*E*W*H)
# a=[1.2382371840000002,6.178668544000001,17.299079168000002,37.098356736,68.113793024,112.94461132800001,193.176141824,319.84680960000003,537.654657024,795.27981875]
# b=[]
# for i in range(1,10):
#     b.append(a[i]-a[i-1])
# print(b)

# num1=[]
# num2=[]
# # quant convin mid
# num1.append(len(y_flops))
# num1.append(len(y_flops)+1)
# num1.append(len(y_flops)+2)
# temp4=[2*B*E**2*K**2*W*H,2*B*E*C*K**2*W*H,2*B*C**2*W*H*9,2*B*C**2*W*H*9,6*B*C**2*W*H,2*B*H**2*W**2*C,B*H**2*W**2*C,2*B*C**2*W*H,2*B*C**2*W*H*9,2*B*C**2*W*H*9]
# temp_w4=[E**2*K**2,E*C*K**2,C**2*K**2,C**2*K**2,3*C**2,B*C*H*W,B*H**2*W**2,C**2,C**2*K**2,C**2*K**2]
# temp_i4=[B*E*W*H,B*E*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H]
# temp_m4=[x + y for x, y in zip(temp_w4, temp_i4)]
# y_flops.extend(temp4)
# y_mem.extend(temp_m4)
# print(len(y_flops),len(y_mem))
# #6K^2C^2+12C^2+12BCHW
# # up s4
# num.append(len(y_flops))
# temp5=[2*B*C**2*W*H*9,2*B*C**2*W*H*9,6*B*C**2*W*H,2*B*H**2*W**2*C,B*H**2*W**2*C,2*B*C**2*W*H]
# temp_w5=[C**2*K**2,C**2*K**2,3*C**2,B*C*H*W,B*H**2*W**2,C**2]
# temp_i5=[B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H,B*C*W*H]
# temp_m5=[x + y for x, y in zip(temp_w5, temp_i5)]
# for j in range(3):
#     y_flops.extend(temp5)
#     y_mem.extend(temp_m5)
# y_flops.append(8*B*C**2*W*H*9)
# y_mem.append(C**2*K**2+B*C*W*H)
# print(len(y_flops),len(y_mem))
# #up s3
# num.append(len(y_flops))
# temp6=[4*B*C**2*W*H*9,2*B*C**2*W*H*9,2*B*C**2*W*H*9,2*B*C**2*W*H*9,2*B*C**2*W*H*9,2*B*C**2*W*H*9]
# temp_w6=[0.5*C**2*K**2,0.25*C**2*K**2,0.25*C**2*K**2,0.25*C**2*K**2,0.25*C**2*K**2,0.25*C**2*K**2]
# temp_i6=[B*C*2*W*2*H,B*C*0.5*2*W*2*H,B*C*0.5*2*W*2*H,B*C*0.5*2*W*2*H,B*C*0.5*2*W*2*H,B*C*0.5*2*W*2*H]
# y_flops.extend(temp6)
# y_flops.append(2*B*320*640*32*32*9)#SHORTCUTs
# y_flops.append(8*B*C**2*W*H*9)#up

# temp_m6=[x + y for x, y in zip(temp_w6, temp_i6)]
# y_mem.extend(temp_m6)
# y_mem.append(C*0.5*C*K*K+B*C*2*W*2*H)
# y_mem.append(0.5*C*0.5*C*K*K+B*0.5*C*2*W*2*H)
# print(len(y_flops),len(y_mem))
# #up s2
# num.append(len(y_flops))
# for j in range(6):
#     y_flops.append(8*B*C**2*W*H*9)
#     y_mem.append(0.5*C*0.5*C*K*K+B*0.5*C*4*W*4*H)
# y_flops.append(32*B*C**2*W*H*9)#up
# y_mem.append(0.5*C*0.5*C*K*K+B*0.5*C*4*W*4*H)
# print(len(y_flops),len(y_mem))
# #up s1
# num.append(len(y_flops))
# temp7=[16*B*C**2*W*H*9,8*B*C**2*W*H*9,8*B*C**2*W*H*9,8*B*C**2*W*H*9,8*B*C**2*W*H*9,8*B*C**2*W*H*9]
# temp_w7=[0.5*C*0.25*C*K*K,0.25*C*0.25*C*K*K,0.25*C*0.25*C*K*K,0.25*C*0.25*C*K*K,0.25*C*0.25*C*K*K,0.25*C*0.25*C*K*K]
# temp_i7=[B*0.5*C*8*W*8*H,B*0.25*C*8*W*8*H,B*0.25*C*8*W*8*H,B*0.25*C*8*W*8*H,B*0.25*C*8*W*8*H,B*0.25*C*8*W*8*H]
# temp_m7=[x + y for x, y in zip(temp_w7, temp_i7)]
# y_mem.extend(temp_m7)
# y_mem.append(C*0.5*C*0.25*K*K+B*0.5*C*8*W*8*H)
# y_mem.append(0.25*C*0.25*C*K*K+B*0.25*C*8*W*8*H)
# y_flops.extend(temp7)
# y_flops.append(2*B*160*320*128*128*9)#shortcut
# y_flops.append(32*B*C**2*W*H*9)#up
# print(len(y_flops),len(y_mem))
# #up s0
# num.append(len(y_flops))
# for j in range(6):
#     print(32*B*C**2*W*H*9)
#     y_flops.append(32*B*C**2*W*H*9)
#     y_mem.append(0.25*C*0.25*C*K*K+B*0.25*C*16*W*16*H)
# # convout
# y_flops.append(2*B*160*3*9*256*256)
# y_mem.append(0.25*C*3*K*K+B*0.25*C*16*W*16*H)
# print("hhh")
# y_mem=[i * 2  for i in y_mem]
# # #y_flops=[i * 1e-9 for i in y_flops]
# # cumulative_list=[x/y for x,y in zip(y_flops,y_mem)]
# # y_flops = [sum(y_flops[:i+1]) for i in range(len(y_flops))]
# y_mem=[sum(y_mem[:i+1]) for i in range(len(y_mem))]
# print("flops")
# print(max(y_flops)*1e-9)
# print("attn_flops")
# print(attn_flops*1e-9)
# print(attn_flops/max(y_flops))
# print(("memory"))

#print(y_flops.index(max(y_flops)))
#cumulative_list = [sum(y_mem[:i+1]) for i in range(len(y_mem))]
print(max(y_mem)/1024/1024/1024)
# print(np.array(cumulative_list[30:40], dtype=float))
# print(np.array(y_flops[30:40], dtype=float))
# print(np.array(y_mem[30:40], dtype=float))
# hello=[cumulative_list[i]>4000 for i in range(len(cumulative_list))]
# print(sum(hello))
# indices = list(range(len(cumulative_list)))
# # print(indices)
# plt.plot(indices, cumulative_list,marker='o', markersize=0.5,linestyle='-', color='b')  
# plt.xlabel('Index')
# plt.ylabel('FLOPs/MEMORY(FLOPs/BYTE)')
# plt.title('FLOPs/MEMORY all Plot')
# plt.xlabel('Index')
# plt.ylabel('FLOPs')
# plt.title('FLOPs Decode Plot')
# plt.xlabel('Index')
# plt.ylabel('MEMORY')
# plt.title('MEMORY ALL Plot depth=30')
# print(num)
# for i in range(10):
#     plt.text(num_m[i], cumulative_list[num_m[i]], f'{i+1}')
# for i in range(5):
#      plt.text(num[i], cumulative_list[num[i]], f'{i}')
# plt.grid(True)
# plt.savefig("./ratio_decode_s.png")

# import bisect
# def insert_x(a, x):
#     # 使用 bisect 模块查找插入位置
#     index = bisect.bisect_right(a.start_time, x)
#     # 在计算出的索引处插入 x
#     a.start_time.insert(index, x)
#     print(index)
#     return a.start_time

# # 假设 a 是一个包含 start_time 列表的对象
# class A:
#     def __init__(self, start_time):
#         self.start_time = start_time

# a = A([1, 3, 5, 7])
# x = 4

# # 插入 x
# result = insert_x(a, x)
# print(result)  # 输出: [1, 3, 4, 5, 7]
# #import numpy as np
# #import pandas as pd
# from typing import List, Dict, Tuple, Union, Optional

# # class Optimization:
# #     def __init__(self, file_path:str, FLOPs:List[float], Parameter:List[float], KV:List[float], FI:List[float],FO:List[float]):
# #         self.file_path = file_path
# #         self.frequency = 1 #GHz
# #         self.compute_peak_throughput_second = 147.456 # 144个tile的 TOPs per second
# #         self.compute_peak_throughput_cycle = self.compute_peak_throughput_second * 1000/self.frequency # 144个tile的 FLOPS per cycle 即147456 没有T,G
# #         self.DRAM_bandwidth = 73.728 * 1024 #MB PER Second 73.728GBps
# #         self.NoC_bandwidth = 32 * 1024 #MB PER Second
# #         self.SRAM_block_write_cycle = self.SRAM_block_read_cycle = 3.89324 * 1e-9 * self.frequency * 1e9 #64B second *(/s) = cycle
# #         self.tile_mapping = self.tile_number_to_coordinates(grid_size=12)
# #         self.FLOPs = FLOPs
# #         self.Parameter = Parameter #MB
# #         self.KV = KV #B
# #         self.FI = FI #B
# #         self.FO = FO
# #         #self.df = pd.read_excel(file_path)
# #         self.m_range = range(1, 80)
# #         self.x_range = range(1, 30)
# #         self.y_range = range(1, 40)
# #         self.z_range = range(1, 80)
# #         self.best_x = None
# #         self.best_value = float('inf')
# #         self.data = []
# #         self.tile_num=[] 
# #         self.write = [None] * 60
# #         self.time = [None] * 6

# #     def latency(self, flops:List[float], weight:List[float],tile_nums: List[int]): #y都是MB的即flops,
# #         tile_nums = [None] * len(flops)
# #         compute_latency = [None] * len(flops)
# #         compute_latency = [flops[i] * 144 / tile_nums[i] for i in range(len(flops))]
# #         # compute_latency[0] = flops[0] * 144 / tile_num[0]  # qkv m
# #         # compute_latency[3] = flops[4] * 144 / tile_num[3]  # f1
# #         # compute_latency[2] = flops[3] * 144 / tile_num[2]  # p
# #         # compute_latency[1] = flops[1] * 144 / tile_num[1]  # m1
# #         # 2 * weight_sram = tile_num - self.FIFO[this layer input] - self.FIFO[this layer output] - self.KV #double buffer 用满片上所有 其他扔给dram
# #         # 预留出来
# #         # weight_dram = weight - weight_sram 
# #         # 不需要考虑
# #         # 假如【layer1 和 layer2 全部串行且依赖，先后用同样的TILE】
# #         # 假如【layer1 和 layer2 全部串行且依赖，先后用同样的TILE】
# #         # 假如【layer1 和 layer2 全部串行且依赖，先后用同样的TILE】

# #         #TODO：还没考虑tile_num前后分配不同的时候，doublebuffer衔接的问题

# #         # 假如【layer1 和 layer2 先后用同样的TILE】double buffering 主要是在同一个tile内部用于提升计算和访存的并行性 依赖不依赖不重要，因为这个是weight哦
# #         # 循环加起来，最后剩一个
# #         # if weight_dram=0 则简单如下面else
# #         # else:
# #             # for 所有layer
# #                 # 如果weight_dram =< weight_sram
# #                 # latency += max(buffer2PEandPEcompute_current_layer, Dram2buffer_next_layer) + buffer2PEandPEcompute_next_layer 
# #                 # 如果weight_dram > weight_sram
# #                 # repeat_num = (weight_dram / weight_sram)向下取整-1
# #                 # latency += max(buffer2PEandPEcompute_current_layer, Dramwhichisweightsram2buffer_next_layer) + repeat_num * max(buffer2PEandPEcompute_next_layer, Dramwhichisweightsram2buffer_next_layer) + (weight_dram - repeat_num-1)的buffer2PEandPEcompute
# #         #没有double buffer 不考虑计算访存并行即能存在buffer就放在bufer，存不下才扔到DRAM去【比如一个块在计算的时候，另一个块可以提前把权重从dram读出来。】
# #         # 权重准备按理说不受计算的影响
# #         weight_ready_latency = [None] * len(weight)
        
# #         weight_ready_latency =[ ((weight[i] - tile_nums[i]) / self.DRAM_bandwidth * self.frequency + (weight[i] - tile_nums[i]) / self.SRAM_block_write_cycle + weight[i] * 1024 * 1024 / 64 * self.SRAM_block_read_cycle 
# #                                  if weight[i] > tile_nums[i] else weight[i]  * 1024 * 1024 / 64 * self.SRAM_block_read_cycle) for i in range(len(weight))]  # qkv

# #         # weight_ready_latency[0] = ((weight[0] - tile_num[0]) / self.DRAM_bandwidth * self.frequency + (weight[0] - tile_num[0]) / self.SRAM_block_write_cycle + weight[0] * 1024 * 1024 / 64 * self.SRAM_block_read_cycle if weight[0] > tile_num[0] else weight[0] / self.SRAM_block_read_cycle)  # qkv
# #         # weight_ready_latency[3] = ((weight[4] - tile_num[3]) / self.DRAM_bandwidth + (weight[4] - tile_num[3]) / self.SRAM_block_write_cycle + weight[4] / self.SRAM_block_read_cycle if weight[4] > tile_num[0] else weight[4] / self.SRAM_block_read_cycle)  # qkv
# #         # weight_ready_latency[2] = ((weight[3] - tile_num[2]) / self.DRAM_bandwidth + (weight[3] - tile_num[2]) / self.SRAM_block_write_cycle + weight[3] / self.SRAM_block_read_cycle if weight[3] > tile_num[2] else weight[3] / self.SRAM_block_read_cycle)  # qkv
# #         # weight_ready_latency[1] = ((weight[1] - tile_num[1]) / self.DRAM_bandwidth + (weight[1] - tile_num[1]) / self.SRAM_block_write_cycle + weight[1] / self.SRAM_block_read_cycle if weight[1] > tile_num[1] else weight[1] / self.SRAM_block_read_cycle)  # qkv

# #         '''input 现有的时间

# # 上一层计算时间 计算的同时完成weight 加载

# # 还需要判断input是不是能在buffer里面 不然得送去dram【预留出来output了】

# #  我那个是所有计算完成的时间

# # input写入buffer 再从buffer传到下一个layer所在tile的时间
# # 如果就在这儿 就不用传了 写入时间加上即可

# # 权重准备时间和计算时间重叠进行的


# # 上一层输出结果会存进来的 

# # max后计算开始是上一个地方start上它是现在的start'''

# #         # 没有double buffer 的存在
# #         output_write_latency = [self.FO[i] / 64 * self.SRAM_block_write_cycle for i in range(len(FO))]
# #         tiles2grid = self.map_tiles_to_grid(tile_nums,)
# #         comm_map = self.generate_comm_map(tiles2grid,self.tile_mapping)
# #         output_NoC_latency = []
# #         for i in range(len(tiles2grid)-1):
# #             output_NoC_latency = self.specific_tile_communication_time(tiles2grid[i], tiles2grid[i+1], comm_map[i], FO[i])

# #         #total = val0 + 2 * max(val1, val2, val3)
# #         #return total, val0, val1, val2, val3
        

# #     def tile_number_to_coordinates(self,grid_size=12):
# #         """
# #         tile numbers[0,1,2...,143] to coordinates[(0,0)..,(11,11)] .
# #         """
# #         tile_mapping = {}
# #         tile_number = 0
# #         for row in range(grid_size):
# #             for col in range(grid_size):
# #                 tile_mapping[tile_number] = (row, col)
# #                 tile_number += 1
# #         return tile_mapping

# #     def coordinates_to_tile_number(self,coordinates:List[Tuple], tile_mapping:Dict[int,Tuple]):
# #         """
# #         coordinates to tile numbers.
# #         """
# #         tile_number_mapping = {v: k for k, v in tile_mapping.items()}
# #         return [tile_number_mapping[coord] for coord in coordinates]

# #     def generate_comm_map(self,layer_coordinates:List[List[Tuple]], tile_mapping:Dict[int,Tuple]):
# #         """
# #         communication map for tiles across different layers.
        
# #         :param layer_coordinates: = [
# #             [(0, 0), (0, 1), (0, 2)],  # Layer 1 tiles
# #             [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # Layer 2 tiles
# #             [(1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]  # Layer 3 tiles
# #         ]
# #         :param tile_mapping: Dictionary mapping tile numbers 2 coordinates.
# #         :return: [{3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2], 10: [2], 11: [2], 12: [0], 13: [1], 14: [2], 15: [2], 16: [2]}, {0: [24, 25, 26, 27, 17], 1: [25, 24, 26, 27, 17], 2: [26, 25, 27, 17, 24]}] 
# #         Communication Map from Layer 0 to Layer 1:
# #     Tile 0 in Layer 0 communicates with Tiles [12, 13, 3, 14] in Layer 1
# #     Tile 1 in Layer 0 communicates with Tiles [13, 3, 12, 14] in Layer 1
# #     Tile 2 in Layer 0 communicates with Tiles [3, 14, 4, 13] in Layer 1
# #         """
# #         comm_maps = []

# #         # coordinates to tile numbers
# #         layer_tile_numbers = [self.coordinates_to_tile_number(layer, tile_mapping) for layer in layer_coordinates]

# #         for layer_index, current_layer_tiles in enumerate(layer_tile_numbers[:-1]):
# #             next_layer_tiles = layer_tile_numbers[layer_index + 1]
# #             comm_map = {}
            
# #             # Get coordinates for current and next layer tiles
# #             #[(0, 0), (0, 1), (0, 2)] 3
# #             current_layer_coords = [tile_mapping[tile] for tile in current_layer_tiles]
# #             #[(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)] 14
# #             next_layer_coords = [tile_mapping[tile] for tile in next_layer_tiles]

# #             for i, current_tile in enumerate(current_layer_coords):
# #                 distances = []
# #                 for j, next_tile in enumerate(next_layer_coords):
# #                     distance = self.manhattan_distance(current_tile[0],current_tile[1], next_tile[0],next_tile[1])
# #                     distances.append((distance, next_layer_tiles[j]))  # Store the tile number instead of index
                
# #                 # 取最近的tile
# #                 distances.sort()
# #                 closest_tiles = [tile for _, tile in distances[:max(1, len(next_layer_tiles)//len(current_layer_tiles))]]
# #                 comm_map[current_layer_tiles[i]] = closest_tiles
# #                 # 分给最近的len(next_layer_tiles)//len(current_layer_tiles)
# #             comm_maps.append(comm_map)

# #         return comm_maps
        
# #     def map_tiles_to_grid(self, tile_nums:List[int], grid_size=12):
# #         """
# #         map 从左到右 从上到下，顺序排列
# #         tile_nums = [3, 14, 16]  #  Layer 1 has 3 tiles, Layer 2 has 14 tiles, Layer 3 has 16 tiles

# #         :return: list [[(0, 0), (0, 1), (0, 2)], [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)], [(1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]]
# #         """
# #         all_positions = [(row, col) for row in range(grid_size) for col in range(grid_size)]
# #         used_positions = set()
        
# #         coordinates = [None] * len(tile_nums)
# #         index = 0

# #         for layer_index, tile_num in enumerate(tile_nums):
# #             if index + tile_num > len(all_positions):
# #                 raise ValueError("Not enough space in the grid to allocate all tiles.")
            
# #             layer_positions = []
            
# #             for _ in range(tile_num):
# #                 while all_positions[index] in used_positions:
# #                     index += 1
# #                     if index >= len(all_positions):
# #                         raise ValueError("Ran out of available positions.")
                
# #                 position = all_positions[index]
# #                 layer_positions.append(position)
# #                 used_positions.add(position)
# #                 index += 1
            
# #             coordinates[layer_index] = layer_positions
        
# #         return coordinates
# #     # 废弃
# #     # def map_tiles_to_grid(tile_num:List[int], grid_size=12):
# #     #     # Example: tile_num = [1,2,]
# #     #     coordinates = [None] * len(tile_num)
        

# #     #     # map方法：从左到右，从上到下
# #     #     for i in range(len(tile_num)):
# #     #         row = tile_num[i] // grid_size # 43//12= 3
# #     #         col = tile_num[i] % grid_size
# #     #         position = [(x, y) for x in range(row) for y in range(grid_size)]
# #     #         temp = [(row + 1, y) for y in range(col)]
# #     #         position.extend(temp)
# #     #         # Ensure position is not reused
# #     #         coordinates[i]=position
# #     #     return coordinates

   



# #     def manhattan_distance(self, x1, y1, x2, y2):
# #         return abs(x1 - x2) + abs(y1 - y2)

# #     def specific_tile_communication_time(self, src_tiles:List[Tuple], dest_tiles:List[Tuple], comm_map:Dict[int,List], total_data_size):
        
# #     #         # Example usage
# #     # # Layer 1 tiles (positions as tuples)
# #     # layer1_tiles = [(1, 2), (2, 3), (4, 5), (6, 7), (7, 8)]

# #     # # Layer 2 tiles (positions as tuples)
# #     # layer2_tiles = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), 
# #     #                 (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), 
# #     #                 (18, 19), (19, 20)]

# #     # # Define communication map
# #     # # Key: index of source tile in layer1_tiles, Value: list of indices of destination tiles in layer2_tiles
# #     # comm_map = {
# #     #     0: [0, 1],  # Tile 0 in Layer 1 communicates with Tile 0 and 1 in Layer 2
# #     #     1: [2],     # Tile 1 in Layer 1 communicates with Tile 2 in Layer 2
# #     #     2: [3, 4],  # Tile 2 in Layer 1 communicates with Tile 3 and 4 in Layer 2
# #     #     3: [5],     # Tile 3 in Layer 1 communicates with Tile 5 in Layer 2
# #     #     4: [6, 7]   # Tile 4 in Layer 1 communicates with Tile 6 and 7 in Layer 2
# #     # }
# #         # TODO: 全部平分处理。且并未考虑output的东西传到下一个layer的tile的buffer会放不下的情况。
# #         # 这里默认FIFO都能在buffer里面 条件要写一下
# #         max_comm_time = 0
# #         num_src_tiles = len(src_tiles)
# #         # Data size per source tile
# #         data_per_src_tile = total_data_size / num_src_tiles
# #         for src_index, dest_indices in comm_map.items():
# #             src_tile = self.tile_mapping[src_index]
# #             num_dest_tiles = len(dest_indices)
# #             data_per_src_tile_allocate_to_dest = data_per_src_tile / num_dest_tiles
# #             for dest_index in dest_indices:
# #                 dest_tile = self.tile_mapping[dest_index]
# #                 # Calculate Manhattan distance (hops)
# #                 hops = self.manhattan_distance(src_tile[0], src_tile[1], dest_tile[0], dest_tile[1])
# #                 transmission_time = data_per_src_tile_allocate_to_dest / self.NoC_bandwidth # second
# #                 # Calculate total communication time for this path
# #                 comm_time = hops * transmission_time
# #                 # Update max communication time
# #                 if comm_time > max_comm_time:
# #                     max_comm_time = comm_time
# #         return max_comm_time


# #     def is_feasible(self, x):
# #         return x[0] + 2 * x[1] + x[2] + 2 * x[3] == 144

# #     def optimize(self):
# #         # Convert lists to numpy arrays
# #         FLOPs = np.array(self.FLOPs)
# #         Parameter = np.array(self.Parameter)
# #         KV = np.array(self.KV)
# #         FIFO = np.array(self.FIFO)

# #         compute_cycle = FLOPs / 147456  # Adjust based on actual formula

# #         for i in range(10):
# #             temp_compute = compute_cycle[i * 6:(i + 1) * 6]
# #             temp_weight = Parameter[i * 6:(i + 1) * 6]
# #             best_x = None
# #             best_value = float('inf')

# #             for m in self.m_range:
# #                 for x in self.x_range:
# #                     for y in self.y_range:
# #                         for z in self.z_range:
# #                             tile_nums = [m,x,x,y,z,z]
# #                             if self.is_feasible([m, x, y, z]):
# #                                 current_value = self.compute_latency(temp_compute, temp_weight,tile_nums) 
# #                                 if current_value[0] < best_value:
# #                                     best_value = current_value[0]
# #                                     best_x = [m, x, y, z]
# #                                     self.time = current_value[1:5]

# #             print(f"stage:{i}")
# #             if best_x:
# #                 print(f"Optimized result: m = {best_x[0]}, x = {best_x[1]}, y = {best_x[2]}, z = {best_x[3]}")
# #                 print(f"Objective value: {best_value}")
# #                 print(self.time)
# #             else:
# #                 print("No feasible solution found.")

# #             self.write[i * 6:(i + 1) * 6 - 1] = [best_x[0], best_x[1], best_x[1], best_x[2], best_x[3], best_x[3]]

# #         self.write = self.write[:60]
# #         self.df['tile num'] = self.write
# #         self.df.to_excel(self.file_path, index=False)

# #         print(f"数据已保存到 {self.file_path}")
# def tile_num_to_bank(bank_num,grid=12):
#         tile_bank_vocabulary = {}
#         for i in range(grid**2):
#             tile_bank_vocabulary[i] = list(range(bank_num))
#         return tile_bank_vocabulary
# def main():
#     # a = Optimization('1',[1],[1],[1],[1],[10,15,5])
#     # tiles_nums=[3,14,16]
#     # FO = [10,15,5]
#     # tiles2grid = a.map_tiles_to_grid(tiles_nums,)
#     # comm_map = a.generate_comm_map(tiles2grid,a.tile_mapping)
#     # output_NoC_latency=[]
#     # for i in range(len(tiles2grid)-1):
#     #     output_NoC_latency.append (a.specific_tile_communication_time(tiles2grid[i], tiles2grid[i+1], comm_map[i], FO[i]))
#     # print(output_NoC_latency)
#     # print(comm_map)
#     # print(tiles2grid)
#     #print(tile_num_to_bank(3,))
#     for i,tile_id in enumerate([1]):
#         print(i)
#         print(tile_id)
# if __name__ == "__main__":
#     main()

# # def manhattan_distance(coord1, coord2):
# #     """
# #     Calculates the Manhattan distance between two coordinates.
# #     """
# #     return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

# # def tile_number_to_coordinates(grid_size):
# #     """
# #     tile numbers[0,1,2...,143] to coordinates[(0,0)..,(11,11)] .
# #     """
# #     tile_mapping = {}
# #     tile_number = 0
# #     for row in range(grid_size):
# #         for col in range(grid_size):
# #             tile_mapping[tile_number] = (row, col)
# #             tile_number += 1
# #     return tile_mapping

# # def coordinates_to_tile_number(coordinates, tile_mapping):
# #     """
# #      coordinates to tile numbers.
# #     """
# #     tile_number_mapping = {v: k for k, v in tile_mapping.items()}
# #     return [tile_number_mapping[coord] for coord in coordinates]

# # def generate_comm_map(layer_coordinates, tile_mapping):
# #     """
# #     communication map for tiles across different layers.
    
# #     :param layer_coordinates: = [
# #         [(0, 0), (0, 1), (0, 2)],  # Layer 1 tiles
# #         [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # Layer 2 tiles
# #         [(1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]  # Layer 3 tiles
# #     ]
# #     :param tile_mapping: Dictionary mapping tile numbers 2 coordinates.
# #     :return: Communication Map from Layer 0 to Layer 1:
# #   Tile 0 in Layer 0 communicates with Tiles [12, 13, 3, 14] in Layer 1
# #   Tile 1 in Layer 0 communicates with Tiles [13, 3, 12, 14] in Layer 1
# #   Tile 2 in Layer 0 communicates with Tiles [3, 14, 4, 13] in Layer 1
# #     """
# #     comm_maps = []

# #     # coordinates to tile numbers
# #     layer_tile_numbers = [coordinates_to_tile_number(layer, tile_mapping) for layer in layer_coordinates]

# #     for layer_index, current_layer_tiles in enumerate(layer_tile_numbers[:-1]):
# #         next_layer_tiles = layer_tile_numbers[layer_index + 1]
# #         comm_map = {}
        
# #         # Get coordinates for current and next layer tiles
# #         #[(0, 0), (0, 1), (0, 2)] 3
# #         current_layer_coords = [tile_mapping[tile] for tile in current_layer_tiles]
# #         #[(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)] 14
# #         next_layer_coords = [tile_mapping[tile] for tile in next_layer_tiles]

# #         for i, current_tile in enumerate(current_layer_coords):
# #             distances = []
# #             for j, next_tile in enumerate(next_layer_coords):
# #                 distance = manhattan_distance(current_tile, next_tile)
# #                 distances.append((distance, next_layer_tiles[j]))  # Store the tile number instead of index
            
# #             # 取最近的tile
# #             distances.sort()
# #             closest_tiles = [tile for _, tile in distances[:max(1, len(next_layer_tiles)//len(current_layer_tiles))]]
# #             comm_map[current_layer_tiles[i]] = closest_tiles
# #             # 分给最近的len(next_layer_tiles)//len(current_layer_tiles)
# #         comm_maps.append(comm_map)

# #     return comm_maps

# # def main():
# #     grid_size = 12
# #     tile_mapping = tile_number_to_coordinates(grid_size)

# #     # Example layer coordinates
# #     layer_coordinates = [
        
# #         [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # Layer 2 tiles
# #        [(0, 0), (0, 1), (0, 2)],  # Layer 1 tiles
# #         [(1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]  # Layer 3 tiles
# #     ]

# #     # Generate communication maps
# #     comm_maps = generate_comm_map(layer_coordinates, tile_mapping)
# #     print(comm_maps)
# #     # Print communication maps
# #     for layer_index, comm_map in enumerate(comm_maps):
# #         print(f"Communication Map from Layer {layer_index} to Layer {layer_index + 1}:")
# #         for src_tile, dst_tiles in comm_map.items():
# #             print(f"  Tile {src_tile} in Layer {layer_index} communicates with Tiles {dst_tiles} in Layer {layer_index + 1}")

# # if __name__ == "__main__":
# #     main()

# # #import numpy as np

# # def manhattan_distance(coord1, coord2):
# #     """
# #     Calculates the Manhattan distance between two coordinates.
# #     """
# #     return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

# # def generate_comm_map(layer_coordinates):
# #     """
# #     Generates a communication map for tiles across different layers.
    
# #     :param layer_coordinates: List of lists, where each sublist contains the coordinates of tiles in a layer.
# #     :return: List of dictionaries where each dictionary represents communication from a layer to the next.
# #     """
# #     comm_maps = []

# #     for layer_index, current_layer in enumerate(layer_coordinates[:-1]):
# #         next_layer = layer_coordinates[layer_index + 1]
# #         comm_map = {}
        
# #         for i, current_tile in enumerate(current_layer):
# #             distances = []
# #             for j, next_tile in enumerate(next_layer):
# #                 distance = manhattan_distance(current_tile, next_tile)
# #                 distances.append((distance, j))
# #             # Sort by distance and get the closest tiles (you can adjust the number as needed)
# #             distances.sort()
# #             closest_tiles = [idx for _, idx in distances[:max(1, len(next_layer)//len(current_layer))]]
# #             comm_map[i] = closest_tiles
        
# #         comm_maps.append(comm_map)

# #     return comm_maps

# # def main():
# #     # Example layer coordinates
# #     layer_coordinates = [
# #         [(0, 0), (0, 1), (0, 2)],  # Layer 1 tiles
# #         [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # Layer 2 tiles
# #         [(1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]  # Layer 3 tiles
# #     ]

# #     # Generate communication maps
# #     comm_maps = generate_comm_map(layer_coordinates)
    
# #     # Print communication maps
# #     for layer_index, comm_map in enumerate(comm_maps):
# #         print(f"Communication Map from Layer {layer_index} to Layer {layer_index + 1}:")
# #         for src_tile, dst_tiles in comm_map.items():
# #             print(f"  Tile {src_tile} in Layer {layer_index} communicates with Tiles {dst_tiles} in Layer {layer_index + 1}")

# # if __name__ == "__main__":
# #     main()

# # def generate_comm_map(layer_coordinates):
# #     """
# #     Generates a communication map for tiles across different layers.

# #     :param layer_coordinates: List of lists, where each sublist contains the coordinates of tiles in a layer.
# #     :return: Dictionary where keys are tile indices in a layer and values are lists of tile indices it communicates with in the next layer.
# #     """
# #     # Flatten coordinates into index-based layers
# #     layer_tile_indices = [list(range(len(layer))) for layer in layer_coordinates]
# #     comm_map = {}

# #     # Function to map each tile to tiles in the next layer
# #     def find_closest_tiles(current_tile_index, num_tiles_to_communicate, next_layer_tiles):
# #         # Simple example: distribute tiles evenly
# #         total_next_tiles = len(next_layer_tiles)
# #         start_index = (current_tile_index * total_next_tiles) // len(layer_tile_indices[layer_index])
# #         end_index = ((current_tile_index + 1) * total_next_tiles) // len(layer_tile_indices[layer_index])
# #         return list(range(start_index, end_index))

# #     # Build communication map
# #     for layer_index, current_layer_tiles in enumerate(layer_tile_indices): #enumerate遍历列表
# #         if layer_index < len(layer_tile_indices) - 1:
# #             next_layer_tiles = layer_tile_indices[layer_index + 1]
# #             for tile_index in range(len(current_layer_tiles)):
# #                 num_tiles_to_communicate = len(next_layer_tiles) // len(current_layer_tiles)
# #                 comm_map[tile_index] = find_closest_tiles(tile_index, num_tiles_to_communicate, next_layer_tiles)

# #     return comm_map

# # def main():
# #     # Example layer coordinates
# #     layer_coordinates = [
# #         [(0, 0), (0, 1), (0, 2)],  # Layer 1 tiles
# #         [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # Layer 2 tiles
# #         [(1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]  # Layer 3 tiles
# #     ]

# #     comm_map = generate_comm_map(layer_coordinates)
    
# #     # Print communication map
# #     print("Communication Map:")
# #     for src_tile, dst_tiles in comm_map.items():
# #         print(f"Tile {src_tile} in Layer communicates with Tiles {dst_tiles}")

# # if __name__ == "__main__":
# #     main()


# # # tile_num=[3,14,16]
# # # coordinates = [None] * len(tile_num)
        
# # # grid_size=12
# # # # map方法：从左到右，从上到下
# # # for i in range(len(tile_num)):
# # #     row = tile_num[i] // grid_size # 43//12= 3
# # #     col = tile_num[i] % grid_size
# # #     if row == 0:
# # #         position = [(0,y) for y in range(col)]
# # #     else:
# # #         position = [(x, y) for x in range(row) for y in range(grid_size)]
# # #         temp = [(row + 1, y) for y in range(col)]
# # #         position.extend(temp)
# # #     # Ensure position is not reused
# # #     coordinates[i]=position
# # #     # 没法接着往后走，没法接着往前走
# # # print(coordinates)
# # def generate_unique_coordinates(tile_nums, grid_size=12):
# #     """
# #     Generates unique coordinates for multiple layers in a grid from left to right, top to bottom.

# #     :param tile_nums: List of integers where each integer represents the number of tiles for each layer.
# #     :param grid_size: Size of the grid (e.g., 12 for a 12x12 grid).
# #     :return: Dictionary where keys are layer indices and values are lists of coordinates.
# #     """
# #     all_positions = [(row, col) for row in range(grid_size) for col in range(grid_size)]
# #     used_positions = set()
    
# #     coordinates = [None] * len(tile_nums)
# #     index = 0

# #     for layer_index, tile_num in enumerate(tile_nums):
# #         if index + tile_num > len(all_positions):
# #             raise ValueError("Not enough space in the grid to allocate all tiles.")
        
# #         layer_positions = []
        
# #         for _ in range(tile_num):
# #             while all_positions[index] in used_positions:
# #                 index += 1
# #                 if index >= len(all_positions):
# #                     raise ValueError("Ran out of available positions.")
            
# #             position = all_positions[index]
# #             layer_positions.append(position)
# #             used_positions.add(position)
# #             index += 1
        
# #         coordinates[layer_index] = layer_positions
    
# #     return coordinates

# # def main():
# #     # Number of tiles for each layer
# #     tile_nums = [3, 14, 16]  # Example: Layer 1 has 3 tiles, Layer 2 has 14 tiles, Layer 3 has 16 tiles
# #     grid_size = 12  # Size of the grid (12x12)
    
# #     # Generate the tile coordinates
# #     layer_coordinates = generate_unique_coordinates(tile_nums, grid_size)
# #     print(layer_coordinates)
# #     # Print the coordinates for each layer
# #     # for layer_index, coords in layer_coordinates.items():
# #     #     print(f"Layer {layer_index + 1} Tile Coordinates:", coords)

# # if __name__ == "__main__":
# #     main()
