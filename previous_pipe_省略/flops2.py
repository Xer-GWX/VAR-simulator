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
#L=[1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
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
    
    temp_w=[6*D*(D+1),3*D*(D+1),2*B*(L[i]+L_base)*D,2*B*(L[i]+L_base)*D,D*(D+1),4*D*(D+1),D*(4*D+1)]
    temp_i=[2*B*L[i]*D,2*B*L[i]*D,2*B*L[i]*D,2*B*16*L[i]*(L[i]+L_base),2*B*L[i]*D,2*B*L[i]*D,2*B*L[i]*4*D]
    #print(2*B*16*L[i]*(L[i]+L_base))
    temp_m=[x + y for x, y in zip(temp_w, temp_i)]
    for j in range(30):
        attn_flops += 4*B*L[i]*(L[i]+L_base)*D + 4*B*L[i]*(L[i]+L_base)*D
        y_flops.extend(temp)
        y_mem.extend(temp_m)
    y_attn.append(4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*D**2)
    y_block.append(24*B*L[i]*D**2+12*B*L[i]*D**2+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*D**2+16*B*L[i]*D**2+16*B*L[i]*D**2)
    

    temp3=[8*B*L[i]*D**2,16*B*L[i]*D**2]
    temp_w3=[2*D*(D+1),4*D*(D+1)]
    temp_i3=[2*B*L[i]*D,2*B*L[i]*D]
    temp_m3=[x + y for x, y in zip(temp_w3, temp_i3)]
    
    y_flops.extend(temp3)
    y_mem.extend(temp_m3)
    #print(len(y_mem))
    if i<9:
        y_flops.append(2*B*L[i+1]*E*D) #word embedding部分
        #print(y_flops)
        y_mem.append(B*L[i+1]*E+E*D)
    
    L_base+=L[i]
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
y_mem=[i   for i in y_mem]
# #y_flops=[i * 1e-9 for i in y_flops]
# cumulative_list=[x/y for x,y in zip(y_flops,y_mem)]
# y_flops = [sum(y_flops[:i+1]) for i in range(len(y_flops))]
y_mem=[sum(y_mem[:i+1]) for i in range(len(y_mem))]
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
