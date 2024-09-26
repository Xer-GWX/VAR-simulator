#给定ra tree 找到最优tilenum,等
# optimization.py

# optimization.py

#import numpy as np
#import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
import bisect
from draw import draw_pipeline
class Bank():
    def __init__(self, bank_id=None, status=None, layer_id=None, sram_type=None, start_time=None, end_time=None):
        self.bank_id = bank_id
        self.status = status            #true: 正在写/读 其他勿扰， False:空闲，可以读写
        self.layer_id = layer_id if layer_id is not None else []        #占用的是哪个layer []
        self.sram_type = sram_type if sram_type is not None else []     #weight/FIFO类别 []
        self.start_time = start_time if start_time is not None else []
        self.end_time = end_time if end_time is not None else []

    def update_bank(self, status=None, layer_id=None, sram_type=None, start_time=None, end_time=None):
        if status is not None:
            self.status.append(status)
        if layer_id is not None:
            self.layer_id.append(layer_id)
        if sram_type is not None:
            self.sram_type.append(sram_type)
        if start_time is not None:
            self.start_time.append(start_time)
        if end_time is not None:
            self.end_time.append(end_time)

    def __repr__(self):
        return f"Bank(status={self.status}, layer_id={self.layer_id}, sram_type={self.sram_type}, start_time={self.start_time}, end_time={self.end_time})"

class Noc_latency():
    def __init__(self,Noc_latency_duration:List[float],):
        Noc_latency_duration.insert(0,0)
        self.Noc_duration = Noc_latency_duration
        #print(Noc_latency_duration)
        self.Noc_start_time = [None] * len(self.Noc_duration) #FO 函数
        self.Noc_end_time = self.Noc_start_time + self.Noc_duration
        #TODO:给duration扩写一下[0,[原来]] ok了

# 以后把这两个类统一成一个tile的

class Compute_unit_latency():
    def __init__(self,layer_id,Compute_duration,) -> None:
        # tile 0-143 其中每个有layer_id，和计算时间
        #layer_id = [[0,2],[0,1,2],[2]] #第i个tile有第几个layer在计算
        #tiles_id = [[0,1],[1],[0,1,2]]
        self.layer_id = self.layer_id = layer_id if layer_id is not None else []
        self.Compute_duration = self.Compute_duration = Compute_duration if Compute_duration is not None else []
        self.Compute_start_time = [None] * len(self.Compute_duration)#[[None for _ in sublist] for sublist in self.Compute_duration]#[None] * len(self.Compute_duration) #compute 函数
        self.Compute_end_time = [None] * len(self.Compute_duration)#self.Compute_start_time + self.Compute_duration
        #TODO:给duration扩写一下[0,[原来]]
    def update_Compute_end_time(self,index,Compute_start_time):
        self.Compute_end_time[index] = self.Compute_start_time[index] + self.Compute_duration[index]


class Optimization():
    def __init__(self, file_path:str, FLOPs:List[float], Parameter:List[float], KV:List[float], FI:List[float],FO:List[float]):
        self.file_path = file_path
        self.frequency = 1 #GHz
        self.compute_peak_throughput_second = 147.456 # 144个tile的 TOPs per second
        self.compute_peak_throughput_cycle = self.compute_peak_throughput_second * 1000/self.frequency # 144个tile的 FLOPS per cycle 即147456 没有T,G
        self.DRAM_bandwidth = 73.728 * 1024 #MB PER Second 73.728GBps
        self.NoC_bandwidth = 32 * 1024 #MB PER Second
        self.SRAM_block_write_cycle = self.SRAM_block_read_cycle = 3.89324 * 1e-9 * self.frequency * 1e9 #64B second *(/s) = cycle
        self.tile_mapping = self.tile_number_to_coordinates(grid_size=12)
        self.FLOPs = FLOPs
        self.Parameter = Parameter #MB
        self.KV = KV #B
        self.FI = FI #B
        self.FO = FO
        #self.df = pd.read_excel(file_path)
        self.m_range = range(1, 80)
        self.x_range = range(1, 30)
        self.y_range = range(1, 40)
        self.z_range = range(1, 80)
        self.best_x = None
        self.best_value = float('inf')
        self.data = []
        self.tile_num=[] 
        self.write = [None] * 60
        self.time = [None] * 6
    '''#不考虑DRAM
layer_indexi+1:  tile_id [2,3,4] 【遍历寻找最优】
               tile_grid 【从左到右上到下顺序】
               com_map 【上一个layer 对应 下一个layer 方法closest】
               weight: weight_value
                       weight_bank_num[0,1] 【bank怎么分配，约束条件，也遍历吗】【每个tile都是这么分配】搞一个true not true矩阵
                       start_time[]【假如layeri weight和FIFO的tile_id重叠 bank_num不重叠/tile_id不重叠】则start_time = 上一个layer的start_time
                                   【假如layeri的tile_id & bank_num和这层没有重叠】则start_time = 
                       end_time = start_time + write + read
               FIFO：  FIFO_value
                       FIFO_bank_num[1,2]
                       start_time[]
                       end_time = start_time + write + read
               
                              '''
    
    def SRAM_latency(self, bank_num,Noc_latency:Noc_latency,Compute_latency:Compute_unit_latency, weight_bank_allocate: List[List[int]],FIFO_bank_allocate:List[List[int]], tiles_nums: List[int]):
        '''
        weight_bank_allocate: 分配的bank列表
        tiles_nums: 每一层所用的tile编号
        '''
        tile_bank_vocabulary = self.tile_num_to_bank(bank_num)  # 每个tile会有特定的bank数量
        tiles_id = self.tile_number_to_id(tiles_nums)  # 根据coordinate对应的tile ID
        tiles_id = [[0,1],[1],[0,1,2]]
        #Noc_latency = [[],[],[]]
        # 
        for layer_id in range(len(tiles_id)):  # 遍历每一层
            print(f"Layer {layer_id}: Before assignment")
            #for tile_id in tiles_id[i]:
                #print(f"Tile {tile_id}: {[vars(bank) for bank in tile_bank_vocabulary[tile_id]]}")
            
            for j in range(len(tiles_id[layer_id])):  # 遍历当前层中的每个tile
                for k in range(len(weight_bank_allocate[layer_id])):  # 遍历分配的bank
                    bank = tile_bank_vocabulary[tiles_id[layer_id][j]][weight_bank_allocate[layer_id][k]]
                    bank.layer_id.append(layer_id)
                    bank.sram_type.append('weight')
                    last_end_time = 0
                    for previous_layer in range(layer_id):
                        for tile_index,tile_id in enumerate(tiles_id[previous_layer]):
                            for bank_index,bank_id in enumerate(weight_bank_allocate[previous_layer]):
                                bank_temp = tile_bank_vocabulary[tiles_id[previous_layer][tile_index]][weight_bank_allocate[previous_layer][bank_index]]
                                if tile_id==tiles_id[layer_id][j] and bank_id==weight_bank_allocate[layer_id][k]:
                                    if bank_temp.end_time:
                                        last_end_time = max(last_end_time, bank_temp.end_time[-1])
                    #last_end_time = bank.end_time[-1] if bank.end_time else 0
                    bank.start_time.append(last_end_time)  # 设置当前的start_time
                    weight_average_allocate = self.Parameter[layer_id]/len(tiles_id[layer_id])/len(weight_bank_allocate[layer_id])
                    #self.Parameter = [3,1,1]
                    some_duration = self.Parameter[layer_id]#weight_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_write_cycle + weight_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_read_cycle 
                    # 凑上去
                    bank.end_time.append(last_end_time + some_duration)  # 根据某个持续时间设置end_time
                    # if not bank.status:  # 如果当前status是False，表示bank空闲
                    #     bank.status = True
                        
                    # else:  # 如果当前status是True，表示bank正被使用
                    #     # current_time需要推进到end_time之后
                    #     current_time = bank.end_time[-1] if bank.end_time else 0
                    #     bank.status = True
                    #     bank.layer_id.append(i)
                    #     bank.sram_type.append('weight')
                    #     bank.start_time.append(current_time)
                    #     bank.end_time.append(current_time + some_duration)

                    # # 在end_time之后设置status为False
                    # bank.status = False
                    print(f"Layer {layer_id}, Tile {tiles_id[layer_id][j]}, Bank {weight_bank_allocate[layer_id][k]} assigned with {bank.sram_type[-1]} start time {bank.start_time[-1]} and end time {bank.end_time[-1]}")

            print(f"Layer {layer_id}: After assignment")
            #for tile_id in tiles_id[i]:
             #   print(f"Tile {tile_id}: {[vars(bank) for bank in tile_bank_vocabulary[tile_id]]}")

            # 更新current_time到下一层的起始时间
              # 更新current_time到下一层的起始时间，确保 end_time 列表非空
        #     current_time = max(
        #     bank.end_time[-1] if bank.end_time else 0 
        #     for tile_id in tiles_id[i] 
        #     for bank in tile_bank_vocabulary[tile_id]
        # )
            # current tiem
        #print(tile_bank_vocabulary)
        '''for FIFO_index in range(len(FIFO_bank_allocate[layer_id])):
                        bank = tile_bank_vocabulary[tiles_id[layer_id][j]][FIFO_bank_allocate[layer_id][FIFO_index]]
                        bank.layer_id.append(layer_id)
                        bank.sram_type.append('FI')
                        FI_last_end_time = 0
                        # 把所有的weight传输打断，优先本层的FI
                        for previous_layer in range(layer_id):
                            for tile_index,tile_id in enumerate(tiles_id[previous_layer]):#fifo还是weight
                                for bank_index,bank_id in enumerate(weight_bank_allocate[previous_layer]):
                                    bank_temp = tile_bank_vocabulary[tiles_id[previous_layer][tile_index]][weight_bank_allocate[previous_layer][bank_index]]
                                    if tile_id==tiles_id[layer_id][j] and bank_id==weight_bank_allocate[layer_id][FIFO_index]:
                                        FI_last_end_time = max(FI_last_end_time, Noc_latency.Noc_end_time[layer_id])
                                        if FI_last_end_time >= bank_temp.end_time:
                                            bank_temp.layer_id.append(layer_id)
                                            bank_temp.sram_type.append('FIFO')
                                            bank_temp.start_time.append(FI_last_end_time) 
                                            FI_average_allocate = self.FI[layer_id]/len(tiles_id)/len(FIFO_bank_allocate)
                                            FI_some_duration = FI_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_write_cycle + FI_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_read_cycle 
                                            bank_temp.end_time.append(FI_last_end_time + FI_some_duration)
                                            pass #没有问题哦，可以正常进行后续操作
                                        else: #需要打断weight的传输了
                                            weight_bank_interrupted = bank_temp.end_time[-1] - FI_last_end_time
                                            bank_temp.end_time[-1] = FI_last_end_time
                                            bank.layer_id.append(layer_id)
                                            bank.sram_type.append('weight')
                                            bank_temp.start_time.append(FI_last_end_time + FI_some_duration) 
                                            bank_temp.end_time.append(FI_last_end_time + FI_some_duration + weight_bank_interrupted)
                                            
                        #last_end_time = bank.end_time[-1] if bank.end_time else 0
                        bank.start_time.append(last_end_time)  # 设置当前的start_time
                        weight_average_allocate = self.Parameter[layer_id]/len(tiles_id)/len(weight_bank_allocate)
                        #self.Parameter = [3,1,1]
                        weight_some_duration = weight_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_write_cycle + weight_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_read_cycle 
                        # 凑上去
                        bank.end_time.append(last_end_time + weight_some_duration)  # 根据某个持续时间设置end_time'''
        
        FI_end_time_Compute_start_time = []
        FO_end_time_Noc_start_time = [] #这里 不太对了 #TODO：这里应该更改，搞成所有bank的max
        for layer_id in range(len(tiles_id)):
            # 可以有的tile准备好就先计算吗？如果可以的话计算时间需要tile 1-144：bank_max_time 
            # 但是给的bank都一样，所以这个layer1 tile[5,7]不同tile情况是一样的，但具体bank0和bank1是不一样的，
            # 下面的计算是按照不同bank的 更正：不太行，因为weight可能分到的bank是[0,1]。FI分到的bank是[0,2,3]重叠不上了，
            # 所以计算和bank没啥关系，所以还是按照所有bank的总计bank_max_time 【每个tile开始的时间会不一样】
            
            #print(f"Layer {layer_id}: Before assignment")
            
            for j in range(len(tiles_id[layer_id])):
                
                for FIFO_index in range(len(FIFO_bank_allocate[layer_id])):
                    FI_end_time_Compute_start_time.clear()
                    # 开始FI
                    if layer_id == 0:
                        Noc_latency.Noc_start_time[0] = Noc_latency.Noc_end_time[0] = 0
                        FO_end_time_Noc_start_time.append(0) #！！！
                    
                    FI_last_end_time = max(0, Noc_latency.Noc_end_time[layer_id])
                    bank = tile_bank_vocabulary[tiles_id[layer_id][j]][FIFO_bank_allocate[layer_id][FIFO_index]]
                    current_tile = tiles_id[layer_id][j]
                    current_bank = FIFO_bank_allocate[layer_id][FIFO_index] #这种是不是都错了，layer_id不对，应该是tile序号
                    if current_bank in weight_bank_allocate[current_tile]:
                        # 计算插入索引
                        index = bisect.bisect_right(bank.start_time, FI_last_end_time)
                        
                        # 插入FI操作
                        bank.start_time.insert(index, FI_last_end_time)
                        bank.layer_id.insert(index, layer_id)
                        bank.sram_type.insert(index, 'FI')
                        
                        FI_average_allocate = self.FI[layer_id] / len(tiles_id[layer_id]) / len(FIFO_bank_allocate[layer_id])
                        FI_some_duration = self.FI[layer_id]#(FI_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_write_cycle +
                                            #FI_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_read_cycle)
                        bank.end_time.insert(index, FI_last_end_time + FI_some_duration)
                        
                        if index > 0:
                            # 处理被打断的weight
                            weight_bank_interrupted = bank.end_time[index - 1] - FI_last_end_time
                            bank.end_time[index - 1] = FI_last_end_time
                            layer_id_temp = bank.layer_id[index - 1] #太重要了
                            bank.layer_id.insert(index + 1, layer_id_temp)
                            bank.sram_type.insert(index + 1, 'weight')
                            bank.start_time.insert(index + 1, FI_last_end_time + FI_some_duration)
                            bank.end_time.insert(index + 1, FI_last_end_time + FI_some_duration + weight_bank_interrupted)
                            
                            # 后面的weight时间往后推
                            if index + 2 < len(bank.start_time):
                                time_increase = bank.end_time[index + 1] - bank.start_time[index + 2]
                                bank.start_time[index + 2:] = [time + time_increase for time in bank.start_time[index + 2:]]
                        FI_end_time_Compute_start_time.append(bank.end_time[index])
                    else:
                        bank.start_time.append(FI_last_end_time)
                        bank.layer_id.append(layer_id)
                        bank.sram_type.append('FI')
                        bank.end_time.append(FI_last_end_time + FI_some_duration)
                        FI_end_time_Compute_start_time.append(bank.end_time[-1])
                FI_ready_time = max(FO_end_time_Noc_start_time)

                for FIFO_index in range(len(FIFO_bank_allocate[layer_id])):
                    bank = tile_bank_vocabulary[tiles_id[layer_id][j]][FIFO_bank_allocate[layer_id][FIFO_index]]
                    FO_end_time_Noc_start_time.clear() #全局变量和局部变量的问题
                    # 开始FO
                    weight_ready_time_origin = [] #忘记FIFO和weight不是相同的bank了
                    # 权重准备好的时间应该是这个tile 所有bank的weight都被ko了，需要统计weight具体的值了 后面没有满足此条件的了
                    for Weight_index in range(len(weight_bank_allocate[layer_id])):
                        bank_weight = tile_bank_vocabulary[tiles_id[layer_id][j]][weight_bank_allocate[layer_id][Weight_index]]
                        weight_ready_time_temp = [bank_weight.end_time[id] for id in range(len(bank_weight.layer_id))
                                                if bank_weight.layer_id[id] == layer_id and bank_weight.sram_type[id] == 'weight']#丢掉[id]了#TODO:可以优化，只需要查找到layer_id+1就停
                        weight_ready_time_origin.extend(weight_ready_time_temp) 
                        
                    weight_ready_time = max(weight_ready_time_origin) #这里是不是应该看所有bank一起的last_time【等这个layer对应的所有bank的weight数据都传完再计算，还是分开处理呢】
                    #tile的名字 #TODO：layer_id对应的地方的位置 TODO:这里还需要改变，layer_id.index(layer_id)可能不止一个如果分块计算
                    index_temp = Compute_latency[tiles_id[layer_id][j]].layer_id.index(layer_id)
                    Compute_latency[tiles_id[layer_id][j]].Compute_start_time[index_temp] = max(FI_ready_time,weight_ready_time)#max(bank.end_time[index],) #FI结束时间，weightreadyok时间
                    Compute_latency[tiles_id[layer_id][j]].update_Compute_end_time(index_temp,Compute_latency[tiles_id[layer_id][j]].Compute_start_time[index_temp])
                    FO_last_end_time = max(0, Compute_latency[tiles_id[layer_id][j]].Compute_end_time[index_temp])
                    
                    
                    # 计算插入索引 这个bank不对 是fifo的 我需要看weight有没有被挡住
                   
                    # 判断是不是在weight的bank范围内如果不是跳过去
                    current_tile = tiles_id[layer_id][j]
                    current_bank = FIFO_bank_allocate[layer_id][FIFO_index] #这种是不是都错了，layer_id不对，应该是tile序号
                    if current_bank in weight_bank_allocate[current_tile]:
                        index = bisect.bisect_right(bank.start_time, FO_last_end_time)
                    # if bank.sram_type[index-1] == 'weight' and bank.layer_id[index-1] == layer_id: #少加索引
                    #     FO_last_end_time = bank.end_time[index-1] #先存
                    #     index = index + 1
                        
                    # 插入FO操作
                        bank.start_time.insert(index, FO_last_end_time)
                        bank.layer_id.insert(index, layer_id)
                        bank.sram_type.insert(index, 'FO')
                    
                        FO_average_allocate = self.FO[layer_id] / len(tiles_id[layer_id]) / len(FIFO_bank_allocate[layer_id])
                        FO_some_duration = self.FO[layer_id]#(FO_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_write_cycle +
                                            #FO_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_read_cycle)
                        bank.end_time.insert(index, FO_last_end_time + FO_some_duration)
                        
                        if index > 0:
                            # 处理被打断的weight
                            weight_bank_interrupted = bank.end_time[index - 1] - FO_last_end_time
                            bank.end_time[index - 1] = FO_last_end_time
                            layer_id_temp = bank.layer_id[index - 1] #太重要了
                            bank.layer_id.insert(index + 1, layer_id_temp)
                            bank.sram_type.insert(index + 1, 'weight')
                            bank.start_time.insert(index + 1, FO_last_end_time + FO_some_duration)
                            bank.end_time.insert(index + 1, FO_last_end_time + FO_some_duration + weight_bank_interrupted)
                            
                            # weight_bank_interrupted = bank.end_time[index - 1] - FO_last_end_time
                            # bank.end_time[index - 1] = FO_last_end_time
                            # bank.layer_id.insert(index + 1, layer_id)
                            # bank.sram_type.insert(index + 1, 'weight')
                            # bank.start_time.insert(index + 1, FO_last_end_time + FO_some_duration)
                            # bank.end_time.insert(index + 1, FO_last_end_time + FO_some_duration + weight_bank_interrupted)
                            
                            # 后面的weight时间往后推
                            if index + 2 < len(bank.start_time):
                                time_increase = bank.end_time[index + 1] - bank.start_time[index + 2]
                                bank.start_time[index + 2:] = [time + time_increase for time in bank.start_time[index + 2:]]
                        
                        FO_end_time_Noc_start_time.append(bank.end_time[index])  
                    else:
                        bank.start_time.append(FO_last_end_time)
                        bank.layer_id.append(layer_id)
                        bank.sram_type.append('FO')
                        bank.end_time.append(FO_last_end_time + FO_some_duration)
                        FO_end_time_Noc_start_time.append(bank.end_time[-1])

                    #print(f"Layer {layer_id}, Tile {tiles_id[layer_id][j]}, Bank {weight_bank_allocate[layer_id][k]} assigned with start time {bank.start_time[-1]} and end time {bank.end_time[-1]}")
                if layer_id < len(tiles_id) - 1:  
                    Noc_latency.Noc_start_time[layer_id + 1] = max(FO_end_time_Noc_start_time)#bank.end_time[index] #FO结束时间，之后是tile com
                    Noc_latency.Noc_end_time[layer_id + 1] = max(FO_end_time_Noc_start_time) + Noc_latency.Noc_duration[layer_id + 1]#bank.end_time[index] + Noc_latency.Noc_duration[layer_id + 1]
            #print(f"Layer {layer_id}: After assignment")
        # # 开始FIFO
        # for layer_id in range(len(tiles_id)):  # 遍历每一层
        #     for j in range(len(tiles_id[layer_id])):
        #         for FIFO_index in range(len(FIFO_bank_allocate[layer_id])):

        #             #开始FI

        #             FI_last_end_time = max(0, Noc_latency.Noc_end_time[layer_id])
        #             bank = tile_bank_vocabulary[tiles_id[layer_id][j]][FIFO_bank_allocate[layer_id][FIFO_index]]
        #             # 通过列表找到
        #             index = bisect.bisect_right(bank.start_time, FI_last_end_time)
        #             # 在计算出的索引处插入 x
        #             bank.start_time.insert(index, FI_last_end_time)
        #             bank.layer_id.insert(index, layer_id)
        #             bank.sram_type.insert(index, 'FI')
        #             FI_average_allocate = self.FI[layer_id]/len(tiles_id[layer_id])/len(FIFO_bank_allocate[layer_id])
        #             FI_some_duration = FI_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_write_cycle + FI_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_read_cycle 
        #             bank.end_time.append(FI_last_end_time + FI_some_duration)
        #             # 处理被打断的weight
        #             weight_bank_interrupted = bank.end_time[index-1] - FI_last_end_time
        #             bank.end_time[index-1] = FI_last_end_time
        #             bank.layer_id.insert(index+1,layer_id)
        #             bank.sram_type.insert(index+1,'weight')
        #             bank.start_time.insert(index+1,FI_last_end_time + FI_some_duration) 
        #             bank.end_time.insert(index+1,FI_last_end_time + FI_some_duration + weight_bank_interrupted)
        #             # 后面的weight时间往后推
        #             time_increase = bank.end_time[index+1] - bank.start_time[index + 2]
        #             bank.start_time[index+2:] = [time + time_increase for time in bank.start_time[index+2:]]
        
        #             # 开始FO
        #             FO_last_end_time = max(0, Compute_latency.Compute_end_time[layer_id])
        #             bank = tile_bank_vocabulary[tiles_id[layer_id][j]][FIFO_bank_allocate[layer_id][FIFO_index]]
        #             # 通过列表找到
        #             index = bisect.bisect_right(bank.start_time, FO_last_end_time)
        #             # 在计算出的索引处插入 x
        #             bank.start_time.insert(index, FO_last_end_time)
        #             bank.layer_id.insert(index, layer_id)
        #             bank.sram_type.insert(index, 'FO')
        #             FO_average_allocate = self.FO[layer_id]/len(tiles_id[layer_id])/len(FIFO_bank_allocate[layer_id])
        #             FO_some_duration = FO_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_write_cycle + FO_average_allocate * 1024 * 1024 / 64 * self.SRAM_block_read_cycle 
        #             bank.end_time.append(FO_last_end_time + FO_some_duration)
        #             # 处理被打断的weight
        #             weight_bank_interrupted = bank.end_time[index-1] - FO_last_end_time
        #             bank.end_time[index-1] = FO_last_end_time
        #             bank.layer_id.insert(index+1,layer_id)
        #             bank.sram_type.insert(index+1,'weight')
        #             bank.start_time.insert(index+1,FO_last_end_time + FO_some_duration) 
        #             bank.end_time.insert(index+1,FO_last_end_time + FO_some_duration + weight_bank_interrupted)
        #             # 后面的weight时间往后推
        #             time_increase = bank.end_time[index+1] - bank.start_time[index + 2]
        #             bank.start_time[index+2:] = [time + time_increase for time in bank.start_time[index+2:]]
        print(f"Add FIFO")
        for layer_id in range(len(tiles_id)):  # 遍历每一层
            print(f"Layer {layer_id}: Before assignment")
            #for tile_id in tiles_id[i]:
                #print(f"Tile {tile_id}: {[vars(bank) for bank in tile_bank_vocabulary[tile_id]]}")
            
            for j in range(len(tiles_id[layer_id])):  # 遍历当前层中的每个tile
                for k in range(len(weight_bank_allocate[layer_id])):  # 遍历分配的bank
                    bank = tile_bank_vocabulary[tiles_id[layer_id][j]][weight_bank_allocate[layer_id][k]]
                    print(f"Layer {layer_id}, Tile {tiles_id[layer_id][j]}, Bank {weight_bank_allocate[layer_id][k]} assigned with {bank.sram_type[-1]} start time {bank.start_time[-1]} and end time {bank.end_time[-1]}")
                for k in range(len(FIFO_bank_allocate[layer_id])):  # 遍历分配的bank
                    bank = tile_bank_vocabulary[tiles_id[layer_id][j]][FIFO_bank_allocate[layer_id][k]]

                    print(f"Layer {layer_id}, Tile {tiles_id[layer_id][j]}, Bank {FIFO_bank_allocate[layer_id][k]} assigned with {bank.sram_type[-1]} start time {bank.start_time[-1]} and end time {bank.end_time[-1]}")

            print(f"Layer {layer_id}: After assignment")
        return tile_bank_vocabulary
   
    '''
        # def SRAM_latency(self, bank_num, weight_bank_allocate: List[List[int]], tiles_nums: List[int], current_time: int):
        #     
        #     weight_bank_allocate: 分配的bank列表
        #     tiles_nums: 每一层所用的tile编号
        #     current_time: 当前时间戳
        #     
        #     tile_bank_vocabulary = self.tile_num_to_bank(bank_num)  # 每个tile会有特定的bank数量
        #     tiles_id = self.tile_number_to_id(tiles_nums)  # 根据coordinate对应的tile ID
        #     tiles_id=[[0,1],[1],[0,1,2]]
        #     for i in range(len(tiles_id)):  # 遍历每一层
        #         print(f"Layer {i}: Before assignment")
        #         for tile_id in tiles_id[i]:
        #             print(f"Tile {tile_id}: {[vars(bank) for bank in tile_bank_vocabulary[tile_id]]}")
                
        #         for j in range(len(tiles_id[i])):  # 遍历当前层中的每个tile
        #             for k in range(len(weight_bank_allocate[i])):  # 遍历分配的bank
        #                 bank = tile_bank_vocabulary[tiles_id[i][j]][weight_bank_allocate[i][k]]
        #                 some_duration = 2
        #                 if not bank.status:  # 如果当前status是False
        #                     bank.status = True
        #                     bank.layer.append(i)
        #                     bank.sram_type.append('weight')
        #                     bank.start_time.append(current_time)  # 设置当前的start_time
        #                     bank.end_time.append(current_time + some_duration)  # 根据某个持续时间设置end_time
        #                 else:  # 如果当前status是True
        #                     # 需要等待，直到当前bank被释放
        #                     while bank.end_time[-1] > current_time:
        #                         current_time += 1  # 增加当前时间，模拟等待
        #                     bank.status = True
        #                     bank.layer.append(i)
        #                     bank.sram_type.append('weight')
        #                     bank.start_time.append(current_time)
        #                     bank.end_time.append(current_time + some_duration)

        #                 # 在end_time之后设置status为False
        #                 bank.status = False
        #                 print(f"Layer {i}, Tile {tiles_id[i][j]}, Bank {k} assigned with start time {bank.start_time[-1]} and end time {bank.end_time[-1]}")

        #         print(f"Layer {i}: After assignment")
        #         for tile_id in tiles_id[i]:
        #             print(f"Tile {tile_id}: {[vars(bank) for bank in tile_bank_vocabulary[tile_id]]}")

        #     return tile_bank_vocabulary

        # def SRAM_latency(self,bank_num,weight_bank_allocate:List[List[int]],tiles_nums:List[int]):

        #     {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2], 4: [0, 1, 2], 5: [0, 1, 2], 6: [0, 1, 2], 7: [0, 1, 2], 8: [0, 1, 2], 9: [0, 1, 2], 10: [0, 1, 2], 11: [0, 1, 2], 12: [0, 1, 2], 13: [0, 1, 2], 14: [0, 1, 2], 15: [0, 1, 2], 16: [0, 1, 2], 17: [0, 1, 2], 18: [0, 1, 2], 19: [0, 1, 2], 20: [0, 1, 2], 21: [0, 1, 2], 22: [0, 1, 2], 23: [0, 1, 2], 24: [0, 1, 2], 25: [0, 1, 2], 26: [0, 1, 2], 27: [0, 1, 2], 28: [0, 1, 2], 29: [0, 1, 2], 30: [0, 1, 2], 31: [0, 1, 2], 32: [0, 1, 2], 33: [0, 1, 2], 34: [0, 1, 2], 35: [0, 1, 2], 36: [0, 1, 2], 37: [0, 1, 2], 38: [0, 1, 2], 39: [0, 1, 2], 40: [0, 1, 2], 41: [0, 1, 2], 42: [0, 1, 2], 43: [0, 1, 2], 44: [0, 1, 2], 45: [0, 1, 2], 46: [0, 1, 2], 47: [0, 1, 2], 48: [0, 1, 2], 49: [0, 1, 2], 50: [0, 1, 2], 51: [0, 1, 2], 52: [0, 1, 2], 53: [0, 1, 2], 54: [0, 1, 2], 55: [0, 1, 2], 56: [0, 1, 2], 57: [0, 1, 2], 58: [0, 1, 2], 59: [0, 1, 2], 60: [0, 1, 2], 61: [0, 1, 2], 62: [0, 1, 2], 63: [0, 1, 2], 64: [0, 1, 2], 65: [0, 1, 2], 66: [0, 1, 2], 67: [0, 1, 2], 68: [0, 1, 2], 69: [0, 1, 2], 70: [0, 1, 2], 71: [0, 1, 2], 72: [0, 1, 2], 73: [0, 1, 2], 74: [0, 1, 2], 75: [0, 1, 2], 76: [0, 1, 2], 77: [0, 1, 2], 78: [0, 1, 2], 79: [0, 1, 2], 80: [0, 1, 2], 81: [0, 1, 2], 82: [0, 1, 2], 83: [0, 1, 2], 84: [0, 1, 2], 85: [0, 1, 2], 86: [0, 1, 2], 87: [0, 1, 2], 88: [0, 1, 2], 89: [0, 1, 2], 90: [0, 1, 2], 91: [0, 1, 2], 92: [0, 1, 2], 93: [0, 1, 2], 94: [0, 1, 2], 95: [0, 1, 2], 96: [0, 1, 2], 97: [0, 1, 2], 98: [0, 1, 2], 99: [0, 1, 2], 100: [0, 1, 2], 101: [0, 1, 2], 102: [0, 1, 2], 103: [0, 1, 2], 104: [0, 1, 2], 105: [0, 1, 2], 106: [0, 1, 2], 107: [0, 1, 2], 108: [0, 1, 2], 109: [0, 1, 2], 110: [0, 1, 2], 111: [0, 1, 2], 112: [0, 1, 2], 113: [0, 1, 2], 114: [0, 1, 2], 115: [0, 1, 2], 116: [0, 1, 2], 117: [0, 1, 2], 118: [0, 1, 2], 119: [0, 1, 2], 120: [0, 1, 2], 121: [0, 1, 2], 122: [0, 1, 2], 123: [0, 1, 2], 124: [0, 1, 2], 125: [0, 1, 2], 126: [0, 1, 2], 127: [0, 1, 2], 128: [0, 1, 2], 129: [0, 1, 2], 130: [0, 1, 2], 131: [0, 1, 2], 132: [0, 1, 2], 133: [0, 1, 2], 134: [0, 1, 2], 135: [0, 1, 2], 136: [0, 1, 2], 137: [0, 1, 2], 138: [0, 1, 2], 139: [0, 1, 2], 140: [0, 1, 2], 141: [0, 1, 2], 142: [0, 1, 2], 143: [0, 1, 2]}
        #     tile_bank_vocabulary = self.tile_num_to_bank(bank_num,)# 每个人会有特定的bank数量
        #     tiles_id = self.tile_number_to_id(tiles_nums) # 根据coordinate对应过来的id
        #     tiles_id=[[0,1],[1],[0,1,2]]
        #     # 默认顺序依赖
        #     for i in range(len(tiles_id)): #layer_id
        #         #layer i tiles_id[i]=[2,3,4] weight_bank_allocate[i]=[bank_分配]=[0,1]
        #         for j in range(len(tiles_id[i])):
        #             for k in range(len(weight_bank_allocate[i])):
        #                 current_tile_id = tiles_id[i][j]
        #                 current_bank_id = weight_bank_allocate[i][k]
        #                 current_bank = tile_bank_vocabulary[current_tile_id][current_bank_id]

        #                 # Check the last status of the current bank
        #                 if len(current_bank.status) == 0 or not current_bank.status[-1]:
        #                     current_bank.status.append(True)
        #                     current_bank.layer_id.append(i)
        #                     current_bank.sram_type.append('weight')
        #                     if len(current_bank.start_time) == 0:
        #                         temp = 0  # Assume initial start time is 0 if no previous time
        #                     else:
        #                         temp = current_bank.start_time[-1]
        #                     current_bank.start_time.append(temp)
        #                 else:
        #                     current_bank.status.append(True)
        #                     current_bank.layer_id.append(i)
        #                     current_bank.sram_type.append('weight')
        #                     temp = current_bank.end_time[-1] if len(current_bank.end_time) > 0 else 0
        #                     current_bank.start_time.append(temp)
        #                 # Append a placeholder end_time (you may update this based on your logic)
        #                 current_bank.end_time.append(temp + 1)  # Assuming a default duration of 1 unit

        #                 # if tile_bank_vocabulary[i][j][weight_bank_allocate[k]].status[最后一个] is not True:
        #                 #     tile_bank_vocabulary[i][j][weight_bank_allocate[k]].status.append(True)
        #                 #     tile_bank_vocabulary[i][j][weight_bank_allocate[k]].layer.append(i)
        #                 #     tile_bank_vocabulary[i][j][weight_bank_allocate[k]].sram_type.append('weight')
        #                 #     temp = tile_bank_vocabulary[i][j][weight_bank_allocate[k]].start_time[最后一个]
        #                 #     tile_bank_vocabulary[i][j][weight_bank_allocate[k]].start_time.append(temp)
        #                 # else:
        #                 #     tile_bank_vocabulary[i][j][weight_bank_allocate[k]].status.append(True)
        #                 #     tile_bank_vocabulary[i][j][weight_bank_allocate[k]].layer.append(i)
        #                 #     tile_bank_vocabulary[i][j][weight_bank_allocate[k]].sram_type.append('weight')
        #                 #     temp = tile_bank_vocabulary[i][j][weight_bank_allocate[k]].end_time[最后一个]
        #                 #     tile_bank_vocabulary[i][j][weight_bank_allocate[k]].start_time.append(temp)
        #     return tile_bank_vocabulary
    '''
    def tile_num_to_bank(self,bank_num,grid=12):
        tile_bank_vocabulary = {}
        for i in range(grid**2):
            tile_bank_vocabulary[i] = [Bank(bank_id=i) for i in range(bank_num)]#list(range(bank_num))
        return tile_bank_vocabulary
    def tile_compute_vocabulary(self,tiles_id,compute_duration,grid=12):
        tiles_id = [[0,1],[1],[0,1,2]]
        #TODO：提取函数写一下
        layer_id = [[0,2],[0,1,2],[2]] #第i个tile有第几个layer在计算
        compute_duration_update = []
        for i in range(len(layer_id)):
            compute_duration_update.append([compute_duration[j] for j in layer_id[i]])
        
        tile_compute_vocabulary = {}
        # TODO:这里duration需要具体计算
        for i in range(3):
            tile_compute_vocabulary[i] = Compute_unit_latency(layer_id[i],compute_duration_update[i])
        return tile_compute_vocabulary
    def tile_number_to_id(self,tiles_nums:List[int]):
        layer_coordinates = self.map_tiles_to_grid(tiles_nums,)
        tiles_id = [self.coordinates_to_tile_number(layer, self.tile_mapping) for layer in layer_coordinates]
        return tiles_id

    def latency(self, flops:List[float], weight:List[float],tile_nums: List[int]): #y都是MB的即flops,
        tile_nums = [None] * len(flops)
        compute_latency = [None] * len(flops)
        compute_latency = [flops[i] * 144 / tile_nums[i] for i in range(len(flops))]
        # compute_latency[0] = flops[0] * 144 / tile_num[0]  # qkv m
        # compute_latency[3] = flops[4] * 144 / tile_num[3]  # f1
        # compute_latency[2] = flops[3] * 144 / tile_num[2]  # p
        # compute_latency[1] = flops[1] * 144 / tile_num[1]  # m1
        # 2 * weight_sram = tile_num - self.FIFO[this layer input] - self.FIFO[this layer output] - self.KV #double buffer 用满片上所有 其他扔给dram
        # 预留出来
        # weight_dram = weight - weight_sram 
        # 不需要考虑
        # 假如【layer1 和 layer2 全部串行且依赖，先后用同样的TILE】
        # 假如【layer1 和 layer2 全部串行且依赖，先后用同样的TILE】
        # 假如【layer1 和 layer2 全部串行且依赖，先后用同样的TILE】

        #TODO：还没考虑tile_num前后分配不同的时候，doublebuffer衔接的问题

        # 假如【layer1 和 layer2 先后用同样的TILE】double buffering 主要是在同一个tile内部用于提升计算和访存的并行性 依赖不依赖不重要，因为这个是weight哦
        # 循环加起来，最后剩一个
        # if weight_dram=0 则简单如下面else
        # else:
            # for 所有layer
                # 如果weight_dram =< weight_sram
                # latency += max(buffer2PEandPEcompute_current_layer, Dram2buffer_next_layer) + buffer2PEandPEcompute_next_layer 
                # 如果weight_dram > weight_sram
                # repeat_num = (weight_dram / weight_sram)向下取整-1
                # latency += max(buffer2PEandPEcompute_current_layer, Dramwhichisweightsram2buffer_next_layer) + repeat_num * max(buffer2PEandPEcompute_next_layer, Dramwhichisweightsram2buffer_next_layer) + (weight_dram - repeat_num-1)的buffer2PEandPEcompute
        #没有double buffer 不考虑计算访存并行即能存在buffer就放在bufer，存不下才扔到DRAM去【比如一个块在计算的时候，另一个块可以提前把权重从dram读出来。】
        # 权重准备按理说不受计算的影响
        weight_ready_latency = [None] * len(weight)
        
        weight_ready_latency =[ ((weight[i] - tile_nums[i]) / self.DRAM_bandwidth * self.frequency + (weight[i] - tile_nums[i]) / self.SRAM_block_write_cycle + weight[i] * 1024 * 1024 / 64 * self.SRAM_block_read_cycle 
                                 if weight[i] > tile_nums[i] else weight[i]  * 1024 * 1024 / 64 * self.SRAM_block_read_cycle) for i in range(len(weight))]  # qkv

        # weight_ready_latency[0] = ((weight[0] - tile_num[0]) / self.DRAM_bandwidth * self.frequency + (weight[0] - tile_num[0]) / self.SRAM_block_write_cycle + weight[0] * 1024 * 1024 / 64 * self.SRAM_block_read_cycle if weight[0] > tile_num[0] else weight[0] / self.SRAM_block_read_cycle)  # qkv
        # weight_ready_latency[3] = ((weight[4] - tile_num[3]) / self.DRAM_bandwidth + (weight[4] - tile_num[3]) / self.SRAM_block_write_cycle + weight[4] / self.SRAM_block_read_cycle if weight[4] > tile_num[0] else weight[4] / self.SRAM_block_read_cycle)  # qkv
        # weight_ready_latency[2] = ((weight[3] - tile_num[2]) / self.DRAM_bandwidth + (weight[3] - tile_num[2]) / self.SRAM_block_write_cycle + weight[3] / self.SRAM_block_read_cycle if weight[3] > tile_num[2] else weight[3] / self.SRAM_block_read_cycle)  # qkv
        # weight_ready_latency[1] = ((weight[1] - tile_num[1]) / self.DRAM_bandwidth + (weight[1] - tile_num[1]) / self.SRAM_block_write_cycle + weight[1] / self.SRAM_block_read_cycle if weight[1] > tile_num[1] else weight[1] / self.SRAM_block_read_cycle)  # qkv

        '''input 现有的时间

上一层计算时间 计算的同时完成weight 加载

还需要判断input是不是能在buffer里面 不然得送去dram【预留出来output了】

 我那个是所有计算完成的时间

input写入buffer 再从buffer传到下一个layer所在tile的时间
如果就在这儿 就不用传了 写入时间加上即可

权重准备时间和计算时间重叠进行的


上一层输出结果会存进来的 注意了哦

max后计算开始是上一个地方start上它是现在的start'''

        # 没有double buffer 的存在
        output_write_latency = [self.FO[i] / 64 * self.SRAM_block_write_cycle for i in range(len(self.FO))]
        tiles2grid = self.map_tiles_to_grid(tile_nums,)
        comm_map = self.generate_comm_map(tiles2grid,self.tile_mapping)
        output_NoC_latency = []
        for i in range(len(tiles2grid)-1):
            output_NoC_latency.append(self.specific_tile_communication_time(tiles2grid[i], tiles2grid[i+1], comm_map[i], FO[i]))

        #total = val0 + 2 * max(val1, val2, val3)
        #return total, val0, val1, val2, val3
        

    def tile_number_to_coordinates(self,grid_size=12):
        """
        tile numbers[0,1,2...,143] to coordinates[(0,0)..,(11,11)] .
        """
        tile_mapping = {}
        tile_number = 0
        for row in range(grid_size):
            for col in range(grid_size):
                tile_mapping[tile_number] = (row, col)
                tile_number += 1
        return tile_mapping

    def coordinates_to_tile_number(self,coordinates:List[Tuple], tile_mapping:Dict[int,Tuple]):
        """
        coordinates to tile numbers.(id)
        """
        tile_number_mapping = {v: k for k, v in tile_mapping.items()}
        return [tile_number_mapping[coord] for coord in coordinates]

    def generate_comm_map(self,layer_coordinates:List[List[Tuple]], tile_mapping:Dict[int,Tuple]):
        """
        communication map for tiles across different layers.
        
        :param layer_coordinates: = [
            [(0, 0), (0, 1), (0, 2)],  # Layer 1 tiles
            [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # Layer 2 tiles
            [(1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]  # Layer 3 tiles
        ]
        :param tile_mapping: Dictionary mapping tile numbers 2 coordinates.
        :return: [{3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2], 10: [2], 11: [2], 12: [0], 13: [1], 14: [2], 15: [2], 16: [2]}, {0: [24, 25, 26, 27, 17], 1: [25, 24, 26, 27, 17], 2: [26, 25, 27, 17, 24]}] 
        Communication Map from Layer 0 to Layer 1:
    Tile 0 in Layer 0 communicates with Tiles [12, 13, 3, 14] in Layer 1
    Tile 1 in Layer 0 communicates with Tiles [13, 3, 12, 14] in Layer 1
    Tile 2 in Layer 0 communicates with Tiles [3, 14, 4, 13] in Layer 1
        """
        comm_maps = []

        # coordinates to tile numbers
        layer_tile_numbers = [self.coordinates_to_tile_number(layer, tile_mapping) for layer in layer_coordinates]

        for layer_index, current_layer_tiles in enumerate(layer_tile_numbers[:-1]):
            next_layer_tiles = layer_tile_numbers[layer_index + 1]
            comm_map = {}
            
            # Get coordinates for current and next layer tiles
            #[(0, 0), (0, 1), (0, 2)] 3
            current_layer_coords = [tile_mapping[tile] for tile in current_layer_tiles]
            #[(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)] 14
            next_layer_coords = [tile_mapping[tile] for tile in next_layer_tiles]

            for i, current_tile in enumerate(current_layer_coords):
                distances = []
                for j, next_tile in enumerate(next_layer_coords):
                    distance = self.manhattan_distance(current_tile[0],current_tile[1], next_tile[0],next_tile[1])
                    distances.append((distance, next_layer_tiles[j]))  # Store the tile number instead of index
                
                # 取最近的tile
                distances.sort()
                closest_tiles = [tile for _, tile in distances[:max(1, len(next_layer_tiles)//len(current_layer_tiles))]]
                comm_map[current_layer_tiles[i]] = closest_tiles
                # 分给最近的len(next_layer_tiles)//len(current_layer_tiles)
            comm_maps.append(comm_map)

        return comm_maps
        
    def map_tiles_to_grid(self, tile_nums:List[int], grid_size=12):
        """
        map 从左到右 从上到下，顺序排列
        tile_nums = [3, 14, 16]  #  Layer 1 has 3 tiles, Layer 2 has 14 tiles, Layer 3 has 16 tiles
        a=self.tile_number_to_coordinates()
        a
        :return: list [[(0, 0), (0, 1), (0, 2)], [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)], [(1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]]
        """
        all_positions = [(row, col) for row in range(grid_size) for col in range(grid_size)]
        used_positions = set()
        
        coordinates = [None] * len(tile_nums)
        index = 0

        for layer_index, tile_num in enumerate(tile_nums):
            if index + tile_num > len(all_positions):
                raise ValueError("Not enough space in the grid to allocate all tiles.")
            
            layer_positions = []
            
            for _ in range(tile_num):
                while all_positions[index] in used_positions:
                    index += 1
                    if index >= len(all_positions):
                        raise ValueError("Ran out of available positions.")
                
                position = all_positions[index]
                layer_positions.append(position)
                used_positions.add(position)
                index += 1
            
            coordinates[layer_index] = layer_positions
        
        return coordinates
    # 废弃
    # def map_tiles_to_grid(tile_num:List[int], grid_size=12):
    #     # Example: tile_num = [1,2,]
    #     coordinates = [None] * len(tile_num)
        

    #     # map方法：从左到右，从上到下
    #     for i in range(len(tile_num)):
    #         row = tile_num[i] // grid_size # 43//12= 3
    #         col = tile_num[i] % grid_size
    #         position = [(x, y) for x in range(row) for y in range(grid_size)]
    #         temp = [(row + 1, y) for y in range(col)]
    #         position.extend(temp)
    #         # Ensure position is not reused
    #         coordinates[i]=position
    #     return coordinates

   



    def manhattan_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def specific_tile_communication_time(self, src_tiles:List[Tuple], dest_tiles:List[Tuple], comm_map:Dict[int,List], total_data_size):
        
    #         # Example usage
    # # Layer 1 tiles (positions as tuples)
    # layer1_tiles = [(1, 2), (2, 3), (4, 5), (6, 7), (7, 8)]

    # # Layer 2 tiles (positions as tuples)
    # layer2_tiles = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), 
    #                 (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), 
    #                 (18, 19), (19, 20)]

    # # Define communication map
    # # Key: index of source tile in layer1_tiles, Value: list of indices of destination tiles in layer2_tiles
    # comm_map = {
    #     0: [0, 1],  # Tile 0 in Layer 1 communicates with Tile 0 and 1 in Layer 2
    #     1: [2],     # Tile 1 in Layer 1 communicates with Tile 2 in Layer 2
    #     2: [3, 4],  # Tile 2 in Layer 1 communicates with Tile 3 and 4 in Layer 2
    #     3: [5],     # Tile 3 in Layer 1 communicates with Tile 5 in Layer 2
    #     4: [6, 7]   # Tile 4 in Layer 1 communicates with Tile 6 and 7 in Layer 2
    # }
        # TODO: 全部平分处理。且并未考虑output的东西传到下一个layer的tile的buffer会放不下的情况。
        # 这里默认FIFO都能在buffer里面 条件要写一下
        max_comm_time = 0
        num_src_tiles = len(src_tiles)
        # Data size per source tile
        data_per_src_tile = total_data_size / num_src_tiles
        for src_index, dest_indices in comm_map.items():
            src_tile = self.tile_mapping[src_index]
            num_dest_tiles = len(dest_indices)
            data_per_src_tile_allocate_to_dest = data_per_src_tile / num_dest_tiles
            for dest_index in dest_indices:
                dest_tile = self.tile_mapping[dest_index]
                # Calculate Manhattan distance (hops)
                hops = self.manhattan_distance(src_tile[0], src_tile[1], dest_tile[0], dest_tile[1])
                print(hops)
                transmission_time = data_per_src_tile_allocate_to_dest / self.NoC_bandwidth # second
                # Calculate total communication time for this path
                comm_time = hops * transmission_time
                # Update max communication time
                if comm_time > max_comm_time:
                    max_comm_time = comm_time
        return max_comm_time


    def is_feasible(self, x):
        return x[0] + 2 * x[1] + x[2] + 2 * x[3] == 144

    def optimize(self):
        # Convert lists to numpy arrays
        FLOPs = np.array(self.FLOPs)
        Parameter = np.array(self.Parameter)
        KV = np.array(self.KV)
        FIFO = np.array(self.FIFO)

        compute_cycle = FLOPs / 147456  # Adjust based on actual formula

        for i in range(10):
            temp_compute = compute_cycle[i * 6:(i + 1) * 6]
            temp_weight = Parameter[i * 6:(i + 1) * 6]
            best_x = None
            best_value = float('inf')

            for m in self.m_range:
                for x in self.x_range:
                    for y in self.y_range:
                        for z in self.z_range:
                            tile_nums = [m,x,x,y,z,z]
                            if self.is_feasible([m, x, y, z]):
                                current_value = self.compute_latency(temp_compute, temp_weight,tile_nums) 
                                if current_value[0] < best_value:
                                    best_value = current_value[0]
                                    best_x = [m, x, y, z]
                                    self.time = current_value[1:5]

            print(f"stage:{i}")
            if best_x:
                print(f"Optimized result: m = {best_x[0]}, x = {best_x[1]}, y = {best_x[2]}, z = {best_x[3]}")
                print(f"Objective value: {best_value}")
                print(self.time)
            else:
                print("No feasible solution found.")

            self.write[i * 6:(i + 1) * 6 - 1] = [best_x[0], best_x[1], best_x[1], best_x[2], best_x[3], best_x[3]]

        self.write = self.write[:60]
        self.df['tile num'] = self.write
        self.df.to_excel(self.file_path, index=False)

        print(f"数据已保存到 {self.file_path}")
def main():
    a = Optimization('1',[1],[1],[1],[1],[10,15,8])
    tiles_nums=[2,1,3] #并行
    tiles2grid = a.map_tiles_to_grid(tiles_nums,)
    comm_map = a.generate_comm_map(tiles2grid,a.tile_mapping)
    a.Parameter = [12,9,18] #注意传参数
    a.FI = [16,20,10]
    a.FO = [10,15,8]
    output_NoC_latency = []
    for i in range(len(tiles2grid)-1):
        output_NoC_latency.append(a.specific_tile_communication_time(tiles2grid[i], tiles2grid[i+1], comm_map[i], a.FO[i]))
    print(output_NoC_latency)
    Noc = Noc_latency(output_NoC_latency)
    
    compute_duration = [15,7,20]
    tiles_id = [[0,1],[1],[0,1,2]]
    #Compute_latency_ex = Compute_latency(compute_duration)
    Compute_latency_ex = a.tile_compute_vocabulary(tiles_id,compute_duration)
    #layer 1
    #weight_bank_allocate=[[0,1],[1,2],[0,2]]
    weight_bank_allocate=[[0,1],[1,2],[0,2]]

    FIFO_bank_allocate=[[0,1,2],[0,2],[1,2]]
    tiles_id = [[0,1],[1],[0,1,2]]
    bank_num=3
    result=a.SRAM_latency(bank_num,Noc,Compute_latency_ex,weight_bank_allocate,FIFO_bank_allocate,tiles_nums)
    print("Noc_communication_time")
    
    for Layer_id in range(len(tiles_nums)):
        print(f"Layer{Layer_id} : Start Time{Noc.Noc_start_time[Layer_id]}, Duration{Noc.Noc_duration[Layer_id]}, End Time{Noc.Noc_end_time[Layer_id]}")
    print("Compute_time")
    for tile_id in range(len(Compute_latency_ex)):
        for Layer_id in range(len(Compute_latency_ex[tile_id].Compute_start_time)):
            print(f"Tile{tile_id},Layer{Compute_latency_ex[tile_id].layer_id[Layer_id]} : Start Time{Compute_latency_ex[tile_id].Compute_start_time[Layer_id]}, Duration{Compute_latency_ex[tile_id].Compute_duration[Layer_id]}, End Time{Compute_latency_ex[tile_id].Compute_end_time[Layer_id]}")
    bank_data = {}
    for i in range(3):
        for j in range(bank_num):
            print(f"Tile{i},Bank{j}:{result[i][j]}")
            bank_data[(i, j)] = {'layer_id': result[i][j].layer_id, 'sram_type': result[i][j].sram_type, 'start_time': result[i][j].start_time, 'end_time': result[i][j].end_time}
    print('showtime')
    print(bank_data)
    draw_pipeline(bank_num,tiles_id,bank_data,Compute_latency_ex,Noc)
    #print(a.tile_number_to_id(tiles_nums))
    #print(a.tile_num_to_bank(3))
    
   
if __name__ == "__main__":
    main()

# import numpy as np
# import pandas as pd
# # 定义目标函数：返回三个数中的最大值
# def objective(y,x):
#     val0 = y[0] * 144 / x[0] #qkv
#     val3 = y[4] * 144 / x[3] #f1
#     val2 = y[3] * 144 / x[2] #p
#     val1 = y[1] * 144 / x[1] #m1
#     return val0 + 8*max(val1, val2, val3)

# # 约束条件
# def is_feasible(x):
#     return x[0] + 2*x[1] + x[2] + 2*x[3] == 144

# # 设置初始解范围（可以根据需要调整范围）
# m_range = range(1, 80)
# x_range = range(1, 30)
# y_range = range(1, 40)
# z_range = range(1, 80)

# best_x = None
# best_value = float('inf')

# file_path = 'a_dict_data.xlsx'
# df = pd.read_excel(file_path)
# data=[]
# temp=[None]*6
# for _, row in df.iterrows():
#     cycle = row['Cycle']#16headbc
#     data.append(cycle)

# for i in range(10):
#     temp = data[i*6:(i+1)*6-1]
#     best_x = None
#     best_value = float('inf')
#     # 穷举所有可能的整数解
#     for m in m_range:
#         #print(m)
#         for x in x_range:
#             for y in y_range:
#                 for z in z_range:
#                     if is_feasible([m, x, y, z]):
#                         current_value = objective(temp,[m, x, y, z])
#                         if current_value < best_value:
#                             best_value = current_value
#                             best_x = [m, x, y, z]
#                             #print(best_value)

#     # 输出结果
#     print(f"stage:{i}")
#     if best_x:
#         print(f"Optimized result: m = {best_x[0]}, x = {best_x[1]}, y = {best_x[2]}, z = {best_x[3]}")
#         print(f"Objective value: {best_value}")
#     else:
#         print("No feasible solution found.")

# import numpy as np
# import pandas as pd
# # 定义目标函数：返回三个数中的最大值
# def compute_latency(y,x):
#     #计算时间
#     val0 = y[0] * 144 / x[0] #qkv
#     val3 = y[4] * 144 / x[3] #f1
#     val2 = y[3] * 144 / x[2] #p
#     val1 = y[1] * 144 / x[1] #m1
#     total = val0 + 2*max(val1, val2, val3)
#     return total,val0,val1,val2,val3

# def weight_ready_latency(y,x):
#     #权重准备时间

#     val0 = y[0] * 144 / x[0] #qkv
#     val3 = y[4] * 144 / x[3] #f1
#     val2 = y[3] * 144 / x[2] #p
#     val1 = y[1] * 144 / x[1] #m1
#     total = val0 + 2*max(val1, val2, val3)
#     return total,val0,val1,val2,val3

# # 约束条件
# def is_feasible(x):
#     return x[0] + 2*x[1] + x[2] + 2*x[3] == 144

# # 设置初始解范围（可以根据需要调整范围）
# m_range = range(1, 80)
# x_range = range(1, 30)
# y_range = range(1, 40)
# z_range = range(1, 80)

# best_x = None
# best_value = float('inf')

# file_path = 'a_dict_data.xlsx'
# df = pd.read_excel(file_path)
# data=[]
# write=[]
# time=[]
# temp=[None]*6
# write=[None]*60
# def core(FLOPs,Parameter,KV,FIFO):
#     #for _, row in df.iterrows():
#     compute_cycle = FLOPs/147456 #row['Cycle']#16headbc


#     #print(len(data))
#     for i in range(10):
#         temp = compute_cycle[i*6:(i+1)*6] #不包括索引
#         #print(temp)
#         best_x = None
#         best_value = float('inf')
#         # 穷举所有可能的整数解
#         for m in m_range:
#             #print(m)
#             for x in x_range:
#                 for y in y_range:
#                     for z in z_range:
#                         if is_feasible([m, x, y, z]):
#                             current_value = compute_latency(temp,[m, x, y, z]) + weight_ready_latency(temp,[m,x,y,z])
#                             if current_value[0] < best_value:
#                                 best_value = current_value[0]
#                                 best_x = [m, x, y, z]
#                                 time = current_value[1:5]
#                                 #print(best_value)

#         # 输出结果
#         print(f"stage:{i}")
#         if best_x:
#             print(f"Optimized result: m = {best_x[0]}, x = {best_x[1]}, y = {best_x[2]}, z = {best_x[3]}")
#             print(f"Objective value: {best_value}")
#             print(time)
#         else:
#             print("No feasible solution found.")
#         write[i*6:(i+1)*6-1]=[best_x[0],best_x[1],best_x[1],best_x[2],best_x[3],best_x[3]]


#     write=write[:60]
#     df['tile num'] = write
#     df.to_excel(file_path, index=False)

#     print(f"数据已保存到 {file_path}")