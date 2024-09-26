import numpy as np
from config import LayerMapConfig,LayerConfig
from hardware import Bank,Buffer,MFU,NoC_Router,Hardware,Tile
import math
from draw_pipeline import draw,draw_sample
class DataTensor:
    def __init__(self, name: str = None, shape: list = None, dtype: str = None, data_type: str = None):
        self.name = name
        self.shape = shape  # [1, 4096, 4096]
        self.dtype = dtype
class Data_Split_Tensor: # split后的小tensor
    def __init__(self, dtensor: DataTensor, data_range: list = None):
        self.dtensor = dtensor
        self.data_range = data_range
        self.shape = self.get_shape()  # [1, 4096, 4096]
        self.amount = math.prod(self.shape)
        self.type = None
        
    def get_shape(self,) -> list:
        return [ite[1] - ite[0] for ite in self.data_range]
    

class Compute_Layer():
    def __init__(self, hardware: Hardware,Layer_Map_config:LayerMapConfig) -> None:
        self.hardware = hardware 
        self.Layer_config = Layer_Map_config.LayerConfig
        self.op = None
        self.fusion_group_id = Layer_Map_config.group_id
        self.C = self.Layer_config.output[0]
    
        self.block_size = Layer_Map_config.block_size
        self.block_size_m = self.block_size[0]; self.block_size_k = self.block_size[1]; self.block_size_n = self.block_size[2] 
        self.tile_num = Layer_Map_config.tile_num
        self.tile_allocation = Layer_Map_config.tile_allocation
        self.execute()
    
    def execute(self,):
        if self.Layer_config.type == 'MM':
            self.op = 'MM'
            self._execute_linear_MM()
            
        elif self.Layer_config.type == 'attention_MM':
            self._execute_attention_MM()
    

    # 进行该layer的计算
    def _execute_linear_MM(self, ):

        assert len(self.Layer_config.input)==1
        if len(self.Layer_config.input) == 1:
            self.A = self.Layer_config.input[0]
            self.B = self.Layer_config.param['weight']
        
        # 给block分配tile
        AB_block_allocation,C_block_allocation = self._allocate_tiles_to_blocks()

        # load Ablock Bblock
        A_block,A_block_num = self._block_parition(self.A,[self.block_size_m,self.block_size_k])
        B_block,B_block_num = self._block_parition(self.B,[self.block_size_k,self.block_size_n])
        C_block,C_block_num = self._block_parition(self.C,[self.block_size_m,self.block_size_n])

        for i in range(A_block_num[0]):
            for j in range(A_block_num[1]):
                for l in range(B_block_num[1]):
                    # load 1个 tile
                    compute_AB_tile_id = AB_block_allocation[math.floor(j/self.block_size_n)][math.floor(l/self.block_size_k)][0]#tile 0 
                    tile_execute = self.hardware.Tile_groups[self.fusion_group_id][compute_AB_tile_id] #这里得到计算1个A_block*1个B_block的那1个tile
                    
                    # 这里需要继续分小小块计算,首先split得到合适的
                    # 计算最佳的分块维度(4,64,64)-->(1,4,32) 
                    # TODO：这里只考虑了compute_capacity还没有考虑bank_capacity的事情，需要重新写一下calculate_split,
                    # TODO:进一步得到FI_bank_num, Param_bank_num, FO_bank_num，Param_dram的值
                    parition_mini_shape = self._calculate_splits(A_block[i][j], B_block[j][l])
                    self._execute_block_mini(parition_mini_shape,A_block[i][j], B_block[j][l],C_block[i][l],tile_execute) 
                    # TODO：这里衔接处需要写一下pipe
                    #C_block +=  加一下这里，其他应该就是查一下tile_execute分的对不对
                
                    #在这里draw/draw_sample一下，即得到图
                    draw_sample(self.hardware,self.fusion_group_id,compute_AB_tile_id)

    def _execute_attention_MM(self,):
        assert len(self.Layer_config.input)==2 
        if len(self.Layer_config.input) == 2:
            self.A = self.Layer_config.input[0]
            self.B = self.Layer_config.input[1]

    
    def _execute_block_mini(self, parition_mini_shape,A_block: Data_Split_Tensor, B_block: Data_Split_Tensor,C_block: Data_Split_Tensor,tile_execute:Tile):
        
        rows_split, cols_split, cols_B_split = parition_mini_shape
        # get_ready
        A_sub_block,A_sub_block_num = self._block_parition(A_block,[rows_split, cols_split])
        B_sub_block,B_sub_block_num = self._block_parition(B_block,[cols_split, cols_B_split])
        C_sub_block,C_sub_block_num = self._block_parition(C_block,[rows_split, cols_B_split])
        
        # 遍历每个分块
        for i in range(A_sub_block_num[0]):
            for j in range(A_sub_block_num[1]):
                for l in range(B_sub_block_num[1]):
                    hbm = self.hardware.HBM # TODO:这里怎么写比较好
                    self.hardware.process(A_sub_block[i][j],B_sub_block[j][l],C_sub_block[i][l],tile_execute,hbm,self.op)
                    
       
    def _allocate_tiles_to_blocks_unit(self,tile_ids, num_blocks):
        # 初始化每个 block 的 tile 列表
        block_allocation = [[] for _ in range(num_blocks)]
        tile_allocation = [[] for _ in range(len(tile_ids))]  # List of lists for tiles
        tile_per_block = len(tile_ids)/num_blocks 
        # 分配 tile 到每个 block
        if  tile_per_block > 1:
            for idx, tile_id in enumerate(tile_ids):
                block_id = idx % num_blocks  # 计算当前 tile 应分配到哪个 block
                block_allocation[block_id].append(tile_id)
                tile_allocation[idx].append(block_id)
        else:
            for i in range(num_blocks):
                assigned_tile_idx = i % len(tile_ids)
                assigned_tile = tile_ids[assigned_tile_idx]
                block_allocation[i].append(assigned_tile)
                tile_allocation[assigned_tile_idx].append(i)
        
        return block_allocation
    
    def _allocate_tiles_to_blocks(self,):
        m, k = self.A.shape
        k, n = self.B.shape
        block_num_n = math.ceil(n / self.block_size_n) # 即算 一行有几个c_block 向上取整
        block_num_k = math.ceil(k / self.block_size_k) #算1个C_block的时候需要算几个A*B
        C_block_allocation = self._allocate_tiles_to_blocks_unit(self.tile_allocation,block_num_n)
        AB_block_allocation = []
        for tile_list in C_block_allocation:        # 这里是把计算1个C_block需要的所有A_block*B_block分到tile上
            AB_block_allocation.append(self._allocate_tiles_to_blocks_unit(tile_list,block_num_k)) #[[0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48]]
        return AB_block_allocation,C_block_allocation
        
    # 小小块怎么分
    def _calculate_splits(self, A_block:Data_Split_Tensor,B_block: Data_Split_Tensor):
        rows_A, cols_A, cols_B = A_block.shape[0], A_block.shape[1], B_block.shape[1]
        max_FLOPs = self.hardware.Tile_list[0].MFU.compute_capacity
        # 这里还需要写一下bank allocate的结果
        best_rows_split = 1
        best_cols_split = 1
        best_cols_B_split = 1
        
        min_splits = float('inf')
        
        # 尝试不同的分块维度
        for rows_split in range(1, rows_A + 1):
            for cols_split in range(1, cols_A + 1):
                for cols_B_split in range(1, cols_B + 1):
                    # 计算每个分块的FLOPs
                    block_FLOPs = 2 * rows_split * cols_split * cols_B_split
                    
                    # 确保每个分块的FLOPs不超过最大FLOPs
                    if block_FLOPs <= max_FLOPs:
                        # 计算所需的分块数
                        num_splits = (rows_A + rows_split - 1) // rows_split * \
                                    (cols_A + cols_split - 1) // cols_split * \
                                    (cols_B + cols_B_split - 1) // cols_B_split
                        
                        # 更新最佳分块维度
                        if num_splits < min_splits:
                            min_splits = num_splits
                            best_rows_split = rows_split
                            best_cols_split = cols_split
                            best_cols_B_split = cols_B_split

        split_result = [best_rows_split,best_cols_split,best_cols_B_split]         
        return split_result
   
    def _block_parition(self,dtensor: DataTensor,parition_shape:list):
        m, n = dtensor.shape
        block_size_m, block_size_n = parition_shape
        block_num_m, block_num_n = math.ceil(m / block_size_m), math.ceil(n / block_size_n) 
        result = [] ; result_shape = [block_num_m, block_num_n]
        # 感觉需要分开三个block的结果 例如silu是一个dtensor
        if len(dtensor.shape) == 2:
            for i in range(0, m, block_size_m):
                row = []
                for j in range(0, n, block_size_n):
                    row.append(Data_Split_Tensor(self.A, [[i,min(i + block_size_m,m)],[j,min(j + block_size_n,n)]]))
                result.append(row)
        return result,result_shape
    
   

