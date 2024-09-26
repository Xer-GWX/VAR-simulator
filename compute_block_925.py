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
    def __init__(self, Tile_configs: Hardware,Layer_Map_config:LayerMapConfig) -> None:
        self.Tile_configs = Tile_configs # 需要它可以全局改变
        self.Layer_config = Layer_Map_config.LayerConfig
        self.fusion_group_id = Layer_Map_config.group_id
        self.type = None
        self.C = self.Layer_config.output[0]
        assert len(self.Layer_config.input)==1 or len(self.Layer_config.input)==2 
        if len(self.Layer_config.input) == 2:
            self.A = self.Layer_config.input[0]
            self.B = self.Layer_config.input[1]
            self.type = 'II' # input*input
        elif len(self.Layer_config.input) == 1:
            self.A = self.Layer_config.input[0]
            self.B = self.Layer_config.param['weight']
            self.type = 'IP' # input*param
    
        self.block_size = Layer_Map_config.block_size
        self.block_size_m = self.block_size[0]; self.block_size_k = self.block_size[1]; self.block_size_n = self.block_size[2] 
        self.tile_num = Layer_Map_config.tile_num
        self.tile_allocation = Layer_Map_config.tile_allocation
        self.execute()
        
    def execute(self,):
        if self.Layer_config.type == 'MM':
            self._execute_linear_MM()
        
    
    def calculate_splits(self, rows_A, cols_A, cols_B):
        max_FLOPs = self.Tile_configs.Tile_list[0].MFU.compute_capacity
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
    
        return best_rows_split, best_cols_split, best_cols_B_split

    
    def compute_block(self, A_block: Data_Split_Tensor, B_block: Data_Split_Tensor,tile_execute:Tile):
        rows_A, cols_A = A_block.shape
        cols_B = B_block.shape[1]
        buffer = tile_execute.Buffer #应该传进来一个tile
        mfu = tile_execute.MFU
        # 计算最佳的分块维度(4,64,64)-->(1,4,32) 
        # TODO：这里只考虑了compute_capacity还没有考虑bank_capacity的事情，需要重新写一下calculate_split,
        # TODO:进一步得到FI_bank_num, Param_bank_num, FO_bank_num，Param_dram的值
        rows_split, cols_split, cols_B_split = self.calculate_splits(rows_A, cols_A, cols_B)
        
        FI_bank_num, Param_bank_num, FO_bank_num = 2,4,4 # 这里应该是上一行的函数给一个结果，这里先随便设的
        Param_dram = 4
        # 这里还得到了bank_num的数量
        buffer.FI_bank_num = FI_bank_num
        buffer.FO_bank_num = FO_bank_num
        buffer.Parameter_bank_num = Param_bank_num
        FI_bank = buffer.FI_bank = buffer.get_FI_bank_unit(); Param_bank = buffer.Param_bank = buffer.get_param_bank_unit(); FO_bank = buffer.FO_bank = buffer.get_FO_bank_unit()

        result = np.zeros((rows_A, cols_B))
        
        # 遍历每个分块
        for start_row in range(0, rows_A, rows_split):
            end_row = min(start_row + rows_split, rows_A)
            for start_col in range(0, cols_A, cols_split):
                end_col = min(start_col + cols_split, cols_A)
                for start_col_B in range(0, cols_B, cols_B_split):
                    end_col_B = min(start_col_B + cols_B_split, cols_B)
                    #A_sub_block_c = A_block[start_row:end_row, start_col:end_col]# TODO：这里是为了验证计算结果
                    #B_sub_block_c = B_block[start_col:end_col, start_col_B:end_col_B]
                    A_sub_block = Data_Split_Tensor(A_block.dtensor,[[start_row,end_row], [start_col,end_col]])#
                    B_sub_block = Data_Split_Tensor(B_block.dtensor,[[start_col,end_col], [start_col_B,end_col_B]])#
                    #result_block = 函数结果
                    # 计算当前分块的结果
                    #split_result_c = np.dot(A_sub_block_c, B_sub_block_c)
                    split_result = Data_Split_Tensor(self.C,[[start_row,end_row], [start_col_B,end_col_B]])
                    
                    
                    # Input*Parameter  #TODO:改成1个接口，后面写matmul还需要把Input*Input的情况写一下
                    for bank in FI_bank[buffer.FI_access_flag % 2]:
                        # 应该是另一组bank正在计算的结果
                        if len(mfu.end_time_list) == 0 or len(mfu.end_time_list) == 1: FI_ready_time = bank.read_data(0,A_sub_block)
                        else: FI_ready_time = bank.read_data(mfu.end_time_list[-2],A_sub_block)
                    buffer.FI_access_flag += 1 # 用于pingpong

                    hbm = self.Tile_configs.HBM
                    if len(mfu.end_time_list) == 0 or len(mfu.end_time_list) == 1: Param_load_time = hbm.load_data(0,Param_dram) # input 的 情况
                    else: Param_load_time = hbm.load_data(mfu.end_time_list[-2],Param_dram)
                    
                    for bank in Param_bank[buffer.Param_access_flag % 2]:
                        Param_pre_ready_time = bank.write_data(Param_load_time,B_sub_block) 
                        Param_ready_time = bank.read_data(Param_pre_ready_time,B_sub_block) 
                    buffer.Param_access_flag += 1 # 用于pingpong
                    
                    FO_compute_ready_time = mfu.process(A_sub_block,B_sub_block,FI_ready_time,Param_ready_time,'mat_mat_mul')

                    if len(mfu.end_time_list) == 0 or len(mfu.end_time_list) == 1: # input的情况
                        for bank in FO_bank[0]:
                            FO_write_ready_time = bank.write_data(FO_compute_ready_time,split_result)
                    else:    
                        for bank in FO_bank[0]:
                            FO_read_ready_time = bank.read_data(bank.last_access_end_time,split_result) # 这里应该是存储的那部分，介于shape一样就直接用的split_result
                        FO_add_ready_time = mfu.process(split_result,split_result,FO_read_ready_time,FO_compute_ready_time,'mat_mat_add')
                        
                        for bank in FO_bank[0]:
                            FO_write_ready_time = bank.write_data(FO_add_ready_time,split_result)
                    
                    # TODO: add没写，上面的mfu.last_access_end_time指的是非add的compute
                    # result[start_row:end_row, start_col_B:end_col_B] += split_result_c
        return result
       
    def allocate_tiles_to_blocks(self,tile_ids, num_blocks):
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
    
    # 进行该layer的计算
    def _execute_linear_MM(self, ):
        m, k = self.A.shape
        k, n = self.B.shape
        block_num_n = math.ceil(n / self.block_size_n) # 即算 一行有几个c_block 向上取整
        block_num_k = math.ceil(k / self.block_size_k) #算1个C_block的时候需要算几个A*B
        A = np.random.rand(m, k)
        B = np.random.rand(k, n)
        C = np.zeros((m, n))
        if self.Layer_config.name == 'qkv':
            #tile_q, tile_k, tile_v = [list(arr) for arr in np.array_split(self.tile_allocation, 3)]
            #block_num_n /= 3
            # 这里是把一行48个C_block分到72个tile上
            C_block_allocation = self.allocate_tiles_to_blocks(self.tile_allocation,block_num_n)
        
        AB_block_allocation = []
        for tile_list in C_block_allocation:
            # 这里是把计算1个C_block需要的所有A_block*B_block分到tile上
            AB_block_allocation.append(self.allocate_tiles_to_blocks(tile_list,block_num_k))
            #[[0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48]]
        
        for i in range(0, m, self.block_size_m):
            for j in range(0, n, self.block_size_n):
                C_block = np.zeros((self.block_size_m, self.block_size_n))
                for l in range(0, k, self.block_size_k):
                    compute_AB_tile_id = AB_block_allocation[math.floor(j/self.block_size_n)][math.floor(l/self.block_size_k)][0]#tile 0 
                    tile_execute = self.Tile_configs.Hardware[self.fusion_group_id][compute_AB_tile_id] #这里得到计算1个A_block*1个B_block的那1个tile
                    # load Ablock Bblock
                    A_block = Data_Split_Tensor(self.A, [[i,i + self.block_size_m],[l,l + self.block_size_k]])
                    B_block = Data_Split_Tensor(self.B, [[l,l + self.block_size_k],[j,j + self.block_size_n]])
                    # TODO:后面需要验证一下
                    # A_block = A[i:i + self.block_size_m, l:l + self.block_size_k] # A1 * B1 的情况 (4,64)
                    # B_block = B[l:l + self.block_size_k, j:j + self.block_size_n] # (64,64)
                    
                    # 这里需要继续分小小块计算
                    C_block += self.compute_block(A_block, B_block,tile_execute) 
                    

                    #在这里draw/draw_sample一下，即得到图
                    draw_sample(self.Tile_configs,self.fusion_group_id,compute_AB_tile_id)
                
        

   

