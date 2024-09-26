import numpy as np
from config import LayerMapConfig
from hardware import Bank,Buffer,MFU,NoC_Router,Hardware,Tile
import math
from draw_pipeline import draw
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
    

# 实现拆分FLOPs得到一个list，拆分parameter等等也是一个list，然后还需要计算分配bank_num 之后得到结果，
class Compute_block():
    def __init__(self, Tile_configs: Hardware,Layer_Map_config:LayerMapConfig) -> None:
        self.Tile_configs = Tile_configs # 需要它可以全局改变
        self.Layer_config = Layer_Map_config.LayerConfig
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
        
        #self.tiles_allocate_num = [[72,18,18,36],[72,72]]#[[1], [0, 1, 2]]
        #self.bank_num_max = 32
        #self.memory_capacity_per_bank = 32 #KB 
        #self.buffer_capacity = #32KB #[1024 * len(tile_id) for tile_id in self.tiles_id]
        #self.compute_PE_capacity = 256 #[256 * len(tile_id) for tile_id in self.tiles_id]
        self.block_size = Layer_Map_config.block_size
        self.block_size_m = self.block_size[0]; self.block_size_k = self.block_size[1]; self.block_size_n = self.block_size[2] 
        self.tile_num = Layer_Map_config.tile_num
        self.tile_allocation = Layer_Map_config.tile_allocation
        
        # self.bank_allocate_num = [] #每个layergroup里面分 即144是怎么分的 列表是[layergroupid][tile_]
        # self.parameter_DRAM = []
        # self.FLOPs = []
        # self.FI = []
        # self.FO = []
        # bank_num 它这个是具体的值是多少
        # 应该是class里面每个layer写一个
        self.generate()

    def generate(self):
        C,A,B = self.block_matrix_multiply()
        C_expected = np.dot(A, B)
        print(np.allclose(C, C_expected)) 
        #return self.FLOPs, self.parameter_buffer, self.parameter_DRAM, self.FI, self.FO

    def load_to_buffer(self, layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l):
        A_block = A[i:i + block_size_m, l:l + block_size_k]
        B_block = B[l:l + block_size_k, j:j + block_size_n]
        return A_block, B_block
    
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

    # 存分块的flops和memory吗
    def compute_block(self, A_block: Data_Split_Tensor, B_block: Data_Split_Tensor,tile_execute:Tile):
        rows_A, cols_A = A_block.shape
        cols_B = B_block.shape[1]
        buffer = tile_execute.Buffer #应该传进来一个tile
        mfu = tile_execute.MFU
        # 计算最佳的分块维度(4,64,64)-->(1,4,32) if
        rows_split, cols_split, cols_B_split = self.calculate_splits(rows_A, cols_A, cols_B)
        #print(rows_split, cols_split, cols_B_split)
        # 可以直接写time情况了
        # 初始化最终的 result 矩阵
        FI_bank_num, Param_bank_num, FO_bank_num = 2,4,4
        Param_dram = 4
        # 这里还得到了bank_num的数量
        buffer.FI_bank_num = FI_bank_num
        buffer.FO_bank_num = FO_bank_num
        buffer.Parameter_bank_num = Param_bank_num
        #FI_bank = buffer.get_FI_bank_unit(); Param_bank = buffer.get_param_bank_unit(); FO_bank = buffer.get_FO_bank_unit()
        FI_bank = buffer.FI_bank = buffer.get_FI_bank_unit(); Param_bank = buffer.Param_bank = buffer.get_param_bank_unit(); FO_bank = buffer.FO_bank = buffer.get_FO_bank_unit()

        # 所以目前的问题是bank的问题，但是bank内部应该改变啊
        result = np.zeros((rows_A, cols_B))
        
        # 遍历每个分块
        for start_row in range(0, rows_A, rows_split):
            end_row = min(start_row + rows_split, rows_A)
            for start_col in range(0, cols_A, cols_split):
                end_col = min(start_col + cols_split, cols_A)
                for start_col_B in range(0, cols_B, cols_B_split):
                    end_col_B = min(start_col_B + cols_B_split, cols_B)
                    
                    # 提取当前分块 按理说应该是dtensor的类型
                    #A_sub_block_c = A_block[start_row:end_row, start_col:end_col]
                    #B_sub_block_c = B_block[start_col:end_col, start_col_B:end_col_B]
                    A_sub_block = Data_Split_Tensor(A_block.dtensor,[[start_row,end_row], [start_col,end_col]])#
                    B_sub_block = Data_Split_Tensor(B_block.dtensor,[[start_col,end_col], [start_col_B,end_col_B]])#
                    #result_block = 函数结果
                    # 计算当前分块的结果
                    #split_result_c = np.dot(A_sub_block_c, B_sub_block_c)
                    split_result = Data_Split_Tensor(self.C,[[start_row,end_row], [start_col_B,end_col_B]])
                    #dtensor维度是A_sub_block 
                    # 这里用一次就记录一次
                    # TODO：需要记录一下上一个bank在计算的情况哈 直接看另一个bank组的compute_end_time 
                    # 所以compute里面需要记录compute的bank是谁，并且记录 这里的mfu.last_access_end_time指的是非add的compute，这里需要记录一下
                    #写入输入
                    for bank in FI_bank[buffer.FI_access_flag % 2]:
                        # 应该是另一组bank正在计算的结果
                        if len(mfu.end_time_list) == 0 or len(mfu.end_time_list) == 1: FI_ready_time = bank.read_data(0,A_sub_block)
                        else: FI_ready_time = bank.read_data(mfu.end_time_list[-2],A_sub_block)
                    buffer.FI_access_flag += 1 # 不知道能不能让全局的改变哈
                    
                    # I*P
                    hbm = self.Tile_configs.HBM
                    if len(mfu.end_time_list) == 0 or len(mfu.end_time_list) == 1: Param_load_time = hbm.load_data(0,Param_dram)
                    else: Param_load_time = hbm.load_data(mfu.end_time_list[-2],Param_dram)
                    
                    for bank in Param_bank[buffer.Param_access_flag % 2]:
                        # 应该是另一组bank正在计算的结果
                        Param_pre_ready_time = bank.write_data(Param_load_time,B_sub_block) 
                        Param_ready_time = bank.read_data(Param_pre_ready_time,B_sub_block) 
                    buffer.Param_access_flag += 1 # 不知道能不能让全局的改变哈
                    
                    mfu.process(A_sub_block,B_sub_block,FI_ready_time,Param_ready_time)

                    for bank in FO_bank[0]:
                        bank.write_data(mfu.last_access_end_time,split_result)
                    

                    # TODO:??fo  result_tensor
                    # TODO: add
                    # 确定拆分结果在最终 result 中的位置
                    #result[start_row:end_row, start_col_B:end_col_B] += split_result_c

                    # 记录每个拆分的 FLOPs、FI、FO 和参数
                    # split_FLOPs = A_sub_block.shape[0] * A_sub_block.shape[1] * B_sub_block.shape[1]
                    # self.FLOPs[layer_id].append(split_FLOPs)
                    
                    # split_FI = A_sub_block.shape[0] * A_sub_block.shape[1]
                    # split_FO = A_sub_block.shape[0] * B_sub_block.shape[1]
                    # self.FI[layer_id].append(split_FI)
                    # self.FO[layer_id].append(split_FO)
                    
                    # split_buffer_size = min(B_sub_block.shape[0] * B_sub_block.shape[1], (self.buffer_capacity[layer_id] - split_FI - split_FO) / 2)
                    # self.parameter_buffer[layer_id].append(split_buffer_size)
                    # split_dram_size = B_sub_block.shape[0] * B_sub_block.shape[1] - split_buffer_size
                    # self.parameter_DRAM[layer_id].append(split_dram_size if split_dram_size > 0 else 0)
                    #print("hi")
        return result

    def store_result(self, layer_id, C, C_block, i, j):
        # 计算切片的有效范围
        end_i = min(i + C_block.shape[0], C.shape[0])
        end_j = min(j + C_block.shape[1], C.shape[1]) #C_block会固定大小，所以需要切片，如果两个数组的维度不同，广播机制会自动将较小的数组扩展到较大数组的形状。
        
        # 调整 C_block 的切片大小
        slice_C_block = C_block[:end_i - i, :end_j - j]
        
        # 将 C_block 加到目标矩阵 C 中
        C[i:end_i, j:end_j] += slice_C_block
       
    def allocate_tiles_to_blocks(self,tile_ids, num_blocks):
        # 初始化每个 block 的 tile 列表
        block_allocation = [[] for _ in range(num_blocks)]
        tile_allocation = [[] for _ in range(len(tile_ids))]  # List of lists for tiles
        tile_per_block = len(tile_ids)/num_blocks #16block 24tile
        # 分配 tile 到每个 block
        if  tile_per_block > 1:
            for idx, tile_id in enumerate(tile_ids):
                block_id = idx % num_blocks  # 计算当前 tile 应分配到哪个 block
                block_allocation[block_id].append(tile_id)
                tile_allocation[idx].append(block_id)
        else:
            for i in range(num_blocks):
                # Assign tiles to blocks in a round-robin manner
                assigned_tile_idx = i % len(tile_ids)
                assigned_tile = tile_ids[assigned_tile_idx]
                block_allocation[i].append(assigned_tile)
                tile_allocation[assigned_tile_idx].append(i)
                
                # for tile_idx, blocks in enumerate(tile_allocation):
                #     print(f"Tile {tile_ids[tile_idx]}: {blocks}")
        
        return block_allocation
    def block_matrix_multiply(self, ):
        m, k = self.A.shape
        k, n = self.B.shape
        block_num_n = math.ceil(n / self.block_size_n) #= 16 # 16 block_num 一行有几个c_block 向上取整
        block_num_k = math.ceil(k / self.block_size_k) #= 16 #算1个C_block的时候需要算几个A*B
        A = np.random.rand(m, k)
        B = np.random.rand(k, n)
        C = np.zeros((m, n))
        if self.Layer_config.name == 'qkv':
            #tile_q, tile_k, tile_v = [list(arr) for arr in np.array_split(self.tile_allocation, 3)]
            #block_num_n /= 3
            C_block_allocation = self.allocate_tiles_to_blocks(self.tile_allocation,block_num_n)
        #[[0, 16], [1, 17], [2, 18], [3, 19], [4, 20], [5, 21], [6, 22], [7, 23], [8], [9], [10], [11], [12], [13], [14], [15]]
        AB_block_allocation = []
        for tile_list in C_block_allocation:
            AB_block_allocation.append(self.allocate_tiles_to_blocks(tile_list,block_num_k))
            #[[0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48], [0], [48]]
        # 先给C的各个块分配在这个layer的tile吧。
        for i in range(0, m, self.block_size_m):
            for j in range(0, n, self.block_size_n):
                C_block = np.zeros((self.block_size_m, self.block_size_n))
                for l in range(0, k, self.block_size_k):# 0,63;64-h;
                    compute_AB_tile_id = AB_block_allocation[math.floor(j/self.block_size_n)][math.floor(l/self.block_size_k)][0]#tile 0 
                    tile_execute = self.Tile_configs.Tile_list[compute_AB_tile_id]
                    # load Ablock Bblock
                    A_block = Data_Split_Tensor(self.A, [[i,i + self.block_size_m],[l,l + self.block_size_k]])
                    B_block = Data_Split_Tensor(self.B, [[l,l + self.block_size_k],[j,j + self.block_size_n]])
                    # TODO:后面需要验证一下
                    # A_block = A[i:i + self.block_size_m, l:l + self.block_size_k] # A1 * B1 的情况 (4,64)
                    # B_block = B[l:l + self.block_size_k, j:j + self.block_size_n] # (64,64)
                    # 分小小块计算
                    C_block += self.compute_block(A_block, B_block,tile_execute) # 这里面要斟酌一下了哈
                    #在这里draw一下
                    draw(self.Tile_configs,compute_AB_tile_id)
                #self.store_result( C, C_block, i, j)
        return C,A,B

   

# def generate_compute_block(Tile_configs,Layer_configs):
#     for layer_id in range(len(Layer_configs)):
#         #送一个layer_id进去
#         Compute

# def main():
#     #TODO:把VAR的情况给整理完 json文件 打包 KV算在FI里面，相当于attn matmul的过程只有2个FI没有parameter
#     tile_total_count = 16 #TODO:还有一个问题是依赖关系咋写 目前是全部前后依赖，似乎应该列出来一个依赖 TODO: KV
    
#     FI_initial_dimension = [[25,1024],[25,1024],[25,55],[25,1024],[25,1024],[25,4096]]        # mat A 先画6个layer
#     Parameter_initial_dimension = [[1024,3072],None,None,[1024,1024],[1024,4096],[4096,1024]] # mat B [涵盖了parameter和KV]
#     KV_initial_dimension = [None,[1024,55],[55,1024],None,None,None] #需要送到DRAM里面 所以doublebuffer一下
#     FO_initial_dimension = [[25,3072],[25,55],[25,1024],[25,1024],[25,4096],[25,1024]]                  # mat C

#     #TODO: 决策出一个group_layer_tile_num 分配方式 
#     # 感觉可以尽量搞成偶数且可拆解【interlayer paper是这样的】
#     # 还得考虑到前后layer通信啥的方便不方便
#     # 其中约束条件是每个group 求和max = tile_total_count 
#     group_layer_tile_num = [[8,2,2,4],[8,8]] #  两个layer_fusion_group layer0 和 layer1在一个group，分别分配3个tile 和 5个tile
#     #TODO： 这里需要写一个group_layer_tile_num 分配到 group_layer_tile_ids的函数 关系到怎么map到空间
#     group_layer_tile_ids = [[[0,1,2,3,4,5,6,7],[8,9],[10,11],[12,13,14,15]],
#                             [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]] #layer_fusion_group id ; layer_id ;tile_id #self.tiles_id = [[0, 1], [2], [0, 1, 2]] 即每个layer被分配的是谁 我希望是1个layer_fusion组id 然后里面的layer_id, tile_id
#     block_size = [[1,2,2],[2,3,2],[2,2,2]] #[1,2,2]是layer1分别在[m,k,n]上的分块大小
#     FLOPs,parameter_buffer,parameter_DRAM,FI,FO = Generate_config().generate()
#     print(FLOPs,parameter_buffer,parameter_DRAM,FI,FO)
    

   
# if __name__ == "__main__":
#     main()