



# import numpy as np

# class Generate_config():
#     def __init__(self, A=None, B=None, block_size=None, tiles_id=None) -> None:
#         self.A = [[16, 105], [3,10]]#[16, 512], [512, 4096][0, 1], [2, 16, 64],
#         self.B = [[105, 1024], [10, 4]]
#         self.tiles_id = [[1], [0, 1, 2]]
#         self.buffer_capacity = [1024 * len(tile_id) for tile_id in self.tiles_id]
#         self.compute_PE_capacity =  [128 * len(tile_id) for tile_id in self.tiles_id]
#         self.block_size = [ [1, 16, 16], [2, 8, 2]] # 这里注意应该<=维度
#         self.block_size_m = [self.block_size[i][0] for i in range(len(self.block_size))]
#         self.block_size_k = [self.block_size[i][1] for i in range(len(self.block_size))]
#         self.block_size_n = [self.block_size[i][2] for i in range(len(self.block_size))]
#         self.parameter_buffer = [[], [], []]
#         self.parameter_DRAM = [[], [], []]
#         self.FLOPs = [[], [], []]
#         self.FI = [[], [], []]
#         self.FO = [[], [], []]
#         #self.generate()

#     def generate(self):
#         for layer_id in range(len(self.A)):
#             C,A,B = self.block_matrix_multiply(layer_id, self.A[layer_id], self.B[layer_id],
#                                            self.block_size_m[layer_id], self.block_size_n[layer_id],
#                                            self.block_size_k[layer_id])
#             C_expected = np.dot(A, B)
#             print(np.allclose(C, C_expected)) 
#         return self.FLOPs, self.parameter_buffer, self.parameter_DRAM, self.FI, self.FO

#     def load_to_buffer(self, layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l):
#         A_block = A[i:i + block_size_m, l:l + block_size_k]
#         B_block = B[l:l + block_size_k, j:j + block_size_n]
        
#         # FI = A_block.shape[0] * A_block.shape[1]
#         # FO = A_block.shape[0] * B_block.shape[1]
#         # self.FI[layer_id].append(FI)
#         # self.FO[layer_id].append(FO)
        
#         # buffer_size = min(B_block.shape[0] * B_block.shape[1], (self.buffer_capacity[layer_id] - FI - FO) / 2)
#         # self.parameter_buffer[layer_id].append(buffer_size)
#         # dram_size = B_block.shape[0] * B_block.shape[1] - buffer_size
#         # self.parameter_DRAM[layer_id].append(dram_size if dram_size > 0 else 0)
        
#         return A_block, B_block
#     def calculate_splits(self, rows_A, cols_A, cols_B, layer_id):
#         max_FLOPs = self.compute_PE_capacity[layer_id]
        
#         best_rows_split = 1
#         best_cols_split = 1
#         best_cols_B_split = 1
        
#         min_splits = float('inf')
        
#         # 尝试不同的分块维度
#         for rows_split in range(1, rows_A + 1):
#             for cols_split in range(1, cols_A + 1):
#                 for cols_B_split in range(1, cols_B + 1):
#                     # 计算每个分块的FLOPs
#                     block_FLOPs = rows_split * cols_split * cols_B_split
                    
#                     # 确保每个分块的FLOPs不超过最大FLOPs
#                     if block_FLOPs <= max_FLOPs:
#                         # 计算所需的分块数
#                         num_splits = (rows_A + rows_split - 1) // rows_split * \
#                                     (cols_A + cols_split - 1) // cols_split * \
#                                     (cols_B + cols_B_split - 1) // cols_B_split
                        
#                         # 更新最佳分块维度
#                         if num_splits < min_splits:
#                             min_splits = num_splits
#                             best_rows_split = rows_split
#                             best_cols_split = cols_split
#                             best_cols_B_split = cols_B_split
    
#         return best_rows_split, best_cols_split, best_cols_B_split

   
#     def compute_block(self, layer_id, A_block, B_block):
#         rows_A, cols_A = A_block.shape
#         cols_B = B_block.shape[1]
        
#         # 计算最佳的分块维度
#         rows_split, cols_split, cols_B_split = self.calculate_splits(rows_A, cols_A, cols_B, layer_id)
        
#         # 初始化最终的 result 矩阵
#         result = np.zeros((rows_A, cols_B))
        
#         # 遍历每个分块
#         for start_row in range(0, rows_A, rows_split):
#             end_row = min(start_row + rows_split, rows_A)
#             for start_col in range(0, cols_A, cols_split):
#                 end_col = min(start_col + cols_split, cols_A)
#                 for start_col_B in range(0, cols_B, cols_B_split):
#                     end_col_B = min(start_col_B + cols_B_split, cols_B)
                    
#                     # 提取当前分块
#                     A_sub_block = A_block[start_row:end_row, start_col:end_col]
#                     B_sub_block = B_block[start_col:end_col, start_col_B:end_col_B]
                    
#                     # 计算当前分块的结果
#                     split_result = np.dot(A_sub_block, B_sub_block)
                    
#                     # 确定拆分结果在最终 result 中的位置
#                     result[start_row:end_row, start_col_B:end_col_B] += split_result

#                     # 记录每个拆分的 FLOPs、FI、FO 和参数
#                     split_FLOPs = A_sub_block.shape[0] * A_sub_block.shape[1] * B_sub_block.shape[1]
#                     self.FLOPs[layer_id].append(split_FLOPs)
                    
#                     split_FI = A_sub_block.shape[0] * A_sub_block.shape[1]
#                     split_FO = A_sub_block.shape[0] * B_sub_block.shape[1]
#                     self.FI[layer_id].append(split_FI)
#                     self.FO[layer_id].append(split_FO)
                    
#                     split_buffer_size = min(B_sub_block.shape[0] * B_sub_block.shape[1], (self.buffer_capacity[layer_id] - split_FI - split_FO) / 2)
#                     self.parameter_buffer[layer_id].append(split_buffer_size)
#                     split_dram_size = B_sub_block.shape[0] * B_sub_block.shape[1] - split_buffer_size
#                     self.parameter_DRAM[layer_id].append(split_dram_size if split_dram_size > 0 else 0)
        
#         return result

#         # # 计算每个块的维度，确保符合 PE 的计算能力
#         # split_num = int(np.ceil(block_FLOPs / self.compute_PE_capacity[layer_id]))
#         # # 使用 split_num 来控制分块数量，并相应调整 rows_A_split 和 cols_B_split，而 cols_A 不进行分割
#         # #rows_A_split = min(rows_A, int(np.ceil(rows_A / np.sqrt(split_num)))) #不太行这样，因为row_A_split 如果是2/8 没有小数就不行了
#         # #cols_A_split = min(cols_A, int(np.ceil(cols_A / np.sqrt(split_num))))#取消这个
#         # #cols_B_split = min(cols_B, int(np.ceil(cols_B / np.sqrt(split_num))))
#         # # 计算的时候开根号np.sqrt(A)
#         # # 初始化最终的 result 矩阵
#         # result = np.zeros((rows_A, cols_B))
        
#         # # 记录每个拆分的 FLOPs、FI、FO 和参数
#         # for split in range(split_num):
#         #     start_row = split * rows_A_split
#         #     end_row = min((split + 1) * rows_A_split, rows_A)

#         #     # 分割 A_block 的行
#         #     split_A_block = A_block[start_row:end_row, :]

#         #     # 对应地分割 B_block 的列
#         #     start_col = split * cols_B_split
#         #     end_col = min((split + 1) * cols_B_split, cols_B)

#         #     split_B_block = B_block[:, start_col:end_col]
            
#         #     split_FLOPs = split_A_block.shape[0] * split_A_block.shape[1] * split_B_block.shape[1]
#         #     self.FLOPs[layer_id].append(split_FLOPs)
            
#         #     split_FI = split_A_block.shape[0] * split_A_block.shape[1]
#         #     split_FO = split_A_block.shape[0] * split_B_block.shape[1]
#         #     self.FI[layer_id].append(split_FI)
#         #     self.FO[layer_id].append(split_FO)
            
#         #     split_buffer_size = min(split_B_block.shape[0] * split_B_block.shape[1], (self.buffer_capacity[layer_id] - split_FI - split_FO) / 2)
#         #     self.parameter_buffer[layer_id].append(split_buffer_size)
#         #     split_dram_size = split_B_block.shape[0] * split_B_block.shape[1] - split_buffer_size
#         #     self.parameter_DRAM[layer_id].append(split_dram_size if split_dram_size > 0 else 0)
            
#         #     # 计算拆分后的结果
#         #     split_result = np.dot(split_A_block, split_B_block)
            
#         #     # 将拆分结果累加到最终 result 中
#         #     result[start_row:end_row, :] += split_result
        
#         return result

#     # def compute_block(self, layer_id, A_block, B_block):
#     #     rows_A, cols_A = A_block.shape
#     #     cols_B = B_block.shape[1]
#     #     block_FLOPs = rows_A * cols_A * cols_B
        
#     #     split_num = int(np.ceil(block_FLOPs / self.compute_PE_capacity[layer_id]))
#     #     block_FLOPs_per_split = block_FLOPs / split_num
#     #     rows_A_split = min(rows_A, int(np.ceil(rows_A / np.sqrt(split_num))))
#     #     cols_A_split = min(cols_A, int(np.ceil(cols_A / np.sqrt(split_num))))
#     #     cols_B_split = min(cols_B, int(np.ceil(cols_B / np.sqrt(split_num))))
#     #     result = np.zeros((rows_A, cols_B))
        
#     #     # 记录每个拆分的 FLOPs、FI、FO 和参数
#     #     for split in range(split_num):
#     #         split_rows_A = min(rows_A, int(np.ceil(block_FLOPs_per_split / (cols_A * cols_B))))
#     #         split_A_block = A_block[:split_rows_A, :]
#     #         split_B_block = B_block
            
#     #         split_FLOPs = split_A_block.shape[0] * split_A_block.shape[1] * split_B_block.shape[1]
#     #         self.FLOPs[layer_id].append(split_FLOPs)
            
#     #         split_FI = split_A_block.shape[0] * split_A_block.shape[1]
#     #         split_FO = split_A_block.shape[0] * split_B_block.shape[1]
#     #         self.FI[layer_id].append(split_FI)
#     #         self.FO[layer_id].append(split_FO)
            
#     #         split_buffer_size = min(split_B_block.shape[0] * split_B_block.shape[1], (self.buffer_capacity[layer_id] - split_FI - split_FO) / 2)
#     #         self.parameter_buffer[layer_id].append(split_buffer_size)
#     #         split_dram_size = split_B_block.shape[0] * split_B_block.shape[1] - split_buffer_size
#     #         self.parameter_DRAM[layer_id].append(split_dram_size if split_dram_size > 0 else 0)
            
#     #         # 计算拆分后的结果
#     #         split_result = np.dot(split_A_block, split_B_block)
            
#     #         # 确定拆分结果在最终 result 中的位置
#     #         start_row = 0
#     #         end_row = split_rows_A
            
#     #         # 如果当前拆分是最后一个，确保合并的行覆盖了整个矩阵的剩余部分
#     #         if split == split_num - 1:
#     #             end_row = rows_A
            
#     #         result[start_row:end_row, :] += split_result[:end_row - start_row, :]
        
#     #     return result

#     # def store_result(self, layer_id, C, C_block, i, j):
        
#     #     self.FI[layer_id].append(C_block.shape[0] * C_block.shape[1])
#     #     self.FO[layer_id].append(C_block.shape[0] * C_block.shape[1])
#     #     self.parameter_buffer[layer_id].append(C_block.shape[0] * C_block.shape[1])
#     #     self.FLOPs[layer_id].append(C_block.shape[0] * C_block.shape[1])
#     #     C[i:i + C_block.shape[0], j:j + C_block.shape[1]] += C_block
#     def store_result(self, layer_id, C, C_block, i, j):
#         # 计算切片的有效范围
#         end_i = min(i + C_block.shape[0], C.shape[0])
#         end_j = min(j + C_block.shape[1], C.shape[1]) #C_block会固定大小，所以需要切片，如果两个数组的维度不同，广播机制会自动将较小的数组扩展到较大数组的形状。
        
#         # 调整 C_block 的切片大小
#         slice_C_block = C_block[:end_i - i, :end_j - j]
        
#         # 将 C_block 加到目标矩阵 C 中
#         C[i:end_i, j:end_j] += slice_C_block
#         # try:
#         #     self.FI[layer_id].append(C_block.shape[0] * C_block.shape[1])
#         #     self.FO[layer_id].append(C_block.shape[0] * C_block.shape[1])
#         #     self.parameter_buffer[layer_id].append(C_block.shape[0] * C_block.shape[1])
#         #     self.FLOPs[layer_id].append(C_block.shape[0] * C_block.shape[1])
            
#         #     # 调试信息
#         #     # print(f"C shape: {C.shape}")
#         #     # print(f"C_block shape: {C_block.shape}")
#         #     # print(f"Target slice shape: {C[i:i + C_block.shape[0], j:j + C_block.shape[1]].shape}")
            
#         #     C[i:i + C_block.shape[0], j:j + C_block.shape[1]] += C_block
#         # except Exception as e:
#         #     print(f"Error: {e}")
#         #     breakpoint()  # 启动调试器
#         #     raise  # 重新抛出异常


#     def block_matrix_multiply(self, layer_id, A, B, block_size_m, block_size_n, block_size_k):
#         m, k = A
#         k, n = B
#         A = np.random.rand(m, k)
#         B = np.random.rand(k, n)
#         C = np.zeros((m, n))

#         for i in range(0, m, block_size_m):
#             for j in range(0, n, block_size_n):
#                 C_block = np.zeros((block_size_m, block_size_n))
#                 for l in range(0, k, block_size_k):
#                     A_block, B_block = self.load_to_buffer(layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l)
#                     C_block += self.compute_block(layer_id, A_block, B_block)
#                 self.store_result(layer_id, C, C_block, i, j)
#         return C,A,B

# def main():
#     #example = Generate_config()
#     FLOPs,parameter_buffer,parameter_DRAM,FI,FO = Generate_config().generate()
#     print(FLOPs,parameter_buffer,parameter_DRAM,FI,FO)
    
   
# if __name__ == "__main__":
#     main()
# # import numpy as np

# # class Generate_config():
# #     def __init__(self, A=None, B=None, block_size=None, tiles_id=None) -> None:
# #         # 初始化参数
# #         self.A = [[2, 3], [4, 6], [2, 2]]
# #         self.B = [[3, 4], [6, 8], [2, 2]]
# #         self.tiles_id = [[0, 1], [1], [0, 1, 2]]
# #         self.buffer_capacity = [1024 * len(tile_id) for tile_id in self.tiles_id]  # 1MB per tile
# #         self.compute_PE_capacity = [128 * len(tile_id) for tile_id in self.tiles_id]  # PE数量

# #         # 定义块大小
# #         self.block_size = [[1, 2, 2], [2, 3, 2], [2, 2, 2]]
# #         self.block_size_m = [self.block_size[i][0] for i in range(len(self.block_size))]
# #         self.block_size_k = [self.block_size[i][1] for i in range(len(self.block_size))]
# #         self.block_size_n = [self.block_size[i][2] for i in range(len(self.block_size))]

# #         # 参数
# #         self.parameter_buffer = [[], [], []]
# #         self.parameter_DRAM = [[], [], []]
# #         self.FLOPs = [[], [], []]
# #         self.FI = [[], [], []]
# #         self.FO = [[], [], []]
# #         self.generate()  # 进行分块矩阵乘法

# #     def generate(self):
# #         for layer_id in range(len(self.A)):
# #             C = self.block_matrix_multiply(
# #                 layer_id,
# #                 self.A[layer_id],
# #                 self.B[layer_id],
# #                 self.block_size_m[layer_id],
# #                 self.block_size_n[layer_id],
# #                 self.block_size_k[layer_id]
# #             )
# #         return self.FLOPs, self.parameter_buffer, self.parameter_DRAM, self.FI, self.FO

# #     def load_to_buffer(self, layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l):
# #         A_block = A[i:i + block_size_m, l:l + block_size_k]
# #         B_block = B[l:l + block_size_k, j:j + block_size_n]

# #         # 计算存储的量
# #         #self.FI[layer_id].append(A_block.shape[0] * A_block.shape[1])
# #         #self.FO[layer_id].append(A_block.shape[0] * B_block.shape[1])
# #         # buffer_capacity = self.buffer_capacity[layer_id]
# #         # parameter_buffer = min(B_block.shape[0] * B_block.shape[1], (buffer_capacity - self.FI[layer_id][-1] - self.FO[layer_id][-1]) / 2)
# #         # self.parameter_buffer[layer_id].append(parameter_buffer)
# #         # parameter_DRAM = max(B_block.shape[0] * B_block.shape[1] - parameter_buffer, 0)
# #         # self.parameter_DRAM[layer_id].append(parameter_DRAM)

# #         return A_block, B_block

# #     def compute_block(self, layer_id, A_block, B_block):
# #         # 计算每个块的 FLOPs
# #         block_FLOPs = A_block.shape[0] * A_block.shape[1] * B_block.shape[1]
# #         self.FLOPs[layer_id].append(block_FLOPs)
        
# #         # 计算所需的计算块数量
# #         split_num = int(np.ceil(block_FLOPs / self.compute_PE_capacity[layer_id]))

# #         # 计算拆分后的块大小
# #         rows_A = A_block.shape[0]
# #         cols_A = A_block.shape[1]
# #         cols_B = B_block.shape[1]

# #         rows_A_split = min(rows_A, int(np.ceil(rows_A / np.sqrt(split_num))))
# #         cols_A_split = min(cols_A, int(np.ceil(cols_A / np.sqrt(split_num))))
# #         cols_B_split = min(cols_B, int(np.ceil(cols_B / np.sqrt(split_num))))

# #         # 记录每个拆分块的 FI, FO 和参数
# #         total_FI = 0
# #         total_FO = 0
# #         total_parameter_buffer = 0
# #         total_parameter_DRAM = 0

# #         result = np.zeros((A_block.shape[0], B_block.shape[1]))

# #         # 拆分 A_block 和 B_block
# #         for i in range(0, rows_A, rows_A_split):
# #             for j in range(0, cols_B, cols_B_split):
# #                 A_sub_block = A_block[i:i + rows_A_split, :]
# #                 B_sub_block = B_block[:, j:j + cols_B_split]
# #                 block_FLOPs = A_sub_block.shape[0] * A_sub_block.shape[1] * B_sub_block.shape[1]
# #                 self.FLOPs[layer_id].append(block_FLOPs)
                
# #                 total_FI += A_sub_block.shape[0] * A_sub_block.shape[1]
# #                 total_FO += A_sub_block.shape[0] * B_sub_block.shape[1]
# #                 parameter_buffer = min(B_sub_block.shape[0] * B_sub_block.shape[1], (self.buffer_capacity[layer_id] - total_FI - total_FO) / 2)
# #                 total_parameter_buffer += parameter_buffer
# #                 total_parameter_DRAM += max(B_sub_block.shape[0] * B_sub_block.shape[1] - parameter_buffer, 0)

# #                 # 计算每个子块的结果
# #                 result[i:i + rows_A_split, j:j + cols_B_split] += np.dot(A_sub_block, B_sub_block)

# #         # 更新 FI, FO 和参数记录
# #         self.FI[layer_id].append(total_FI)
# #         self.FO[layer_id].append(total_FO)
# #         self.parameter_buffer[layer_id].append(total_parameter_buffer)
# #         self.parameter_DRAM[layer_id].append(total_parameter_DRAM)

# #         return result

# #     def store_result(self, layer_id, C, C_block, i, j):
# #         self.FI[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         self.parameter_buffer[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         self.FO[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         self.FLOPs[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         C[i:i + C_block.shape[0], j:j + C_block.shape[1]] += C_block

# #     def block_matrix_multiply(self, layer_id, A, B, block_size_m, block_size_n, block_size_k):
# #         m, k = A
# #         k, n = B
# #         A = np.random.rand(m, k)
# #         B = np.random.rand(k, n)
# #         C = np.zeros((m, n))

# #         for i in range(0, m, block_size_m):
# #             for j in range(0, n, block_size_n):
# #                 C_block = np.zeros((block_size_m, block_size_n))
# #                 for l in range(0, k, block_size_k):
# #                     # 从DRAM中加载分块到buffer
# #                     A_block, B_block = self.load_to_buffer(layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l)

# #                     # 计算分块的结果
# #                     C_block += self.compute_block(layer_id, A_block, B_block)
                
# #                 # 将分块结果存储到矩阵C中
# #                 self.store_result(layer_id, C, C_block, i, j)
# #         return C

# # import numpy as np

# # class Generate_config():
# #     def __init__(self, A=None, B=None, block_size=None, tiles_id=None) -> None:
# #         self.A = [[16, 512], [16, 1024], [3,1024]]
# #         self.B = [[512, 4096], [1024, 1024], [1024, 4096]]
# #         self.tiles_id = [[0, 1], [1], [0, 1, 2]]
# #         self.buffer_capacity = [1024 * len(tile_id) for tile_id in self.tiles_id]
# #         self.compute_PE_capacity =  [128 * len(tile_id) for tile_id in self.tiles_id]
# #         self.block_size = [[2, 4, 4], [2, 8, 8], [2, 8, 16]]
# #         self.block_size_m = [self.block_size[i][0] for i in range(len(self.block_size))]
# #         self.block_size_k = [self.block_size[i][1] for i in range(len(self.block_size))]
# #         self.block_size_n = [self.block_size[i][2] for i in range(len(self.block_size))]
# #         self.parameter_buffer = [[], [], []]
# #         self.parameter_DRAM = [[], [], []]
# #         self.FLOPs = [[], [], []]
# #         self.FI = [[], [], []]
# #         self.FO = [[], [], []]
# #         self.generate()

# #     def generate(self):
# #         for layer_id in range(len(self.A)):
# #             C = self.block_matrix_multiply(layer_id, self.A[layer_id], self.B[layer_id], 
# #                                             self.block_size_m[layer_id], self.block_size_n[layer_id], 
# #                                             self.block_size_k[layer_id])
# #         return self.FLOPs, self.parameter_buffer, self.parameter_DRAM, self.FI, self.FO

# #     def load_to_buffer(self, layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l):
# #         A_block = A[i:i + block_size_m, l:l + block_size_k]
# #         B_block = B[l:l + block_size_k, j:j + block_size_n]
# #         self.FI[layer_id].append(A_block.shape[0] * A_block.shape[1])
# #         self.FO[layer_id].append(A_block.shape[0] * B_block.shape[1])
# #         buffer_needed = min(B_block.shape[0] * B_block.shape[1], 
# #                             (self.buffer_capacity[layer_id] - self.FI[layer_id][-1] - self.FO[layer_id][-1]) / 2)
# #         self.parameter_buffer[layer_id].append(buffer_needed)
# #         self.parameter_DRAM[layer_id].append(B_block.shape[0] * B_block.shape[1] - buffer_needed if B_block.shape[0] * B_block.shape[1] - buffer_needed > 0 else 0)
# #         return A_block, B_block

# #     def compute_block(self, layer_id, A_block, B_block):
# #         num_operations = A_block.shape[0] * A_block.shape[1] * B_block.shape[1]
# #         self.FLOPs[layer_id].append(num_operations)
# #         split_num = int(np.ceil(num_operations / self.compute_PE_capacity[layer_id])) #向上取整
# #         result = np.zeros((A_block.shape[0], B_block.shape[1]))
        
# #         # 分块计算
# #         for split in range(split_num):
# #             start = split * self.compute_PE_capacity[layer_id]
# #             end = min(start + self.compute_PE_capacity[layer_id], num_operations)
# #             num_operations_split = end - start
            
# #             if num_operations_split > 0:
# #                 # 对于每个分块，计算并累加结果
# #                 partial_result = np.dot(A_block, B_block) / split_num
# #                 result += partial_result
                
# #                 # 更新计算统计信息
# #                 self.FLOPs[layer_id][-1] -= num_operations_split
# #                 self.FLOPs[layer_id].append(num_operations_split)
# #                 self.FI[layer_id].append(A_block.shape[0] * A_block.shape[1])
# #                 self.FO[layer_id].append(A_block.shape[0] * B_block.shape[1])
        
# #         return result

# #     def store_result(self, layer_id, C, C_block, i, j):
# #         self.FI[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         self.parameter_buffer[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         self.FO[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         self.FLOPs[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         C[i:i + C_block.shape[0], j:j + C_block.shape[1]] += C_block

# #     def block_matrix_multiply(self, layer_id, A_dim, B_dim, block_size_m, block_size_n, block_size_k):
# #         m, k = A_dim
# #         k, n = B_dim
# #         A = np.random.rand(m, k)
# #         B = np.random.rand(k, n)
# #         C = np.zeros((m, n))
    
# #         for i in range(0, m, block_size_m):
# #             for j in range(0, n, block_size_n):
# #                 C_block = np.zeros((block_size_m, block_size_n))
# #                 for l in range(0, k, block_size_k):
# #                     A_block, B_block = self.load_to_buffer(layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l)
# #                     C_block += self.compute_block(layer_id, A_block, B_block)
# #                 self.store_result(layer_id, C, C_block, i, j)
# #         return C

# # # import numpy as np

# # # # # 参数设置
# # # # m, k, n = 1024, 1024, 1024  # 矩阵A的维度为mxk，矩阵B的维度为kxn
# # # # block_size_m = 256  # 在m维度上的分块大小
# # # # block_size_n = 256  # 在n维度上的分块大小
# # # # block_size_k = 256  # 在k维度上的分块大小
# # # # buffer_capacity = 256 * 256  # buffer的容量

# # # # # 初始化矩阵
# # # # A = np.random.rand(m, k)
# # # # B = np.random.rand(k, n)
# # # # C = np.zeros((m, n))

# # # # # 从DRAM读取到buffer中的函数
# # # # def load_to_buffer(A, B, block_size_m, block_size_n, block_size_k, start_m, start_n, start_k):
# # # #     A_block = A[start_m:start_m+block_size_m, start_k:start_k+block_size_k]
# # # #     B_block = B[start_k:start_k+block_size_k, start_n:start_n+block_size_n]
# # # #     return A_block, B_block

# # # # # 分块矩阵相乘并存储结果
# # # # def compute_block(A_block, B_block):
# # # #     return np.dot(A_block, B_block)

# # # # # 将分块结果整合到最终结果矩阵C
# # # # def store_result(C, C_block, start_m, start_n):
# # # #     C[start_m:start_m+C_block.shape[0], start_n:start_n+C_block.shape[1]] += C_block

# # # # # 主流程
# # # # for i in range(0, m, block_size_m):
# # # #     for j in range(0, n, block_size_n):
# # # #         C_block = np.zeros((block_size_m, block_size_n))
# # # #         for l in range(0, k, block_size_k):
# # # #             # 从DRAM中加载分块到buffer
# # # #             A_block, B_block = load_to_buffer(A, B, block_size_m, block_size_n, block_size_k, i, j, l)
            
# # # #             # 计算分块的结果
# # # #             C_block += compute_block(A_block, B_block)
        
# # # #         # 将分块结果存储到矩阵C中
# # # #         store_result(C, C_block, i, j)

# # # # print("计算完成，最终结果矩阵C：", C)
# # # # import numpy as np

# # # # # Assume A is MxK, B is KxN, and C is MxN
# # # # M, K, N = 128, 128, 128
# # # # A = np.random.rand(M, K)
# # # # B = np.random.rand(K, N)
# # # # C = np.zeros((M, N))

# # # # # Decomposition parameters
# # # # TM, TN, TK = 32, 32, 32  # Block sizes

# # # # # Iterate over the blocks
# # # # for i in range(0, M, TM):
# # # #     for j in range(0, N, TN):
# # # #         for k in range(0, K, TK):
# # # #             # Determine the block sizes for edge cases
# # # #             end_i = min(i + TM, M)
# # # #             end_j = min(j + TN, N)
# # # #             end_k = min(k + TK, K)
            
# # # #             # Extract blocks from A and B
# # # #             A_block = A[i:end_i, k:end_k]
# # # #             B_block = B[k:end_k, j:end_j]
            
# # # #             # Perform block multiplication and accumulate results
# # # #             C[i:end_i, j:end_j] += np.dot(A_block, B_block)

# # # # # Verification
# # # # C_expected = np.dot(A, B)
# # # # print("Are the results identical?", np.allclose(C, C_expected))

import numpy as np

def load_to_buffer(A, B, block_size_m, block_size_n, block_size_k, i, j, l):
    A_block = A[i:i+block_size_m, l:l+block_size_k]
    B_block = B[l:l+block_size_k, j:j+block_size_n]
    return A_block, B_block

def compute_block(A_block, B_block):
    return np.dot(A_block, B_block)

def store_result(C, C_block, i, j):
    C[i:i+C_block.shape[0], j:j+C_block.shape[1]] += C_block

def block_matrix_multiply(A, B, block_size_m, block_size_n, block_size_k):
    m, k = A.shape
    k, n = B.shape
    C = np.zeros((m, n))
    
    # 循环所有layer
    for i in range(0, m, block_size_m):
        for j in range(0, n, block_size_n):
            C_block = np.zeros((block_size_m, block_size_n))
            for l in range(0, k, block_size_k):
                # 从DRAM中加载分块到buffer 即可以形成一下DRAM的量 每层的量
                A_block, B_block = load_to_buffer(A, B, block_size_m, block_size_n, block_size_k, i, j, l)
                
                # 计算分块的结果 即可以形成一下Compute的量
                C_block += compute_block(A_block, B_block) # 这个就是整合的过程
            
            # 将分块结果存储到矩阵C中
            store_result(C, C_block, i, j)
            print(C)
    
    return C

# 示例矩阵
A = np.random.rand(2, 3)
B = np.random.rand(3, 4)

# 定义块大小
block_size_m = 1
block_size_n = 2
block_size_k = 2

# 进行分块矩阵乘法
C = block_matrix_multiply(A, B, block_size_m, block_size_n, block_size_k)

# 验证结果
C_expected = np.dot(A, B)
print(np.allclose(C, C_expected))  # 如果正确应该输出True
print(C_expected)
print('hh')
print(C)

# # # # 计算的tile
# # # # 是一个计算流程吗 还是可以先生成


# import numpy as np

# class Generate_config():
#     def __init__(self, A=None, B=None, block_size=None, tiles_id=None) -> None:
#         self.A = [[25,1024],[25,1024],[25,55],[25,1024],[25,1024],[25,4096]]#[[16, 105], [3,10]]#[16, 512], [512, 4096][0, 1], [2, 16, 64],
#         self.B = [[1024,3072],[1024,55],[55,1024],[1024,1024],[1024,4096],[4096,1024]]#[[105, 1024], [10, 4]]
#         self.tiles_allocate_num = [[72,18,18,36],[72,72]]#[[1], [0, 1, 2]]
#         self.bank_num_max = 32
#         self.memory_capacity_per_bank = 32 #KB 
#         #self.buffer_capacity = #32KB #[1024 * len(tile_id) for tile_id in self.tiles_id]
#         self.compute_PE_capacity = 256 #[256 * len(tile_id) for tile_id in self.tiles_id]
#         self.block_size = [[4,64,64],[4,64,4],[4,4,64],[4,64,64],[4,64,64],[4,64,64]]#[ [1, 16, 16], [2, 8, 2]] # 这里注意应该<=维度
#         self.block_size_m = [self.block_size[i][0] for i in range(len(self.block_size))]
#         self.block_size_k = [self.block_size[i][1] for i in range(len(self.block_size))]
#         self.block_size_n = [self.block_size[i][2] for i in range(len(self.block_size))]
#         self.bank_allocate_num = []#每个layergroup里面分 即144是怎么分的 列表是[layergroupid][tile_]
#         self.parameter_DRAM = []
#         self.FLOPs = [[], [], []]
#         self.FI = [[], [], []]
#         self.FO = [[], [], []]
#         # 应该是class里面每个layer写一个

#     def generate(self):
#         for layer_id in range(len(self.A)):
#             C,A,B = self.block_matrix_multiply(layer_id, self.A[layer_id], self.B[layer_id],
#                                            self.block_size_m[layer_id], self.block_size_n[layer_id],
#                                            self.block_size_k[layer_id])
#             C_expected = np.dot(A, B)
#             print(np.allclose(C, C_expected)) 
#         return self.FLOPs, self.parameter_buffer, self.parameter_DRAM, self.FI, self.FO

#     def load_to_buffer(self, layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l):
#         A_block = A[i:i + block_size_m, l:l + block_size_k]
#         B_block = B[l:l + block_size_k, j:j + block_size_n]
#         return A_block, B_block
    
#     def calculate_splits(self, rows_A, cols_A, cols_B, layer_id):
#         max_FLOPs = self.compute_PE_capacity[layer_id]

#         best_rows_split = 1
#         best_cols_split = 1
#         best_cols_B_split = 1
        
#         min_splits = float('inf')
        
#         # 尝试不同的分块维度
#         for rows_split in range(1, rows_A + 1):
#             for cols_split in range(1, cols_A + 1):
#                 for cols_B_split in range(1, cols_B + 1):
#                     # 计算每个分块的FLOPs
#                     block_FLOPs = 2 * rows_split * cols_split * cols_B_split
                    
#                     # 确保每个分块的FLOPs不超过最大FLOPs
#                     if block_FLOPs <= max_FLOPs:
#                         # 计算所需的分块数
#                         num_splits = (rows_A + rows_split - 1) // rows_split * \
#                                     (cols_A + cols_split - 1) // cols_split * \
#                                     (cols_B + cols_B_split - 1) // cols_B_split
                        
#                         # 更新最佳分块维度
#                         if num_splits < min_splits:
#                             min_splits = num_splits
#                             best_rows_split = rows_split
#                             best_cols_split = cols_split
#                             best_cols_B_split = cols_B_split
    
#         return best_rows_split, best_cols_split, best_cols_B_split

   
#     def compute_block(self, layer_id, A_block, B_block):
#         rows_A, cols_A = A_block.shape
#         cols_B = B_block.shape[1]
        
#         # 计算最佳的分块维度
#         rows_split, cols_split, cols_B_split = self.calculate_splits(rows_A, cols_A, cols_B, layer_id)
#         print(rows_split, cols_split, cols_B_split)
#         # 初始化最终的 result 矩阵
#         result = np.zeros((rows_A, cols_B))
        
#         # 遍历每个分块
#         for start_row in range(0, rows_A, rows_split):
#             end_row = min(start_row + rows_split, rows_A)
#             for start_col in range(0, cols_A, cols_split):
#                 end_col = min(start_col + cols_split, cols_A)
#                 for start_col_B in range(0, cols_B, cols_B_split):
#                     end_col_B = min(start_col_B + cols_B_split, cols_B)
                    
#                     # 提取当前分块
#                     A_sub_block = A_block[start_row:end_row, start_col:end_col]
#                     B_sub_block = B_block[start_col:end_col, start_col_B:end_col_B]
                    
#                     # 计算当前分块的结果
#                     split_result = np.dot(A_sub_block, B_sub_block)
                    
#                     # 确定拆分结果在最终 result 中的位置
#                     result[start_row:end_row, start_col_B:end_col_B] += split_result

#                     # 记录每个拆分的 FLOPs、FI、FO 和参数
#                     split_FLOPs = A_sub_block.shape[0] * A_sub_block.shape[1] * B_sub_block.shape[1]
#                     self.FLOPs[layer_id].append(split_FLOPs)
                    
#                     split_FI = A_sub_block.shape[0] * A_sub_block.shape[1]
#                     split_FO = A_sub_block.shape[0] * B_sub_block.shape[1]
#                     self.FI[layer_id].append(split_FI)
#                     self.FO[layer_id].append(split_FO)
                    
#                     split_buffer_size = min(B_sub_block.shape[0] * B_sub_block.shape[1], (self.buffer_capacity[layer_id] - split_FI - split_FO) / 2)
#                     self.parameter_buffer[layer_id].append(split_buffer_size)
#                     split_dram_size = B_sub_block.shape[0] * B_sub_block.shape[1] - split_buffer_size
#                     self.parameter_DRAM[layer_id].append(split_dram_size if split_dram_size > 0 else 0)
        
#         return result

#     def store_result(self, layer_id, C, C_block, i, j):
#         # 计算切片的有效范围
#         end_i = min(i + C_block.shape[0], C.shape[0])
#         end_j = min(j + C_block.shape[1], C.shape[1]) #C_block会固定大小，所以需要切片，如果两个数组的维度不同，广播机制会自动将较小的数组扩展到较大数组的形状。
        
#         # 调整 C_block 的切片大小
#         slice_C_block = C_block[:end_i - i, :end_j - j]
        
#         # 将 C_block 加到目标矩阵 C 中
#         C[i:end_i, j:end_j] += slice_C_block
       

#     def block_matrix_multiply(self, layer_id, A, B, block_size_m, block_size_n, block_size_k):
#         m, k = A
#         k, n = B
#         A = np.random.rand(m, k)
#         B = np.random.rand(k, n)
#         C = np.zeros((m, n))

#         for i in range(0, m, block_size_m):
#             for j in range(0, n, block_size_n):
#                 C_block = np.zeros((block_size_m, block_size_n))
#                 for l in range(0, k, block_size_k):
#                     A_block, B_block = self.load_to_buffer(layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l)
#                     C_block += self.compute_block(layer_id, A_block, B_block)
#                 self.store_result(layer_id, C, C_block, i, j)
#         return C,A,B

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
# # import numpy as np

# # class Generate_config():
# #     def __init__(self, A=None, B=None, block_size=None, tiles_id=None) -> None:
# #         self.A = [[16, 105], [3,10]]#[16, 512], [512, 4096][0, 1], [2, 16, 64],
# #         self.B = [[105, 1024], [10, 4]]
# #         self.tiles_id = [[1], [0, 1, 2]]
# #         self.buffer_capacity = [1024 * len(tile_id) for tile_id in self.tiles_id]
# #         self.compute_PE_capacity =  [128 * len(tile_id) for tile_id in self.tiles_id]
# #         self.block_size = [ [1, 16, 16], [2, 8, 2]] # 这里注意应该<=维度
# #         self.block_size_m = [self.block_size[i][0] for i in range(len(self.block_size))]
# #         self.block_size_k = [self.block_size[i][1] for i in range(len(self.block_size))]
# #         self.block_size_n = [self.block_size[i][2] for i in range(len(self.block_size))]
# #         self.parameter_buffer = [[], [], []]
# #         self.parameter_DRAM = [[], [], []]
# #         self.FLOPs = [[], [], []]
# #         self.FI = [[], [], []]
# #         self.FO = [[], [], []]
# #         #self.generate()

# #     def generate(self):
# #         for layer_id in range(len(self.A)):
# #             C,A,B = self.block_matrix_multiply(layer_id, self.A[layer_id], self.B[layer_id],
# #                                            self.block_size_m[layer_id], self.block_size_n[layer_id],
# #                                            self.block_size_k[layer_id])
# #             C_expected = np.dot(A, B)
# #             print(np.allclose(C, C_expected)) 
# #         return self.FLOPs, self.parameter_buffer, self.parameter_DRAM, self.FI, self.FO

# #     def load_to_buffer(self, layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l):
# #         A_block = A[i:i + block_size_m, l:l + block_size_k]
# #         B_block = B[l:l + block_size_k, j:j + block_size_n]
        
# #         # FI = A_block.shape[0] * A_block.shape[1]
# #         # FO = A_block.shape[0] * B_block.shape[1]
# #         # self.FI[layer_id].append(FI)
# #         # self.FO[layer_id].append(FO)
        
# #         # buffer_size = min(B_block.shape[0] * B_block.shape[1], (self.buffer_capacity[layer_id] - FI - FO) / 2)
# #         # self.parameter_buffer[layer_id].append(buffer_size)
# #         # dram_size = B_block.shape[0] * B_block.shape[1] - buffer_size
# #         # self.parameter_DRAM[layer_id].append(dram_size if dram_size > 0 else 0)
        
# #         return A_block, B_block
# #     def calculate_splits(self, rows_A, cols_A, cols_B, layer_id):
# #         max_FLOPs = self.compute_PE_capacity[layer_id]
        
# #         best_rows_split = 1
# #         best_cols_split = 1
# #         best_cols_B_split = 1
        
# #         min_splits = float('inf')
        
# #         # 尝试不同的分块维度
# #         for rows_split in range(1, rows_A + 1):
# #             for cols_split in range(1, cols_A + 1):
# #                 for cols_B_split in range(1, cols_B + 1):
# #                     # 计算每个分块的FLOPs
# #                     block_FLOPs = rows_split * cols_split * cols_B_split
                    
# #                     # 确保每个分块的FLOPs不超过最大FLOPs
# #                     if block_FLOPs <= max_FLOPs:
# #                         # 计算所需的分块数
# #                         num_splits = (rows_A + rows_split - 1) // rows_split * \
# #                                     (cols_A + cols_split - 1) // cols_split * \
# #                                     (cols_B + cols_B_split - 1) // cols_B_split
                        
# #                         # 更新最佳分块维度
# #                         if num_splits < min_splits:
# #                             min_splits = num_splits
# #                             best_rows_split = rows_split
# #                             best_cols_split = cols_split
# #                             best_cols_B_split = cols_B_split
    
# #         return best_rows_split, best_cols_split, best_cols_B_split

   
# #     def compute_block(self, layer_id, A_block, B_block):
# #         rows_A, cols_A = A_block.shape
# #         cols_B = B_block.shape[1]
        
# #         # 计算最佳的分块维度
# #         rows_split, cols_split, cols_B_split = self.calculate_splits(rows_A, cols_A, cols_B, layer_id)
        
# #         # 初始化最终的 result 矩阵
# #         result = np.zeros((rows_A, cols_B))
        
# #         # 遍历每个分块
# #         for start_row in range(0, rows_A, rows_split):
# #             end_row = min(start_row + rows_split, rows_A)
# #             for start_col in range(0, cols_A, cols_split):
# #                 end_col = min(start_col + cols_split, cols_A)
# #                 for start_col_B in range(0, cols_B, cols_B_split):
# #                     end_col_B = min(start_col_B + cols_B_split, cols_B)
                    
# #                     # 提取当前分块
# #                     A_sub_block = A_block[start_row:end_row, start_col:end_col]
# #                     B_sub_block = B_block[start_col:end_col, start_col_B:end_col_B]
                    
# #                     # 计算当前分块的结果
# #                     split_result = np.dot(A_sub_block, B_sub_block)
                    
# #                     # 确定拆分结果在最终 result 中的位置
# #                     result[start_row:end_row, start_col_B:end_col_B] += split_result

# #                     # 记录每个拆分的 FLOPs、FI、FO 和参数
# #                     split_FLOPs = A_sub_block.shape[0] * A_sub_block.shape[1] * B_sub_block.shape[1]
# #                     self.FLOPs[layer_id].append(split_FLOPs)
                    
# #                     split_FI = A_sub_block.shape[0] * A_sub_block.shape[1]
# #                     split_FO = A_sub_block.shape[0] * B_sub_block.shape[1]
# #                     self.FI[layer_id].append(split_FI)
# #                     self.FO[layer_id].append(split_FO)
                    
# #                     split_buffer_size = min(B_sub_block.shape[0] * B_sub_block.shape[1], (self.buffer_capacity[layer_id] - split_FI - split_FO) / 2)
# #                     self.parameter_buffer[layer_id].append(split_buffer_size)
# #                     split_dram_size = B_sub_block.shape[0] * B_sub_block.shape[1] - split_buffer_size
# #                     self.parameter_DRAM[layer_id].append(split_dram_size if split_dram_size > 0 else 0)
        
# #         return result

# #         # # 计算每个块的维度，确保符合 PE 的计算能力
# #         # split_num = int(np.ceil(block_FLOPs / self.compute_PE_capacity[layer_id]))
# #         # # 使用 split_num 来控制分块数量，并相应调整 rows_A_split 和 cols_B_split，而 cols_A 不进行分割
# #         # #rows_A_split = min(rows_A, int(np.ceil(rows_A / np.sqrt(split_num)))) #不太行这样，因为row_A_split 如果是2/8 没有小数就不行了
# #         # #cols_A_split = min(cols_A, int(np.ceil(cols_A / np.sqrt(split_num))))#取消这个
# #         # #cols_B_split = min(cols_B, int(np.ceil(cols_B / np.sqrt(split_num))))
# #         # # 计算的时候开根号np.sqrt(A)
# #         # # 初始化最终的 result 矩阵
# #         # result = np.zeros((rows_A, cols_B))
        
# #         # # 记录每个拆分的 FLOPs、FI、FO 和参数
# #         # for split in range(split_num):
# #         #     start_row = split * rows_A_split
# #         #     end_row = min((split + 1) * rows_A_split, rows_A)

# #         #     # 分割 A_block 的行
# #         #     split_A_block = A_block[start_row:end_row, :]

# #         #     # 对应地分割 B_block 的列
# #         #     start_col = split * cols_B_split
# #         #     end_col = min((split + 1) * cols_B_split, cols_B)

# #         #     split_B_block = B_block[:, start_col:end_col]
            
# #         #     split_FLOPs = split_A_block.shape[0] * split_A_block.shape[1] * split_B_block.shape[1]
# #         #     self.FLOPs[layer_id].append(split_FLOPs)
            
# #         #     split_FI = split_A_block.shape[0] * split_A_block.shape[1]
# #         #     split_FO = split_A_block.shape[0] * split_B_block.shape[1]
# #         #     self.FI[layer_id].append(split_FI)
# #         #     self.FO[layer_id].append(split_FO)
            
# #         #     split_buffer_size = min(split_B_block.shape[0] * split_B_block.shape[1], (self.buffer_capacity[layer_id] - split_FI - split_FO) / 2)
# #         #     self.parameter_buffer[layer_id].append(split_buffer_size)
# #         #     split_dram_size = split_B_block.shape[0] * split_B_block.shape[1] - split_buffer_size
# #         #     self.parameter_DRAM[layer_id].append(split_dram_size if split_dram_size > 0 else 0)
            
# #         #     # 计算拆分后的结果
# #         #     split_result = np.dot(split_A_block, split_B_block)
            
# #         #     # 将拆分结果累加到最终 result 中
# #         #     result[start_row:end_row, :] += split_result
        
# #         return result

# #     # def compute_block(self, layer_id, A_block, B_block):
# #     #     rows_A, cols_A = A_block.shape
# #     #     cols_B = B_block.shape[1]
# #     #     block_FLOPs = rows_A * cols_A * cols_B
        
# #     #     split_num = int(np.ceil(block_FLOPs / self.compute_PE_capacity[layer_id]))
# #     #     block_FLOPs_per_split = block_FLOPs / split_num
# #     #     rows_A_split = min(rows_A, int(np.ceil(rows_A / np.sqrt(split_num))))
# #     #     cols_A_split = min(cols_A, int(np.ceil(cols_A / np.sqrt(split_num))))
# #     #     cols_B_split = min(cols_B, int(np.ceil(cols_B / np.sqrt(split_num))))
# #     #     result = np.zeros((rows_A, cols_B))
        
# #     #     # 记录每个拆分的 FLOPs、FI、FO 和参数
# #     #     for split in range(split_num):
# #     #         split_rows_A = min(rows_A, int(np.ceil(block_FLOPs_per_split / (cols_A * cols_B))))
# #     #         split_A_block = A_block[:split_rows_A, :]
# #     #         split_B_block = B_block
            
# #     #         split_FLOPs = split_A_block.shape[0] * split_A_block.shape[1] * split_B_block.shape[1]
# #     #         self.FLOPs[layer_id].append(split_FLOPs)
            
# #     #         split_FI = split_A_block.shape[0] * split_A_block.shape[1]
# #     #         split_FO = split_A_block.shape[0] * split_B_block.shape[1]
# #     #         self.FI[layer_id].append(split_FI)
# #     #         self.FO[layer_id].append(split_FO)
            
# #     #         split_buffer_size = min(split_B_block.shape[0] * split_B_block.shape[1], (self.buffer_capacity[layer_id] - split_FI - split_FO) / 2)
# #     #         self.parameter_buffer[layer_id].append(split_buffer_size)
# #     #         split_dram_size = split_B_block.shape[0] * split_B_block.shape[1] - split_buffer_size
# #     #         self.parameter_DRAM[layer_id].append(split_dram_size if split_dram_size > 0 else 0)
            
# #     #         # 计算拆分后的结果
# #     #         split_result = np.dot(split_A_block, split_B_block)
            
# #     #         # 确定拆分结果在最终 result 中的位置
# #     #         start_row = 0
# #     #         end_row = split_rows_A
            
# #     #         # 如果当前拆分是最后一个，确保合并的行覆盖了整个矩阵的剩余部分
# #     #         if split == split_num - 1:
# #     #             end_row = rows_A
            
# #     #         result[start_row:end_row, :] += split_result[:end_row - start_row, :]
        
# #     #     return result

# #     # def store_result(self, layer_id, C, C_block, i, j):
        
# #     #     self.FI[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #     #     self.FO[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #     #     self.parameter_buffer[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #     #     self.FLOPs[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #     #     C[i:i + C_block.shape[0], j:j + C_block.shape[1]] += C_block
# #     def store_result(self, layer_id, C, C_block, i, j):
# #         # 计算切片的有效范围
# #         end_i = min(i + C_block.shape[0], C.shape[0])
# #         end_j = min(j + C_block.shape[1], C.shape[1]) #C_block会固定大小，所以需要切片，如果两个数组的维度不同，广播机制会自动将较小的数组扩展到较大数组的形状。
        
# #         # 调整 C_block 的切片大小
# #         slice_C_block = C_block[:end_i - i, :end_j - j]
        
# #         # 将 C_block 加到目标矩阵 C 中
# #         C[i:end_i, j:end_j] += slice_C_block
# #         # try:
# #         #     self.FI[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         #     self.FO[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         #     self.parameter_buffer[layer_id].append(C_block.shape[0] * C_block.shape[1])
# #         #     self.FLOPs[layer_id].append(C_block.shape[0] * C_block.shape[1])
            
# #         #     # 调试信息
# #         #     # print(f"C shape: {C.shape}")
# #         #     # print(f"C_block shape: {C_block.shape}")
# #         #     # print(f"Target slice shape: {C[i:i + C_block.shape[0], j:j + C_block.shape[1]].shape}")
            
# #         #     C[i:i + C_block.shape[0], j:j + C_block.shape[1]] += C_block
# #         # except Exception as e:
# #         #     print(f"Error: {e}")
# #         #     breakpoint()  # 启动调试器
# #         #     raise  # 重新抛出异常


# #     def block_matrix_multiply(self, layer_id, A, B, block_size_m, block_size_n, block_size_k):
# #         m, k = A
# #         k, n = B
# #         A = np.random.rand(m, k)
# #         B = np.random.rand(k, n)
# #         C = np.zeros((m, n))

# #         for i in range(0, m, block_size_m):
# #             for j in range(0, n, block_size_n):
# #                 C_block = np.zeros((block_size_m, block_size_n))
# #                 for l in range(0, k, block_size_k):
# #                     A_block, B_block = self.load_to_buffer(layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l)
# #                     C_block += self.compute_block(layer_id, A_block, B_block)
# #                 self.store_result(layer_id, C, C_block, i, j)
# #         return C,A,B

# # def main():
# #     #example = Generate_config()
# #     FLOPs,parameter_buffer,parameter_DRAM,FI,FO = Generate_config().generate()
# #     print(FLOPs,parameter_buffer,parameter_DRAM,FI,FO)
    
   
# # if __name__ == "__main__":
# #     main()
# # # import numpy as np

# # # class Generate_config():
# # #     def __init__(self, A=None, B=None, block_size=None, tiles_id=None) -> None:
# # #         # 初始化参数
# # #         self.A = [[2, 3], [4, 6], [2, 2]]
# # #         self.B = [[3, 4], [6, 8], [2, 2]]
# # #         self.tiles_id = [[0, 1], [1], [0, 1, 2]]
# # #         self.buffer_capacity = [1024 * len(tile_id) for tile_id in self.tiles_id]  # 1MB per tile
# # #         self.compute_PE_capacity = [128 * len(tile_id) for tile_id in self.tiles_id]  # PE数量

# # #         # 定义块大小
# # #         self.block_size = [[1, 2, 2], [2, 3, 2], [2, 2, 2]]
# # #         self.block_size_m = [self.block_size[i][0] for i in range(len(self.block_size))]
# # #         self.block_size_k = [self.block_size[i][1] for i in range(len(self.block_size))]
# # #         self.block_size_n = [self.block_size[i][2] for i in range(len(self.block_size))]

# # #         # 参数
# # #         self.parameter_buffer = [[], [], []]
# # #         self.parameter_DRAM = [[], [], []]
# # #         self.FLOPs = [[], [], []]
# # #         self.FI = [[], [], []]
# # #         self.FO = [[], [], []]
# # #         self.generate()  # 进行分块矩阵乘法

# # #     def generate(self):
# # #         for layer_id in range(len(self.A)):
# # #             C = self.block_matrix_multiply(
# # #                 layer_id,
# # #                 self.A[layer_id],
# # #                 self.B[layer_id],
# # #                 self.block_size_m[layer_id],
# # #                 self.block_size_n[layer_id],
# # #                 self.block_size_k[layer_id]
# # #             )
# # #         return self.FLOPs, self.parameter_buffer, self.parameter_DRAM, self.FI, self.FO

# # #     def load_to_buffer(self, layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l):
# # #         A_block = A[i:i + block_size_m, l:l + block_size_k]
# # #         B_block = B[l:l + block_size_k, j:j + block_size_n]

# # #         # 计算存储的量
# # #         #self.FI[layer_id].append(A_block.shape[0] * A_block.shape[1])
# # #         #self.FO[layer_id].append(A_block.shape[0] * B_block.shape[1])
# # #         # buffer_capacity = self.buffer_capacity[layer_id]
# # #         # parameter_buffer = min(B_block.shape[0] * B_block.shape[1], (buffer_capacity - self.FI[layer_id][-1] - self.FO[layer_id][-1]) / 2)
# # #         # self.parameter_buffer[layer_id].append(parameter_buffer)
# # #         # parameter_DRAM = max(B_block.shape[0] * B_block.shape[1] - parameter_buffer, 0)
# # #         # self.parameter_DRAM[layer_id].append(parameter_DRAM)

# # #         return A_block, B_block

# # #     def compute_block(self, layer_id, A_block, B_block):
# # #         # 计算每个块的 FLOPs
# # #         block_FLOPs = A_block.shape[0] * A_block.shape[1] * B_block.shape[1]
# # #         self.FLOPs[layer_id].append(block_FLOPs)
        
# # #         # 计算所需的计算块数量
# # #         split_num = int(np.ceil(block_FLOPs / self.compute_PE_capacity[layer_id]))

# # #         # 计算拆分后的块大小
# # #         rows_A = A_block.shape[0]
# # #         cols_A = A_block.shape[1]
# # #         cols_B = B_block.shape[1]

# # #         rows_A_split = min(rows_A, int(np.ceil(rows_A / np.sqrt(split_num))))
# # #         cols_A_split = min(cols_A, int(np.ceil(cols_A / np.sqrt(split_num))))
# # #         cols_B_split = min(cols_B, int(np.ceil(cols_B / np.sqrt(split_num))))

# # #         # 记录每个拆分块的 FI, FO 和参数
# # #         total_FI = 0
# # #         total_FO = 0
# # #         total_parameter_buffer = 0
# # #         total_parameter_DRAM = 0

# # #         result = np.zeros((A_block.shape[0], B_block.shape[1]))

# # #         # 拆分 A_block 和 B_block
# # #         for i in range(0, rows_A, rows_A_split):
# # #             for j in range(0, cols_B, cols_B_split):
# # #                 A_sub_block = A_block[i:i + rows_A_split, :]
# # #                 B_sub_block = B_block[:, j:j + cols_B_split]
# # #                 block_FLOPs = A_sub_block.shape[0] * A_sub_block.shape[1] * B_sub_block.shape[1]
# # #                 self.FLOPs[layer_id].append(block_FLOPs)
                
# # #                 total_FI += A_sub_block.shape[0] * A_sub_block.shape[1]
# # #                 total_FO += A_sub_block.shape[0] * B_sub_block.shape[1]
# # #                 parameter_buffer = min(B_sub_block.shape[0] * B_sub_block.shape[1], (self.buffer_capacity[layer_id] - total_FI - total_FO) / 2)
# # #                 total_parameter_buffer += parameter_buffer
# # #                 total_parameter_DRAM += max(B_sub_block.shape[0] * B_sub_block.shape[1] - parameter_buffer, 0)

# # #                 # 计算每个子块的结果
# # #                 result[i:i + rows_A_split, j:j + cols_B_split] += np.dot(A_sub_block, B_sub_block)

# # #         # 更新 FI, FO 和参数记录
# # #         self.FI[layer_id].append(total_FI)
# # #         self.FO[layer_id].append(total_FO)
# # #         self.parameter_buffer[layer_id].append(total_parameter_buffer)
# # #         self.parameter_DRAM[layer_id].append(total_parameter_DRAM)

# # #         return result

# # #     def store_result(self, layer_id, C, C_block, i, j):
# # #         self.FI[layer_id].append(C_block.shape[0] * C_block.shape[1])
# # #         self.parameter_buffer[layer_id].append(C_block.shape[0] * C_block.shape[1])
# # #         self.FO[layer_id].append(C_block.shape[0] * C_block.shape[1])
# # #         self.FLOPs[layer_id].append(C_block.shape[0] * C_block.shape[1])
# # #         C[i:i + C_block.shape[0], j:j + C_block.shape[1]] += C_block

# # #     def block_matrix_multiply(self, layer_id, A, B, block_size_m, block_size_n, block_size_k):
# # #         m, k = A
# # #         k, n = B
# # #         A = np.random.rand(m, k)
# # #         B = np.random.rand(k, n)
# # #         C = np.zeros((m, n))

# # #         for i in range(0, m, block_size_m):
# # #             for j in range(0, n, block_size_n):
# # #                 C_block = np.zeros((block_size_m, block_size_n))
# # #                 for l in range(0, k, block_size_k):
# # #                     # 从DRAM中加载分块到buffer
# # #                     A_block, B_block = self.load_to_buffer(layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l)

# # #                     # 计算分块的结果
# # #                     C_block += self.compute_block(layer_id, A_block, B_block)
                
# # #                 # 将分块结果存储到矩阵C中
# # #                 self.store_result(layer_id, C, C_block, i, j)
# # #         return C

# # # import numpy as np

# # # class Generate_config():
# # #     def __init__(self, A=None, B=None, block_size=None, tiles_id=None) -> None:
# # #         self.A = [[16, 512], [16, 1024], [3,1024]]
# # #         self.B = [[512, 4096], [1024, 1024], [1024, 4096]]
# # #         self.tiles_id = [[0, 1], [1], [0, 1, 2]]
# # #         self.buffer_capacity = [1024 * len(tile_id) for tile_id in self.tiles_id]
# # #         self.compute_PE_capacity =  [128 * len(tile_id) for tile_id in self.tiles_id]
# # #         self.block_size = [[2, 4, 4], [2, 8, 8], [2, 8, 16]]
# # #         self.block_size_m = [self.block_size[i][0] for i in range(len(self.block_size))]
# # #         self.block_size_k = [self.block_size[i][1] for i in range(len(self.block_size))]
# # #         self.block_size_n = [self.block_size[i][2] for i in range(len(self.block_size))]
# # #         self.parameter_buffer = [[], [], []]
# # #         self.parameter_DRAM = [[], [], []]
# # #         self.FLOPs = [[], [], []]
# # #         self.FI = [[], [], []]
# # #         self.FO = [[], [], []]
# # #         self.generate()

# # #     def generate(self):
# # #         for layer_id in range(len(self.A)):
# # #             C = self.block_matrix_multiply(layer_id, self.A[layer_id], self.B[layer_id], 
# # #                                             self.block_size_m[layer_id], self.block_size_n[layer_id], 
# # #                                             self.block_size_k[layer_id])
# # #         return self.FLOPs, self.parameter_buffer, self.parameter_DRAM, self.FI, self.FO

# # #     def load_to_buffer(self, layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l):
# # #         A_block = A[i:i + block_size_m, l:l + block_size_k]
# # #         B_block = B[l:l + block_size_k, j:j + block_size_n]
# # #         self.FI[layer_id].append(A_block.shape[0] * A_block.shape[1])
# # #         self.FO[layer_id].append(A_block.shape[0] * B_block.shape[1])
# # #         buffer_needed = min(B_block.shape[0] * B_block.shape[1], 
# # #                             (self.buffer_capacity[layer_id] - self.FI[layer_id][-1] - self.FO[layer_id][-1]) / 2)
# # #         self.parameter_buffer[layer_id].append(buffer_needed)
# # #         self.parameter_DRAM[layer_id].append(B_block.shape[0] * B_block.shape[1] - buffer_needed if B_block.shape[0] * B_block.shape[1] - buffer_needed > 0 else 0)
# # #         return A_block, B_block

# # #     def compute_block(self, layer_id, A_block, B_block):
# # #         num_operations = A_block.shape[0] * A_block.shape[1] * B_block.shape[1]
# # #         self.FLOPs[layer_id].append(num_operations)
# # #         split_num = int(np.ceil(num_operations / self.compute_PE_capacity[layer_id])) #向上取整
# # #         result = np.zeros((A_block.shape[0], B_block.shape[1]))
        
# # #         # 分块计算
# # #         for split in range(split_num):
# # #             start = split * self.compute_PE_capacity[layer_id]
# # #             end = min(start + self.compute_PE_capacity[layer_id], num_operations)
# # #             num_operations_split = end - start
            
# # #             if num_operations_split > 0:
# # #                 # 对于每个分块，计算并累加结果
# # #                 partial_result = np.dot(A_block, B_block) / split_num
# # #                 result += partial_result
                
# # #                 # 更新计算统计信息
# # #                 self.FLOPs[layer_id][-1] -= num_operations_split
# # #                 self.FLOPs[layer_id].append(num_operations_split)
# # #                 self.FI[layer_id].append(A_block.shape[0] * A_block.shape[1])
# # #                 self.FO[layer_id].append(A_block.shape[0] * B_block.shape[1])
        
# # #         return result

# # #     def store_result(self, layer_id, C, C_block, i, j):
# # #         self.FI[layer_id].append(C_block.shape[0] * C_block.shape[1])
# # #         self.parameter_buffer[layer_id].append(C_block.shape[0] * C_block.shape[1])
# # #         self.FO[layer_id].append(C_block.shape[0] * C_block.shape[1])
# # #         self.FLOPs[layer_id].append(C_block.shape[0] * C_block.shape[1])
# # #         C[i:i + C_block.shape[0], j:j + C_block.shape[1]] += C_block

# # #     def block_matrix_multiply(self, layer_id, A_dim, B_dim, block_size_m, block_size_n, block_size_k):
# # #         m, k = A_dim
# # #         k, n = B_dim
# # #         A = np.random.rand(m, k)
# # #         B = np.random.rand(k, n)
# # #         C = np.zeros((m, n))
    
# # #         for i in range(0, m, block_size_m):
# # #             for j in range(0, n, block_size_n):
# # #                 C_block = np.zeros((block_size_m, block_size_n))
# # #                 for l in range(0, k, block_size_k):
# # #                     A_block, B_block = self.load_to_buffer(layer_id, A, B, block_size_m, block_size_n, block_size_k, i, j, l)
# # #                     C_block += self.compute_block(layer_id, A_block, B_block)
# # #                 self.store_result(layer_id, C, C_block, i, j)
# # #         return C

# # # # import numpy as np

# # # # # # 参数设置
# # # # # m, k, n = 1024, 1024, 1024  # 矩阵A的维度为mxk，矩阵B的维度为kxn
# # # # # block_size_m = 256  # 在m维度上的分块大小
# # # # # block_size_n = 256  # 在n维度上的分块大小
# # # # # block_size_k = 256  # 在k维度上的分块大小
# # # # # buffer_capacity = 256 * 256  # buffer的容量

# # # # # # 初始化矩阵
# # # # # A = np.random.rand(m, k)
# # # # # B = np.random.rand(k, n)
# # # # # C = np.zeros((m, n))

# # # # # # 从DRAM读取到buffer中的函数
# # # # # def load_to_buffer(A, B, block_size_m, block_size_n, block_size_k, start_m, start_n, start_k):
# # # # #     A_block = A[start_m:start_m+block_size_m, start_k:start_k+block_size_k]
# # # # #     B_block = B[start_k:start_k+block_size_k, start_n:start_n+block_size_n]
# # # # #     return A_block, B_block

# # # # # # 分块矩阵相乘并存储结果
# # # # # def compute_block(A_block, B_block):
# # # # #     return np.dot(A_block, B_block)

# # # # # # 将分块结果整合到最终结果矩阵C
# # # # # def store_result(C, C_block, start_m, start_n):
# # # # #     C[start_m:start_m+C_block.shape[0], start_n:start_n+C_block.shape[1]] += C_block

# # # # # # 主流程
# # # # # for i in range(0, m, block_size_m):
# # # # #     for j in range(0, n, block_size_n):
# # # # #         C_block = np.zeros((block_size_m, block_size_n))
# # # # #         for l in range(0, k, block_size_k):
# # # # #             # 从DRAM中加载分块到buffer
# # # # #             A_block, B_block = load_to_buffer(A, B, block_size_m, block_size_n, block_size_k, i, j, l)
            
# # # # #             # 计算分块的结果
# # # # #             C_block += compute_block(A_block, B_block)
        
# # # # #         # 将分块结果存储到矩阵C中
# # # # #         store_result(C, C_block, i, j)

# # # # # print("计算完成，最终结果矩阵C：", C)
# # # # # import numpy as np

# # # # # # Assume A is MxK, B is KxN, and C is MxN
# # # # # M, K, N = 128, 128, 128
# # # # # A = np.random.rand(M, K)
# # # # # B = np.random.rand(K, N)
# # # # # C = np.zeros((M, N))

# # # # # # Decomposition parameters
# # # # # TM, TN, TK = 32, 32, 32  # Block sizes

# # # # # # Iterate over the blocks
# # # # # for i in range(0, M, TM):
# # # # #     for j in range(0, N, TN):
# # # # #         for k in range(0, K, TK):
# # # # #             # Determine the block sizes for edge cases
# # # # #             end_i = min(i + TM, M)
# # # # #             end_j = min(j + TN, N)
# # # # #             end_k = min(k + TK, K)
            
# # # # #             # Extract blocks from A and B
# # # # #             A_block = A[i:end_i, k:end_k]
# # # # #             B_block = B[k:end_k, j:end_j]
            
# # # # #             # Perform block multiplication and accumulate results
# # # # #             C[i:end_i, j:end_j] += np.dot(A_block, B_block)

# # # # # # Verification
# # # # # C_expected = np.dot(A, B)
# # # # # print("Are the results identical?", np.allclose(C, C_expected))

# import numpy as np

# def load_to_buffer(A, B, block_size_m, block_size_n, block_size_k, i, j, l):
#     A_block = A[i:i+block_size_m, l:l+block_size_k]
#     B_block = B[l:l+block_size_k, j:j+block_size_n]
#     return A_block, B_block

# def compute_block(A_block, B_block):
#     return np.dot(A_block, B_block)

# def store_result(C, C_block, i, j):
#     C[i:i+C_block.shape[0], j:j+C_block.shape[1]] += C_block

# def block_matrix_multiply(A, B, block_size_m, block_size_n, block_size_k):
#     m, k = A.shape
#     k, n = B.shape
#     C = np.zeros((m, n))
    
#     # 循环所有layer
#     for i in range(0, m, block_size_m):
#         for j in range(0, n, block_size_n):
#             C_block = np.zeros((block_size_m, block_size_n))
#             for l in range(0, k, block_size_k):
#                 # 从DRAM中加载分块到buffer 即可以形成一下DRAM的量 每层的量
#                 A_block, B_block = load_to_buffer(A, B, block_size_m, block_size_n, block_size_k, i, j, l)
                
#                 # 计算分块的结果 即可以形成一下Compute的量
#                 C_block += compute_block(A_block, B_block) # 这个就是整合的过程
            
#             # 将分块结果存储到矩阵C中
#             store_result(C, C_block, i, j)
#             print(C)
    
#     return C

# # 示例矩阵
# A = np.random.rand(2, 3)
# B = np.random.rand(3, 4)

# # 定义块大小
# block_size_m = 1
# block_size_n = 2
# block_size_k = 2

# # 进行分块矩阵乘法
# C = block_matrix_multiply(A, B, block_size_m, block_size_n, block_size_k)

# # 验证结果
# C_expected = np.dot(A, B)
# print(np.allclose(C, C_expected))  # 如果正确应该输出True
# print(C_expected)
# print('hh')
# print(C)

# # # # # 计算的tile
# # # # # 是一个计算流程吗 还是可以先生成