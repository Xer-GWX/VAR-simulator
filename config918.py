class Tile:
    def __init__(self,tile_id):
        self.tile_id = tile_id
        self.bank_num_max = 32
        self.memory_capacity_per_bank = 32 #KB
        self.FI_bank_num = None
        self.Parameter_bank_num = None
        self.FO_bank_num = None

class TileGroup:
    def __init__(self, group_id, num_tiles):
        self.group_id = group_id
        self.num_tiles = num_tiles  # 该group中包含的tile数量
        self.tiles = [Tile(i) for i in range(num_tiles)]  # 每个tile的ID列表
      
    # def __repr__(self):
    #     return f"TileGroup {self.group_id}: Tiles={self.tiles}, Bank Allocation={self.tile_bank_allocation}"


class LayerConfig:
    def __init__(self, layer_id, A_shape, B_shape, block_size, tile_group,tile_num,tile_allocation):
        self.layer_id = layer_id
        self.FI_shape = A_shape  # A矩阵形状 (对应FI)
        self.Parameter_shape = B_shape  # B矩阵形状 (对应Parameter)
        self.FO_shape = self.generate_C_shape()
        self.block_size = block_size  # block size: [m, k, n]
        self.tile_group = tile_group  # 当前layer使用的tile group
        self.tile_num = tile_num
        self.tile_allocation = tile_allocation
        self.FLOPs_split = []
        self.FI_split = []
        self.FO_split = []
        self.Parameter_split = []
        self.FLOPs_total = self.calculate_flops()
        self.FI_memory_total = self.calculate_fi()
        self.Parameter_memory_total = self.calculate_parameter()
        self.FO_memory_total = self.calculate_fo()

    def generate_C_shape(self):
        return [self.FI_shape[0],self.Parameter_shape[1]]
    
    def calculate_flops(self):
        # FLOPs = 2 * m * n * k (浮点运算量)
        return 2 * self.FI_shape[0] * self.Parameter_shape[1] * self.FI_shape[1]

    def calculate_fi(self):
        # FI = A矩阵的大小 (例如: m * k)
        return self.FI_shape[0] * self.FI_shape[1]

    def calculate_fo(self):
        # FO = B矩阵输出结果的大小 (例如: m * n)
        return self.FI_shape[0] * self.Parameter_shape[1]
    
    def calculate_parameter(self):
        # FO = B矩阵输出结果的大小 (例如: m * n)
        return self.Parameter_shape[0] * self.Parameter_shape[1]

    # def __repr__(self):
    #     return (f"Layer {self.layer_id}: A(FI)={self.A_shape}, B(Parameter)={self.B_shape}, BlockSize={self.block_size}\n"
    #             f"  FLOPs={self.FLOPs}, FI={self.FI}, FO={self.FO}\n"
    #             f"  Tile Group: {self.tile_group.group_id}, Tile Allocation: {self.tile_allocation}\n"
    #             f"  Bank Allocation: FI={self.bank_allocation['FI']}, FO={self.bank_allocation['FO']}, "
    #             f"Parameter={self.bank_allocation['Parameter']}")
# 搞一个def

def generate_config(A,B,block_size,group_total,group_id,tile_num,tile_allocate):
    Tile_configs=[]
    Layer_configs=[]
    for i in range(group_total):
        Tile_configs.append(TileGroup(i, 144))
        
    for i in range(len(A)):
        Layer_configs.append(LayerConfig(i,A[i],B[i],block_size[i],Tile_configs[group_id[i]],tile_num[i],tile_allocate[i]))

    return Tile_configs,Layer_configs


