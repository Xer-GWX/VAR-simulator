from config import generate_config
from compute_block import Compute_Layer

# A = [[25,1024],[25,1024],[25,55],[25,1024],[25,1024],[25,4096]]
# B = [[1024,3072],[1024,55],[55,1024],[1024,1024],[1024,4096],[4096,1024]]
# tiles_group_total = 2
# tiles_group_num = [0,0,0,0,1,1]
# tiles_allocate_num = [72,18,18,36,72,72]
# tile_allocate = [list(range(0, 72)),list(range(72, 90)),list(range(90, 108)),list(range(108, 144)),list(range(0, 72)),list(range(72, 144))]
# block_size = [[4,64,64],[4,64,4],[4,4,64],[4,64,64],[4,64,64],[4,64,64]]

Tile_configs,Layer_configs,Layer_Map_configs = generate_config() # 得到hardware config；layer config；map的结果
for Layer_Map_config in Layer_Map_configs:
    # 这里是开始算第一个layer的情况
    Compute_Layer(Tile_configs,Layer_Map_config)
    