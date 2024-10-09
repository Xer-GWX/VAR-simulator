import tools
from hardware import load_hardware_json,Hardware

class DataTensor:
    def __init__(self, name: str = None, shape: list = None, dtype: str = None, data_type: str = None):
        self.name = name
        self.shape = shape  # [1, 4096, 4096]
        self.dtype = dtype

class LayerConfig:
    def __init__(self, name, type, input: list[DataTensor], output: list[DataTensor], param: dict[DataTensor]):
        self.name = name
        self.type = type
        self.input = input # 列表 input[0].shape
        self.output = output
        self.param = param
class LayerMapConfig:
    def __init__(self, LayerConfig:LayerConfig, block_size, layer_group_id, tile_num, tile_allocation):
        self.LayerConfig = LayerConfig
        self.block_size = block_size  # block size: [m, k, n]
        self.group_id = layer_group_id  # 当前layer使用的tile group
        self.tile_num = tile_num
        self.tile_allocation = tile_allocation

class Mapper:
    def __init__(self,Layer_configs,Tile_configs) -> None:
        self.Layer_configs = Layer_configs
        self.Tile_configs = Tile_configs
        self.fusion_group_num = self.layer_fusion_partition()[-1] + 1

    def execute(self,):
        layer_block_size = self.block_partition() #TODO：这里没有做，不知道要不要返回input_list output_list等等
        layer_fusion_group = self.layer_fusion_partition() # 类似每个layer_id对应一个layer_group_id
        layer_tile_allocate_num = self.layer_tile_partition(layer_fusion_group)
        layer_tile_allocate = self.layer_to_tile_allocate(layer_fusion_group,layer_tile_allocate_num)
        #这个留给后面具体计算的过程吧 dtensor0_list, dtensor0_list_shape = self.data_tensor_partition()#这里是分1个block的结果
        Layer_Map_Config=[]
        
        for index, layer in enumerate(self.Layer_configs):
            layer_map = LayerMapConfig(layer,layer_block_size[index],layer_fusion_group[index], layer_tile_allocate_num[index], layer_tile_allocate[index])
            Layer_Map_Config.append(layer_map)
        return Layer_Map_Config
    
    # TODO:后面看看怎么写，这里先固定每个stage每个block的六个layer的group分法（写的话就按照），test只处理1个stage一个block
    def block_partition(self,):
        layer_block_size = [[4,64,64],[4,64,4],[4,4,64],[4,64,64],[4,64,64],[4,64,64]]
        return layer_block_size
    
    def layer_fusion_partition(self,):
        layer_fusion_group = [0,0,0,0,1,1] #len = layer_num
        return layer_fusion_group
    
    def layer_tile_partition(self,layer_fusion_group):
        layer_tile_allocate_num = [72,18,16,36,72,72] #18
        return layer_tile_allocate_num
    
    def layer_to_tile_allocate(self,layer_fusion_group,layer_tile_allocate_num):
        tile_allocate = [list(range(0, 72)),list(range(72, 90)),list(range(90, 108)),list(range(108, 144)),list(range(0, 72)),list(range(72, 144))]
        return tile_allocate


        
def load_layer_json(file_path) -> list[LayerConfig]:
    config_data = tools.load_json(file_path) # json-->dict
    layer_list = []
    for item in config_data:
        name = item['name']
        type = item['type']
        input_data = [DataTensor(**d) for d in item['input']]
        output_data = [DataTensor(**d) for d in item['output']]
        param_data = {k: DataTensor(**v) for k, v in item['param'].items()}
        layer = LayerConfig(name,type,input_data,output_data,param_data)
        layer_list.append(layer)
    return layer_list
   
def generate_config(): 
    ir_path = ".output/ir_output/basic_list.json"
    Layer_configs = load_layer_json(ir_path)
    #Layer_configs[0].input[0].shape
    hardware_path = "./hardware.json"
    Tile_configs = load_hardware_json(hardware_path)

    Mapper_config = Mapper(Layer_configs,Tile_configs)
    Layer_Map_Configs = Mapper_config.execute()

    fusion_group_num = Mapper_config.fusion_group_num
    Tile_configs.update_map_result(fusion_group_num)

    return Tile_configs,Layer_configs,Layer_Map_Configs


