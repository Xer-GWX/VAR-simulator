import tools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
class DataTensor:
    def __init__(self, name: str = None, shape: list = None, dtype: str = None, data_type: str = None,tag: str =None):
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
        self.flops = None
def extract_number(name, start_idx):
    """Helper function to extract consecutive digits from a string starting at a given index."""
    num_str = ''
    while start_idx < len(name) and name[start_idx].isdigit():
        num_str += name[start_idx]
        start_idx += 1
    return num_str  
def load_layer_json(file_path) -> list[LayerConfig]:
    config_data = tools.load_json(".output/ir_output/Layer_list_prefill.json") # json-->dict
    config_data_decode = tools.load_json(".output/ir_output/Layer_list_decode.json")
    layer_list_prefill = []
    layer_list_decode = []
    layer_list = []
    layer_types = [] 
    # Open a text file to save layer types for each stage and block
    # with open("layer_types_per_stage_block.txt", "w") as file:
    for item in config_data:
        name = item['name']
        type = item['type']
        input_data = [DataTensor(**d) for d in item['input']]
        output_data = [DataTensor(**d) for d in item['output']]
        param_data = {k: DataTensor(**v) for k, v in item['param'].items()}
        layer = LayerConfig(name, type, input_data, output_data, param_data)
        layer_list_prefill.append(layer)
        layer_list.append(layer)
        # Extract 'stagei' and 'blockj' from the name
        # Assuming the format 'stagei_blockj' exists in the name
        # if 'stage' in name and 'block' in name:
        #     # Extract stage and block information
        #     stage_idx = name.find('stage') + 5  # Start after 'stage'
        #     block_idx = name.find('block') + 5  # Start after 'block'
            
        #     # Extract numbers for stage and block
        #     stage = extract_number(name, stage_idx)
        #     block = extract_number(name, block_idx)
        #     # Write the formatted information to the file
        #     file.write(f"Stage {stage}, Block {block}: {type}\n")
    for item in config_data_decode:
        name = item['name']
        type = item['type']
        input_data = [DataTensor(**d) for d in item['input']]
        output_data = [DataTensor(**d) for d in item['output']]
        param_data = {k: DataTensor(**v) for k, v in item['param'].items()}
        layer = LayerConfig(name, type, input_data, output_data, param_data)
        layer_list_decode.append(layer)
        layer_list.append(layer)
    decode_flag = False
    layer_list_prefill = calculate_flops(decode_flag,layer_list_prefill)        
    decode_flag = True
    layer_list_decode = calculate_flops(decode_flag,layer_list_decode)

    flops = []
    for layer in layer_list_prefill:
        if layer.flops == None:
            print(layer.name)
        flops.append(layer.flops)
        # print(layer.name)
    for layer in layer_list_decode:
        #if layer.flops == None:
        #print(f"{layer.name},{layer.type}")
        flops.append(layer.flops)
    # sum_flops = [sum(flops[:i+1]) for i in range(len(flops))]
    # print(max(sum_flops)*1e-12)
    return layer_list,layer_list_prefill,layer_list_decode
   
def generate_config(): 
    ir_path = ".output/ir_output/Layer_list_decode.json"
    Layer_configs = load_layer_json(ir_path)
    return Layer_configs

def calculate_memory(decode_flag:bool,layer_list:list[LayerConfig]):
    if decode_flag:
        for layer in layer_list:
            if layer.type == 'MM':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 4
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.param['weight'].shape[0]
        
            elif layer.type == 'MV':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.param['weight'].shape[2]
            
            elif layer.type == 'attentionMM':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[1].shape[2]
            
            elif layer.type == 'attentionMV':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.input[1].shape[3]
            
            elif layer.type == 'eltwiseadd':
                assert len(layer.input)==2
                assert (len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3) or (len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4)
                input = layer.input[0] if layer.input[0].shape[1] > layer.input[1].shape[1] else layer.input[1]
                if len(layer.input[0].shape) == 3:
                    layer.flops = input.shape[0] * input.shape[1] * input.shape[2] 
                elif len(layer.input[0].shape) == 4:
                    layer.flops = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]

            
            elif layer.type == 'eltwisemul':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3
                input = layer.input[0] if layer.input[0].shape[1] > layer.input[1].shape[1] else layer.input[1]
                layer.flops = input.shape[0] * input.shape[1] * input.shape[2] 
            
            elif layer.type == 'layernorm':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                square_sum_flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                sqrt_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] # vector
                layer.flops = square_sum_flops + sqrt_flops + div_flops
            
            elif layer.type == 'gelu':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 8 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2]
            
            elif layer.type == 'silu':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 4
                layer.flops = 8 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] 


            elif layer.type == 'concat':
                layer.flops = 0
            
            elif layer.type == 'transpose':
                layer.flops = 0
            
            elif layer.type == 'softmax':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                # exp sum div
                exp_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                sum_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2]
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] # vector
                layer.flops = exp_flops + sum_flops + div_flops
            
            elif layer.type == 'conv':
                assert len(layer.input)==1 #2*B*E**2*K**2*W*H
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.param['weight'].shape[0] * layer.param['weight'].shape[1] * layer.param['weight'].shape[2] * layer.param['weight'].shape[3]

            elif layer.type == 'groupnorm':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 4
                square_sum_flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3]
                sqrt_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3]
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3]# vector
                layer.flops = square_sum_flops + sqrt_flops + div_flops
                # TODO: 补一下scale

    else:
        for layer in layer_list:
            if layer.type == 'MM':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.param['weight'].shape[2]
        
            elif layer.type == 'MV':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.param['weight'].shape[2]
            
            elif layer.type == 'attentionMM':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.input[1].shape[3]
            
            elif layer.type == 'attentionMV':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.input[1].shape[3]
            
            elif layer.type == 'eltwiseadd':
                assert len(layer.input)==2
                assert (len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3) or (len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4)
                input = layer.input[0] if layer.input[0].shape[1] > layer.input[1].shape[1] else layer.input[1]
                if len(layer.input[0].shape) == 3:
                    layer.flops = input.shape[0] * input.shape[1] * input.shape[2] 
                elif len(layer.input[0].shape) == 4:
                    layer.flops = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]

            
            elif layer.type == 'eltwisemul':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3
                input = layer.input[0] if layer.input[0].shape[1] > layer.input[1].shape[1] else layer.input[1]
                layer.flops = input.shape[0] * input.shape[1] * input.shape[2] 
            
            elif layer.type == 'layernorm':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                square_sum_flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                sqrt_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] # vector
                layer.flops = square_sum_flops + sqrt_flops + div_flops
            
            elif layer.type == 'gelu':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 8 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2]
            
            elif layer.type == 'silu':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 8 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 


            elif layer.type == 'concat':
                layer.flops = 0
            
            elif layer.type == 'transpose':
                layer.flops = 0
            
            elif layer.type == 'softmax':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 4
                # exp sum div
                exp_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] 
                sum_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] 
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] # vector
                layer.flops = exp_flops + sum_flops + div_flops
            
            elif layer.type == 'conv':
                assert len(layer.input)==1 #2*B*E**2*K**2*W*H
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.param['weight'].shape[0] * layer.param['weight'].shape[1] * layer.param['weight'].shape[2] * layer.param['weight'].shape[3]
        
    return layer_list

def calculate_flops(decode_flag:bool,layer_list:list[LayerConfig]):
    if decode_flag:
        for layer in layer_list:
            if layer.type == 'MM':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 4
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.param['weight'].shape[0]
        
            elif layer.type == 'MV':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.param['weight'].shape[2]
            
            elif layer.type == 'attentionMM':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[1].shape[2]
            
            elif layer.type == 'attentionMV':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.input[1].shape[3]
            
            elif layer.type == 'eltwiseadd':
                assert len(layer.input)==2
                assert (len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3) or (len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4)
                input = layer.input[0] if layer.input[0].shape[1] > layer.input[1].shape[1] else layer.input[1]
                if len(layer.input[0].shape) == 3:
                    layer.flops = input.shape[0] * input.shape[1] * input.shape[2] 
                elif len(layer.input[0].shape) == 4:
                    layer.flops = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]

            
            elif layer.type == 'eltwisemul':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3
                input = layer.input[0] if layer.input[0].shape[1] > layer.input[1].shape[1] else layer.input[1]
                layer.flops = input.shape[0] * input.shape[1] * input.shape[2] 
            
            elif layer.type == 'layernorm':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                square_sum_flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                sqrt_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] # vector
                layer.flops = square_sum_flops + sqrt_flops + div_flops
            
            elif layer.type == 'gelu':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 8 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2]
            
            elif layer.type == 'silu':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 4
                layer.flops = 8 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] 


            elif layer.type == 'concat':
                layer.flops = 0
            
            elif layer.type == 'transpose':
                layer.flops = 0
            
            elif layer.type == 'softmax':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                # exp sum div
                exp_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                sum_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2]
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] # vector
                layer.flops = exp_flops + sum_flops + div_flops
            
            elif layer.type == 'conv':
                assert len(layer.input)==1 #2*B*E**2*K**2*W*H
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.param['weight'].shape[0] * layer.param['weight'].shape[1] * layer.param['weight'].shape[2] * layer.param['weight'].shape[3]

            elif layer.type == 'groupnorm':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 4
                square_sum_flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3]
                sqrt_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3]
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3]# vector
                layer.flops = square_sum_flops + sqrt_flops + div_flops
                # TODO: 补一下scale

    else:
        for layer in layer_list:
            if layer.type == 'MM':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.param['weight'].shape[2]
        
            elif layer.type == 'MV':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.param['weight'].shape[2]
            
            elif layer.type == 'attentionMM':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.input[1].shape[3]
            
            elif layer.type == 'attentionMV':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.input[1].shape[3]
            
            elif layer.type == 'eltwiseadd':
                assert len(layer.input)==2
                assert (len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3) or (len(layer.input[0].shape) == 4 and len(layer.input[1].shape) == 4)
                input = layer.input[0] if layer.input[0].shape[1] > layer.input[1].shape[1] else layer.input[1]
                if len(layer.input[0].shape) == 3:
                    layer.flops = input.shape[0] * input.shape[1] * input.shape[2] 
                elif len(layer.input[0].shape) == 4:
                    layer.flops = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]

            
            elif layer.type == 'eltwisemul':
                assert len(layer.input)==2
                assert len(layer.input[0].shape) == 3 and len(layer.input[1].shape) == 3
                input = layer.input[0] if layer.input[0].shape[1] > layer.input[1].shape[1] else layer.input[1]
                layer.flops = input.shape[0] * input.shape[1] * input.shape[2] 
            
            elif layer.type == 'layernorm':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                square_sum_flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                sqrt_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] # vector
                layer.flops = square_sum_flops + sqrt_flops + div_flops
            
            elif layer.type == 'gelu':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 8 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2]
            
            elif layer.type == 'silu':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 3
                layer.flops = 8 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] 


            elif layer.type == 'concat':
                layer.flops = 0
            
            elif layer.type == 'transpose':
                layer.flops = 0
            
            elif layer.type == 'softmax':
                assert len(layer.input)==1
                assert len(layer.input[0].shape) == 4
                # exp sum div
                exp_flops = 4 * layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] 
                sum_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] 
                div_flops = layer.input[0].shape[0] * layer.input[0].shape[1] * layer.input[0].shape[2] * layer.input[0].shape[3] # vector
                layer.flops = exp_flops + sum_flops + div_flops
            
            elif layer.type == 'conv':
                assert len(layer.input)==1 #2*B*E**2*K**2*W*H
                layer.flops = 2 * layer.input[0].shape[0] * layer.input[0].shape[2] * layer.input[0].shape[3] * layer.param['weight'].shape[0] * layer.param['weight'].shape[1] * layer.param['weight'].shape[2] * layer.param['weight'].shape[3]
        
    return layer_list

def draw_flops(layer_list: list[LayerConfig],list_type:str):
    depth = 16 ; bs = 8
    index, flops = [], []
    
    # 用于跟踪重复标签的计数
    layer_count = {}
    stage_starts = {}
    stage_flops = defaultdict(float)
  
    for i,layer in enumerate(layer_list):
        layer_type = layer.name
        index.append(layer_type)
        assert layer.flops is not None
        flops.append(layer.flops)
        if list_type == "prefill":
            layer_type = layer.name[:6]  # 仅取前6位
            stage_flops[layer.name[:6]] += layer.flops  # 累加 FLOPs 字典可解决
            
            if layer_type.startswith('stage') and layer_type[5].isdigit():
                stage_id = layer_type[:6]  # 例如 'stage0'
                if stage_id not in stage_starts:
                    stage_starts[stage_id] = i  # 记录该stage的第一个index
    stages = list(stage_flops.keys())
    flops_values = list(stage_flops.values())
    sum_flops = [sum(flops[:i+1]) for i in range(len(flops))]
    print(f"[{depth},{bs}]_{list_type}_FLOPs: {max(sum_flops)*1e-12}")

    # 绘制第一个图：FLOPs
    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(index))
    plt.bar(x_positions, flops, color='b', alpha=0.7)
    plt.title(f'{list_type}_FLOPs_[{depth},{bs}]')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('FLOPs', fontsize=12)
    plt.grid(axis='y')
    step = 100  # x轴刻度步长
    plt.xticks(ticks=range(0, len(index), step), labels=x_positions[::step], rotation=90, fontsize=3)
    plt.tight_layout()
    plt.savefig(f"./[{depth},{bs}]_{list_type}_FLOPs.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 绘制第二个图：累积 FLOPs (sum_flops)
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, sum_flops, marker='o', color='b', linestyle='-')
    plt.title(f'{list_type}_Cumulative FLOPs_[{depth},{bs}]')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Cumulative FLOPs', fontsize=12)
    plt.grid(axis='y')
    if list_type == "prefill" or list_type == "all":
        for stage, start_idx in stage_starts.items():
            plt.text(start_idx, sum_flops[start_idx], stage[5], fontsize=12, color='red', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"./[{depth},{bs}]_{list_type}_Cumulative_FLOPs.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 绘制第三个图：stage_flops
    if list_type == "prefill":
        plt.figure(figsize=(10, 6))
        stage_positions = np.arange(len(stages))
        plt.bar(stages, flops_values, color='b', alpha=0.7)
        plt.title(f'{list_type}_Total FLOPs by Stage_[{depth},{bs}]')
        plt.xlabel('Stage', fontsize=12)
        plt.ylabel('Total FLOPs', fontsize=12)
        plt.xticks(stage_positions, stages, rotation=45, fontsize=10)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f"./[{depth},{bs}]_{list_type}_Total_FLOPs_by_Stage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
def draw_analysis(layer_list, layer_list_prefill, layer_list_decode, list_type="prefill", analyze_type="flops"):
    """
    Draws analysis for a specified layer list based on list_type and analyze_type.

    Args:
        layer_list: list, the full list of layers.
        layer_list_prefill: list, the prefill list of layers.
        layer_list_decode: list, the decode list of layers.
        list_type: str, one of "prefill", "decode", or "all" to specify which list to analyze.
        analyze_type: str, either "flops" or "memory" to specify the type of analysis.

    Returns:
        None
    """
    if list_type == "prefill":
        selected_list = layer_list_prefill
    elif list_type == "decode":
        selected_list = layer_list_decode
    elif list_type == "all":
        selected_list = layer_list
    else:
        raise ValueError("Invalid list_type. Choose from 'prefill', 'decode', or 'all'.")

    if analyze_type == "flops":
        draw_flops(selected_list,list_type)
    elif analyze_type == "memory":
        #draw_memory(selected_list)
        pass
    else:
        raise ValueError("Invalid analyze_type. Choose 'flops' or 'memory'.")



if __name__ == "__main__":
    layer_list,layer_list_prefill,layer_list_decode = generate_config()
    draw_analysis(layer_list,layer_list_prefill,layer_list_decode,"all","flops")
    #draw_flops(layer_list) 




    # index, flops = [], []
    
    # # 用于跟踪重复标签的计数
    # layer_count = {}
    # for layer in layer_list:
    #     layer_type = layer.name#[:6]  # 仅取前6位
    #     index.append(layer_type)

    #     # # 更新计数并生成唯一标签
    #     # if layer_type in layer_count:
    #     #     layer_count[layer_type] += 1
    #     # else:
    #     #     layer_count[layer_type] = 1

    #     # # 创建唯一标签，例如 'layer1_1', 'layer1_2' 等
    #     # unique_label = f"{layer_type}_{layer_count[layer_type]}"
    #     # index[-1] = unique_label  # 更新当前索引为唯一标签
        
    #     assert layer.flops is not None
    #     flops.append(layer.flops)
    
    # # 设置图表大小
    # plt.figure(figsize=(120, 40))  # 增大宽度

    # # 绘制柱状图，使用 numpy 的 arange 生成 x 坐标
    # #x = np.arange(len(index))  # 生成柱状图的 x 坐标

    # # 绘制柱状图
    # plt.bar(index, flops, color='b', alpha=0.7)  # width 调整柱子的宽度
    # plt.grid(axis='y')  # 只显示y轴的网格线

    # # 设置 x 轴刻度和标签
    # plt.xticks(rotation=90, fontsize=3)  
    # plt.xlabel('Layer', fontsize=20)
    # plt.ylabel('FLOPs', fontsize=20)
    # plt.title('FLOPs_prefill_depth16', fontsize=30)

    # # 保存图像，设置DPI
    # plt.savefig("./FLOPs_prefill_depth16.png", dpi=300, bbox_inches='tight')  # 高DPI和紧凑布局
    # plt.close()  # 关闭图表以节省内存
    # index, flops = [], []
    # for layer in layer_list:
    #     index.append(layer.name[:7])
    #     assert layer.flops is not None
    #     flops.append(layer.flops)
    
    # # 设置图表大小
    # plt.figure(figsize=(90, 30))  # 宽度20，高度8

    # # 绘制柱状图
    # plt.bar(index, flops, color='b', alpha=0.7)  
    # plt.grid(axis='y')  # 只显示y轴的网格线

    # # 设置x轴的刻度
    # plt.xticks(rotation=90, fontsize=8)  # 旋转90度以防重叠，设置字体大小
    # plt.xlabel('Layer', fontsize=12)
    # plt.ylabel('FLOPs', fontsize=12)
    # plt.title('FLOPs_prefill_depth16', fontsize=14)

    # # 保存图像，设置DPI
    # plt.savefig("./FLOPs_prefill_depth16.png", dpi=300, bbox_inches='tight')  # 高DPI和紧凑布局
    # plt.close()  # 关闭图表以节省内存




    #     # 设置图表大小
    # plt.figure(figsize=(10, 6))  # 增大宽度
    # # 调整柱子的宽度
    # width = 0.8  # 设置柱子的宽度
    # # 绘制柱状图
    # x_positions = np.arange(len(index))  # 生成柱状图的 x 坐标
    # # plt.bar(stages, flops_values, color='b', alpha=0.7)
    # # # 设置 y 轴网格线
    # # plt.grid(axis='y')  

    # # # 设置 x 轴和 y 轴标签
    # # plt.xlabel('Stage', fontsize=12)
    # # plt.ylabel('Total FLOPs', fontsize=12)
    # # plt.title('Total FLOPs by Stage', fontsize=14)

    # # # 保存图像，设置DPI
    # # plt.savefig("./Total_FLOPs_by_Stage.png", dpi=300, bbox_inches='tight')  # 高DPI和紧凑布局
    # # plt.close()  # 关闭图表以节省内存
    # plt.bar(x_positions, flops, color='b', alpha=0.7)

    # plt.grid(axis='y')  # 只显示y轴的网格线
    # # for stage, start_idx in stage_starts.items():
    # #     plt.text(start_idx, sum_flops[start_idx], stage[5], fontsize=12, color='red', ha='center', va='bottom')

    # # 设置 x 轴刻度和标签
    # # 设置x轴的刻度
    # step = 100  # 可以根据需要调整
    # plt.xticks(ticks=range(0, len(index), step), labels=x_positions[::step], rotation=90, fontsize=3)      
    
    # plt.ylabel('FLOPs', fontsize=12)
    # #plt.title('FLOPs_prefill_depth16', fontsize=12)
    # plt.title('FLOPs_all', fontsize=12)

    # plt.tight_layout()
    # #plt.tight_layout()  # 自动调整布局以防止标签重叠

    # # 保存图像，设置DPI
    # #plt.savefig("./FLOPs_prefill_depth16_sum.png", dpi=600, bbox_inches='tight')  # 高DPI和紧凑布局
    # plt.savefig("./FLOPs_all.png", dpi=600, bbox_inches='tight')  # 高DPI和紧凑布局

    # plt.close()  # 关闭图表以节省内存