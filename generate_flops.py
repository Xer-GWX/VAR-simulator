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
  
def load_layer_json(file_path) -> list[LayerConfig]:
    config_data = tools.load_json(file_path) # json-->dict
    layer_list = []
    # layer_types = set() 
    for item in config_data:
        name = item['name']
        type = item['type']
        input_data = [DataTensor(**d) for d in item['input']]
        output_data = [DataTensor(**d) for d in item['output']]
        param_data = {k: DataTensor(**v) for k, v in item['param'].items()}
        layer = LayerConfig(name,type,input_data,output_data,param_data)
        layer_list.append(layer)
        # layer_types.add(type)
    # for type in layer_types:
    #     print(type)
    layer_list = calculate_flops(layer_list)
    return layer_list
   
def generate_config(): 
    ir_path = ".output/ir_output/Layer_list_prefill.json"
    Layer_configs = load_layer_json(ir_path)
    return Layer_configs

def calculate_flops(layer_list:list[LayerConfig]):
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

def draw_flops(layer_list: list[LayerConfig]):
    index, flops = [], []
    
    # 用于跟踪重复标签的计数
    layer_count = {}
    stage_starts = {}
    stage_flops = defaultdict(float)
    max=0
    name = None
    for i,layer in enumerate(layer_list):
        # if layer.name[:6] == "stage9":
        #     print(max)
        #     print(name)
        #     print(1)
        
        layer_type = layer.name
        index.append(layer_type)
        assert layer.flops is not None
        flops.append(layer.flops)
        #layer_type = layer.name[:6]  # 仅取前6位
        # stage_flops[layer.name[:6]] += layer.flops  # 累加 FLOPs 字典可解决
        # if layer.flops>max:
        #     max = layer.flops
        #     name = layer.name
        
        if layer_type.startswith('stage') and layer_type[5].isdigit():
            stage_id = layer_type[:6]  # 例如 'stage0'
            if stage_id not in stage_starts:
                stage_starts[stage_id] = i  # 记录该stage的第一个index
    stages = list(stage_flops.keys())
    flops_values = list(stage_flops.values())
    sum_flops = [sum(flops[:i+1]) for i in range(len(flops))]
    # 设置图表大小
    plt.figure(figsize=(10, 6))  # 增大宽度

    # 调整柱子的宽度
    width = 0.8  # 设置柱子的宽度

    # 绘制柱状图
    x_positions = np.arange(len(index))  # 生成柱状图的 x 坐标
    # plt.bar(stages, flops_values, color='b', alpha=0.7)
    # # 设置 y 轴网格线
    # plt.grid(axis='y')  

    # # 设置 x 轴和 y 轴标签
    # plt.xlabel('Stage', fontsize=12)
    # plt.ylabel('Total FLOPs', fontsize=12)
    # plt.title('Total FLOPs by Stage', fontsize=14)

    # # 保存图像，设置DPI
    # plt.savefig("./Total_FLOPs_by_Stage.png", dpi=300, bbox_inches='tight')  # 高DPI和紧凑布局
    # plt.close()  # 关闭图表以节省内存
    plt.bar(x_positions, sum_flops, color='b', alpha=0.7)

    plt.grid(axis='y')  # 只显示y轴的网格线
    for stage, start_idx in stage_starts.items():
        plt.text(start_idx, sum_flops[start_idx], stage[5], fontsize=12, color='red', ha='center', va='bottom')

    # 设置 x 轴刻度和标签
    # 设置x轴的刻度
    step = 100  # 可以根据需要调整
    plt.xticks(ticks=range(0, len(index), step), labels=x_positions[::step], rotation=90, fontsize=3)      
    
    plt.ylabel('FLOPs', fontsize=12)
    plt.title('FLOPs_prefill_depth16', fontsize=12)
    plt.tight_layout()
    #plt.tight_layout()  # 自动调整布局以防止标签重叠

    # 保存图像，设置DPI
    plt.savefig("./FLOPs_prefill_depth16_sum.png", dpi=600, bbox_inches='tight')  # 高DPI和紧凑布局
    plt.close()  # 关闭图表以节省内存
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
    

if __name__ == "__main__":
    layer_list = generate_config()
    draw_flops(layer_list) 