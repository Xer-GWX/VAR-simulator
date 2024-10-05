import tools
import math
        
class DataTensor:
    def __init__(self, name: str = None, shape: list = None, dtype: str = None, data_type: str = None):
        
        self.name = name
        self.shape = shape  # [1, 4096, 4096]
        self.dtype = dtype
        self.amount = math.prod(shape)
class Data_Split_Tensor: # split后的小tensor
    def __init__(self, dtensor: DataTensor, data_range: list = None):
        # assert (name is not None and range is not None and dtype is not None and data_dict is None) or (name is None and range is None and dtype is None and data_dict is not None)
        # assert locality_type in ALL_LOCALITY_TYPE or locality_type is None
        self.dtensor = dtensor
        self.data_range = data_range
        self.shape = self.get_shape()  # [1, 4096, 4096]
        self.amount = math.prod(self.shape)
        
    def get_shape(self,):
        return [ite[1] - ite[0] for ite in self.data_range]
    
class LayerConfig:
    def __init__(self, name, input, output, param):
        self.name = name
        self.input = input # 列表 input[0].shape
        self.output = output
        self.param = param

# TODO:这里所有duration_time后面需要根据具体参数写一下，这里只是按照dtensor大小（相对值）写的
class Tile:
    def __init__(self,tile_id: int, compute_capacity:int,Bank_num_max: int ,bank_capacity: int, LayerConfig: LayerConfig = None, ):
        self.tile_id = tile_id
        self.LayerConfig = LayerConfig
        self.MFU = MFU(compute_capacity)
        self.Buffer = Buffer(Bank_num_max,bank_capacity)
        
        # 已经解决OK：今天check的结果写n个tile_group对应n个layer_group 需要重新写一下 之前是：不用额外写tile_group ，每次处理layer的时候会分配bank 到时候更新即可，但是可以存一个layerconfig
    # def update_tile(self,):
    #     # layerconfig需要传给tile 然后 传dtensor给具体的buffer buffer传给bank 因为主要是那些bank需要有time的描写
    #     pass
   
class HBM():
    def __init__(self) -> None:
        self.last_access_end_time = 0
        self.start_time_list = []
        self.end_time_list = []
    def load_data(self,compute_end_time,data_amount):
        start_time = max(self.last_access_end_time,compute_end_time)
        duration_time = data_amount * 100 # TODO:把这个东西写成dtensor.ammount 这里100以及后面*10啥的只是为了图看起来可读一点 方便判断流程写的是不是ok
        self.last_access_end_time = start_time + duration_time
        self.start_time_list.append(start_time)
        self.end_time_list.append(self.last_access_end_time)
        return self.last_access_end_time
    # TODO：后面存kv_cache 以及 多出来的FI写一下  
    def store_data(self,):
        pass

class Bank():
    def __init__(self,bank_id,bank_capacity) -> None:
        self.bank_id = bank_id
        self.last_access_end_time = 0
        self.memory_capacity_per_bank = bank_capacity #KB
        self.type = None
        self.double_buffer_groupid = None
        self.start_time_list = []
        self.end_time_list = []
        self.access_flag = 0
        

    #从buffer上面读data出来进入计算
    def read_data(self,compute_end_time:int, dtensor:DataTensor): 
        self.access_flag += 1
        if self.type == 'FI':
            start_time = max(self.last_access_end_time,compute_end_time)
            duration_time = dtensor.shape[0] * dtensor.shape[1] * 10 # TODO：具体时间之后推一下
            self.last_access_end_time = start_time + duration_time
            self.start_time_list.append(start_time)
            self.end_time_list.append(self.last_access_end_time)
            # TODO：需要加载的地方条件判断 以及 需要store的条件判读
        elif self.type == 'Param':
            start_time = max(self.last_access_end_time,compute_end_time)
            duration_time = dtensor.shape[0] * dtensor.shape[1] * 10# TODO：具体时间之后推一下
            self.last_access_end_time = start_time + duration_time
            self.start_time_list.append(start_time)
            self.end_time_list.append(self.last_access_end_time)
        elif self.type == 'FO':
            start_time = max(self.last_access_end_time,compute_end_time)
            duration_time = dtensor.shape[0] * dtensor.shape[1] * 50# TODO：具体时间之后推一下
            self.last_access_end_time = start_time + duration_time
            self.start_time_list.append(start_time)
            self.end_time_list.append(self.last_access_end_time)

        return self.last_access_end_time

    #从计算单元写data到buffer
    def write_data(self,end_time:int,dtensor:DataTensor): 
        if self.type == 'FO':
            start_time = max(self.last_access_end_time,end_time)
            duration_time = dtensor.shape[0] * dtensor.shape[1] * 50 # TODO:把这个东西写成dtensor.ammount
            self.last_access_end_time = start_time + duration_time
            self.start_time_list.append(start_time)
            self.end_time_list.append(self.last_access_end_time)
        elif self.type == 'Param':
            #bank = FI_bank_list[self.access_flag % 2]
            start_time = max(self.last_access_end_time,end_time)
            duration_time = dtensor.shape[0] * dtensor.shape[1] * 10 # TODO:把这个东西写成dtensor.ammount
            self.last_access_end_time = start_time + duration_time
            self.start_time_list.append(start_time)
            self.end_time_list.append(self.last_access_end_time)
        elif self.type == 'FI': #通信过来写入/从dram读取写入进来
            start_time = max(self.last_access_end_time,end_time)
            duration_time = dtensor.shape[0] * dtensor.shape[1] * 10 # TODO:把这个东西写成dtensor.ammount
            self.last_access_end_time = start_time + duration_time
            self.start_time_list.append(start_time)
            self.end_time_list.append(self.last_access_end_time)
        return self.last_access_end_time
    

class Buffer():
    def __init__(self,bank_num_max:int,bank_capacity,FI_bank_num: int = None, Parameter_bank_num: int = None, FO_bank_num: int = None,HBM_amount = None) -> None:
        self.bank_num_max = bank_num_max
        self.bank_list = [Bank(bank_id,bank_capacity) for bank_id in range(self.bank_num_max)]
        
        self.FI_access_flag = 0
        self.Param_access_flag = 0

        self.FI_bank_num = FI_bank_num
        self.Parameter_bank_num = Parameter_bank_num
        self.FO_bank_num = FO_bank_num
        self.HBM_amount = HBM_amount
        # 把分配给FI P FO的bank又提取了一下list 这里后面get_FI_bank_unit()等会填上
        self.FI_bank = None; self.Param_bank = None; self.FO_bank = None
       

    # ！！！！！！！
    def get_FI_bank_unit(self,) -> list[list[Bank]]: 
        # """ Get or assign FI banks (double-buffered) """
        # if self.FI_bank is None:  # Check if already assigned 但是我还需要它每次execute一个tile的时候能变化
        FI_bank_list = [] 
        single_buffer = int(self.FI_bank_num / 2)
        single_bank_list_0 = [] ; single_bank_list_1 = []
        for bank in self.bank_list[:single_buffer]:
            bank.type = 'FI'
            bank.double_buffer_groupid = 0
            single_bank_list_0.append(bank)
        for bank in self.bank_list[single_buffer:2*single_buffer]:
            bank.type = 'FI'
            bank.double_buffer_groupid = 1
            single_bank_list_1.append(bank)
        FI_bank_list.append(single_bank_list_0)
        FI_bank_list.append(single_bank_list_1)
        self.FI_bank = FI_bank_list
        return self.FI_bank

    def get_param_bank_unit(self,) -> list[list[Bank]]: 
        param_bank_list = [] 
        single_buffer = int(self.Parameter_bank_num / 2)
        single_bank_list_0 = [] ; single_bank_list_1 = []
        for bank in self.bank_list[self.FI_bank_num : self.FI_bank_num+single_buffer]:
            bank.type = 'Param'
            bank.double_buffer_groupid = 0
            single_bank_list_0.append(bank)
        for bank in self.bank_list[self.FI_bank_num+single_buffer:self.FI_bank_num+self.Parameter_bank_num]:
            bank.type = 'Param'
            bank.double_buffer_groupid = 1
            single_bank_list_1.append(bank)
        param_bank_list.append(single_bank_list_0)
        param_bank_list.append(single_bank_list_1)
        self.Param_bank = param_bank_list
        return self.Param_bank

    def get_FO_bank_unit(self,) -> list[list[Bank]]:
        FO_bank_list = [] 
        single_bank_list_0 = [] 
        for bank in self.bank_list[self.FI_bank_num + self.Parameter_bank_num:]:
            bank.type = 'FO'
            bank.double_buffer_groupid = 0
            single_bank_list_0.append(bank)
        FO_bank_list.append(single_bank_list_0)
        self.FO_bank = FO_bank_list
        return self.FO_bank
          

class NoC_Router():
    def __init__(self) -> None:
        self.last_access_end_time = 0
        self.NoC_bandwidth = 32 * 1024 #MB PER Second
        self.start_time_list = []
        self.end_time_list = []
    def calculate_noc_distance(self,src_tile:Tile, dst_tile:Tile,hardware:'Hardware'):
        x1, y1 = hardware.NoC_router.map_tile_to_grid(src_tile,hardware.Tile_total)
        x2, y2 = hardware.NoC_router.map_tile_to_grid(dst_tile,hardware.Tile_total)
        return self.manhattan_distance(x1,y1,x2,y2)

    def map_tile_to_grid(self,tile:Tile,tile_total:int):
        # 1个 tile_id --> grid
        x = tile.tile_id / tile_total 
        y = tile.tile_id % tile_total
        return [x,y]
        
    def manhattan_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def transfer_data(self, ready_time:int,dstensor0:Data_Split_Tensor,src_tile:Tile, dst_tile:Tile,hardware:'Hardware'):
        """
        从 source_tile 通信到 dest_tile 并返回数据传输完成后的时间 ready_time是dstensor0读取后的时间
        """
        noc_distance = self.calculate_noc_distance(src_tile, dst_tile, hardware)
        transfer_time = 100000 * noc_distance * dstensor0.amount / self.NoC_bandwidth # TODO: 需要check
        # 开始时间应该是：dstensor0 准备好的时间 TODO：所有dstensor都应该存一个访问时间？还是什么
        start_time = max(ready_time,self.last_access_end_time)
        self.last_access_end_time = start_time + transfer_time
        #print(f"Data transfer from Tile {source_tile.tile_id} to Tile {dest_tile.tile_id} takes {transfer_time} cycles")
        # 更新 NoC 传输时间
        self.start_time_list.append(start_time)
        self.end_time_list.append(self.last_access_end_time)
        return self.last_access_end_time

class MFU():
    def __init__(self,compute_capacity:int) -> None:
        self.compute_capacity = compute_capacity
        self.last_access_end_time = 0
        self.compute_list = []
        self.start_time_list = []
        self.end_time_list = []

    def process(self,dstensor0: Data_Split_Tensor, dstensor1: Data_Split_Tensor,result_tensor: Data_Split_Tensor,tile_execute:Tile, hardware:'Hardware'):
        hbm = hardware.HBM
        buffer = tile_execute.Buffer #应该传进来一个tile
        mfu = tile_execute.MFU
        # 每次启用tile的时候需要做的，似乎这里也ok，因为没有重新例化bank，只不过给bank分类了

        buffer.FI_bank_num, buffer.Parameter_bank_num, buffer.FO_bank_num = FI_bank_num, Param_bank_num, FO_bank_num = 2,4,4 # 这里应该是上一行的函数给一个结果，这里先随便设的
        Param_dram = 4
        
        FI_bank = buffer.get_FI_bank_unit(); Param_bank = buffer.get_param_bank_unit(); FO_bank = buffer.get_FO_bank_unit()
        # 这里每次get的bank是一样的，但是我不希望它们重新例化，我希望都是
        # TODO: 每个bank存的多少需要check！！！
        for bank in FI_bank[buffer.FI_access_flag % 2]:
            # 应该是另一组bank正在计算的结果
            if len(mfu.end_time_list) == 0 or len(mfu.end_time_list) == 1: FI_ready_time = bank.read_data(0,dstensor0)
            else: FI_ready_time = bank.read_data(mfu.end_time_list[-2],dstensor0)
        buffer.FI_access_flag += 1 # 用于pingpong

        if len(mfu.end_time_list) == 0 or len(mfu.end_time_list) == 1: Param_load_time = hbm.load_data(0,Param_dram) # input 的 情况
        else: Param_load_time = hbm.load_data(mfu.end_time_list[-2],Param_dram)
        
        for bank in Param_bank[buffer.Param_access_flag % 2]:
            Param_pre_ready_time = bank.write_data(Param_load_time,dstensor1) 
            Param_ready_time = bank.read_data(Param_pre_ready_time,dstensor1) 
        buffer.Param_access_flag += 1 # 用于pingpong
        
        FO_compute_ready_time = mfu.compute(dstensor0,dstensor1,FI_ready_time,Param_ready_time,'mat_mat_mul')

        if len(mfu.end_time_list) == 0 or len(mfu.end_time_list) == 1: # input的情况
            for bank in FO_bank[0]:
                FO_write_ready_time = bank.write_data(FO_compute_ready_time,result_tensor)
        # 需要加上滞留的FO
        else:    
            for bank in FO_bank[0]:
                FO_read_ready_time = bank.read_data(bank.last_access_end_time,result_tensor) # 这里应该是存储的那部分，介于shape一样就直接用的result_tensor
            FO_add_ready_time = mfu.compute(result_tensor,result_tensor,FO_read_ready_time,FO_compute_ready_time,'mat_mat_add')
            
            for bank in FO_bank[0]:
                FO_write_ready_time = bank.write_data(FO_add_ready_time,result_tensor)
    
    def merge(self,dstensor0: Data_Split_Tensor,src_tile:Tile,dst_tile:Tile,hardware:'Hardware'):
        # 从src_tile读取data
        src_FO_bank = src_tile.Buffer.get_FO_bank_unit() # TODO: 查一下这里有没有bug
        dst_FI_bank = dst_tile.Buffer.get_FI_bank_unit(); dst_FO_bank = dst_tile.Buffer.get_FO_bank_unit()
        for bank in src_FO_bank[0]:
            src_FO_read_ready_time = bank.read_data(bank.last_access_end_time,dstensor0)
        for bank in dst_FO_bank[0]:
            dst_FO_ready_time = bank.last_access_end_time
        # 从src通信到dst
        ready_time = max(src_FO_read_ready_time,dst_FO_ready_time)
        com_ready_time = hardware.NoC_router.transfer_data(ready_time,dstensor0,src_tile,dst_tile,hardware)
        
        # 把data写入dst 读取 并进行add计算 然后 写入
        for bank in dst_FI_bank[dst_tile.Buffer.FI_access_flag % 2]:
            FI_write_ready_time = bank.write_data(com_ready_time,dstensor0)
        
        for bank in dst_FI_bank[dst_tile.Buffer.FI_access_flag % 2]:
            FI_read_ready_time = bank.write_data(FI_write_ready_time,dstensor0)
        
        dst_tile.Buffer.FI_access_flag += 1 # 用于pingpong

        FO_add_ready_time = dst_tile.MFU.compute(dstensor0,dstensor0,FI_read_ready_time,dst_FO_ready_time,'mat_mat_add')
        
        for bank in dst_FO_bank[0]:
            FO_write_ready_time = bank.write_data(FO_add_ready_time,dstensor0)
        

    def compute(self,dstensor0: Data_Split_Tensor,dstensor1: Data_Split_Tensor,FI_ready_time,Param_ready_time,op):
        if op == 'mat_mat_mul':
            start_time = max(self.last_access_end_time,FI_ready_time,Param_ready_time)
            duration_time = dstensor0.shape[0] * dstensor0.shape[1] * dstensor1.shape[0]/5
            self.last_access_end_time = start_time + duration_time
            self.start_time_list.append(start_time)
            self.end_time_list.append(self.last_access_end_time)
        elif op == 'mat_mat_add':
            start_time = max(self.last_access_end_time,FI_ready_time,Param_ready_time)
            duration_time = dstensor0.shape[0] * dstensor0.shape[1] * 50
            self.last_access_end_time = start_time + duration_time
            self.start_time_list.append(start_time)
            self.end_time_list.append(self.last_access_end_time)
        

        return self.last_access_end_time


class Hardware:
    def __init__(self,Tile_total,compute_capacity,Bank_num_max,bank_capacity):
        self.HBM = HBM()
        self.NoC_router = NoC_Router()
        self.Tile_total = Tile_total
        # tile_list是一组tile
        self.Tile_list = [Tile(i,compute_capacity,Bank_num_max,bank_capacity) for i in range(Tile_total)]
        # hardware是n个fusion group 得到的 n个tile_list
        self.Tile_groups = None
    def update_map_result(self,fusion_group_num):
        self.Tile_groups = [self.Tile_list for _ in range(fusion_group_num)]
    
    def process(self,dstensor0: Data_Split_Tensor, dstensor1: Data_Split_Tensor,result_tensor: Data_Split_Tensor,tile_execute:Tile,op:str):
        if op == "MM":
            tile_execute.MFU.process(dstensor0,dstensor1,result_tensor,tile_execute,self)
    def merge(self,dstensor0: Data_Split_Tensor,src_tile:Tile,dst_tile:Tile,op:str):
        if op == 'MM':
            dst_tile.MFU.merge(dstensor0,src_tile,dst_tile,self)
        return dst_tile

def load_hardware_json(file_path):
    config_data = tools.load_json(file_path)
    Tile_total = config_data["Tile_total"]
    compute_capacity = config_data['Compute_capacity']
    Bank_num_max = config_data["Bank_num_max"]
    bank_capacity = config_data["Bank_capacity"]
    hardware = Hardware(Tile_total,compute_capacity,Bank_num_max,bank_capacity)
    return hardware




