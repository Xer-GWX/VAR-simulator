import json
import os
import shutil

import numpy as np
import torch
from loguru import logger

from utils.tools import dump_yaml, dump_json

# ALL_DATA_TYPE = (
#     "linear_weight", "linear_weight_scale", "linear_weight_zero", "linear_bias",  # linear_param
#     "attention_matrix_scale_duplicate",  # attention_param
#     "kernel_weight", "kernel_bias",  # convolution_param
#     "rotary_sin_cos_table", "layernorm_weight", "layernorm_bias",  # misc param
#     "sequence",  # sequence activation
#     "figure",  # figure activation
#     "k_cache", "v_cache", "k_cache_t", "v_cache_t",  # k/v cache activation
#     "value"  # value
# )

# ALL_HW_LAYER_TYPE = ("MM", "MV", "attentionMM", "attentionMV",
#                      "conv",
#                      "layernorm", "softmax",
#                      "eltwise", "silu", "rotary",
#                      "concat", "transpose",
#                      "input", "output")

# ALL_LAYER_TYPE = ("MM", "MV", "attentionMM", "attentionMV",
#                   "conv",
#                   "layernorm", "rmsnorm", "softmax",
#                   "gelu", "silu", "rotary", "eltwiseadd", "eltwisemul",
#                   "transpose", "concat",
#                   "input", "output")

# GRANULARITY = ("matrix", "vector", "element", "memory", )


class DataInfoBase:
    def to_dict(self):
        # need to be overwritten
        raise NotImplementedError("Base class DataInfo should not be used")

    def dump_data(self, param_dump_dir):
        return


class ValueInfo(DataInfoBase):  # 常量存储
    def __init__(
        self,
        name: str,
        # value: float,
        dtype: str,
        data_type: str,
    ):
        # #assert value is not None
        #assert name is not None
        #assert dtype == "float16"
        #assert data_type in ALL_DATA_TYPE
        #assert data_type == "value"
        dict_key = name.split(".")[-1]
        self.dict_key = dict_key
        # fp16_value = float(value) # 使用np.float会导致导出yaml的时候格式不统一
        # if fp16_value == float("inf") or fp16_value == float("-inf"):
        #     logger.warning("output_int_scale is INF!")
        #     raise ValueError
        # elif fp16_value == float(0):
        #     logger.warning("output_int_scale is 0!")
        #     raise ValueError
        # elif math.isnan(fp16_value):
        #     pass

        # self.value = fp16_value # torch格式的转为float
        self.name = name
        self.dtype = dtype
        self.data_type = data_type

    def to_dict(self):  # overwrite func
        return {
            self.dict_key: {
                "name": self.name,
                # "value": self.value,
                "dtype": self.dtype,
                "data_type": self.data_type,
            }
        }


class ParamInfo(DataInfoBase):      # 应该是权重
    def __init__(
        self,
        name: str,
        data: torch.Tensor,
        dtype: str,
        data_type: str,
    ):
        #assert data is not None
        #assert name is not None
        #assert dtype in ("float16", "uint4", "int8", "uint16")
        #assert data_type in ALL_DATA_TYPE
        # #assert data_type in (
        #     "linear_weight", "linear_weight_scale", "linear_weight_zero", "linear_bias",  # linear_param
        #     "attention_matrix_scale_duplicate",  # attention_param
        #     "rotary_sin_cos_table", "layernorm_weight", "layernorm_bias",  # misc param
        # )
        dict_key = name.split(".")[-1]

        self.dict_key = dict_key
        self.name = name
        shape = list(data.shape)
        if len(shape) == 1:
            self.shape = [1, 1, shape[0]]
        elif len(shape) == 2:
            self.shape = [1, shape[0], shape[1]]
        elif len(shape) == 3:
            self.shape = [shape[0], shape[1], shape[2]]
        elif len(shape) == 4:
            self.shape = [shape[0] * shape[1], shape[2], shape[3]]
        else:
            raise ValueError("Data shape {} is not supported".format(shape))
        
        #assert len(self.shape) == 3
        self.dtype = dtype
        self.data_type = data_type

        # if dtype in ("uint4", "int8"): # uint4也是使用int8存的，高4bit全0
        #     self.data = data.detach().numpy().astype(np.int8).reshape(self.shape)
        # elif dtype == "uint16":
        #     self.data = data.detach().numpy().astype(np.uint16).reshape(self.shape)
        # elif dtype == "float16":
        #     self.data = data.detach().numpy().astype(np.float16).reshape(self.shape)
        # else:
        #     raise ValueError("Data {} type {} is not supported".format(name, dtype))
    
    def to_dict(self):  # overwrite func
        dict_to_return = {
            "name": self.name,
            "shape": list(self.shape),  # torch.Size转成list
            "dtype": self.dtype,
            "data_type": self.data_type,
        }

        return {
            self.dict_key: dict_to_return
        }
        
    def __json__(self):
        return {'name': self.name, 'shape': self.shape, 'dtype': self.dtype, 'data_type': self.data_type}
    
    def dump_data(self, param_dump_dir):  # overwrite dump func
        return
        # logger.info(f"Dump param to {os.path.join(param_dump_dir, self.name + '.bin as {}'.format(self.dtype))}")
        # self.data.tofile(os.path.join(param_dump_dir, self.name + ".bin"))


class ActivationInfo(DataInfoBase):
    def __init__(
        self,
        name: str,
        shape,
        dtype: str,
        data_type: str,
        keepdim: bool = None,  # 只针对4维的情况
    ):
        if keepdim is None:
            keepdim = False
        #assert shape is not None
        #assert name is not None
        #assert dtype in ("int8", "float16")
        #assert data_type in ALL_DATA_TYPE
        #assert data_type in ("sequence", "figure", "k_cache", "k_cache_t", "v_cache", "v_cache_t")
        #assert (not data_type == "figure") or keepdim
        #assert isinstance(shape, torch.Size) or isinstance(shape, list)
        self.name = name
        if len(shape) == 1:
            self.shape = [1, 1, shape[0]]
        elif len(shape) == 2:
            self.shape = [1, shape[0], shape[1]]
        elif len(shape) == 3:
            self.shape = [shape[0], shape[1], shape[2]]
        elif len(shape) == 4:
            if keepdim:
                self.shape = shape
            else:
                self.shape = [shape[0] * shape[1], shape[2], shape[3]]
        else:
            raise ValueError("Data shape {} is not supported".format(shape))
        # #assert len(self.shape) == 3
        self.dtype = dtype
        self.data_type = data_type
        self.tag = None
    
    def add_data_tag(self, tag):
        self.tag = tag

    def to_dict(self):  # overwrite func
        dict_to_return = {
            "name": self.name,
            "shape": list(self.shape),  # torch.Size转成list
            "dtype": self.dtype,
            "data_type": self.data_type,
        }
        if self.tag is not None:
            dict_to_return["tag"] = self.tag
            
        return dict_to_return
    
    def dump_data(self, act_dump_dir):  # overwrite dump func
        return
    
    def __json__(self):
        return {'name': self.name, 'shape': self.shape, 'dtype': self.dtype, 'data_type': self.data_type}
    
    
class IRGenerator:
    # 随推理流程产生IR并导出权重（不包括K/V Cache）
    def __init__(
            self,
            ir_output_dir: str = ".output/ir_output",
            ir_name: str = "Layer_list",
            stage: str = "prefill",
    ):
        self.ir = list()
        self.all_layer_name = list()  # 用于记录所有层的名称，防止重名
        self.all_tensor_usage = dict()  # 用于记录每个tensor在所有层input和output中出现的情况
        self.all_param_name = list()  # 用于记录所有参数的名称，避免重复导出
        
        self.all_param_list = set()  # 记录所有parainfo的set
        self.all_activation_list = set()  # 记录所有activationinfo的set
        
        self.ir_output_dir = ir_output_dir
        self.ir_dump_dir = os.path.join(self.ir_output_dir, ir_name + "_" + stage)
        if os.path.exists(self.ir_dump_dir):
            os.remove(self.ir_dump_dir)
        # os.mkdir(self.ir_output_dir)

    def delete_head_tail_layers(self):
        #assert len(self.ir) > 0
        #assert self.ir[0]["type"] == "input"
        #assert self.ir[-1]["type"] == "output"
        self.ir = self.ir[1:-1]

    def dump_ir(self, decode_flag):
        out_ir = list()
        input_layer_cnt = 0
        for layer in self.ir:
            if layer["type"] == "input":
                input_layer_cnt += 1
            if decode_flag and input_layer_cnt <= 1:
                continue
            else:
                out_ir.append(layer)
        self.ir = out_ir
        # self.remove_transpose_layer()
        #assert self.check_ir()
        logger.info("Dump high level IR (.yaml) to {}".format(self.ir_dump_dir))
        dump_yaml(out_ir, self.ir_dump_dir + ".yaml")

    def dump_json(self, dump_dir=None):
        if dump_dir is None:
            dump_dir = self.ir_dump_dir + ".json"
        dump_json(self.ir, dump_dir)
        logger.info("Dump high level IR (.json) to {}".format(dump_dir))

    def _set_layer_def(
        self,
        layer_name,
        layer_type,
        input_list,
        output_list,
        param_list,
        granularity,
        structure,
    ):
        layer_def = dict()
        #assert layer_name not in self.all_layer_name
        self.all_layer_name.append(layer_name)
        layer_def["name"] = layer_name
        #assert layer_type in ALL_LAYER_TYPE
        layer_def["type"] = layer_type

        if layer_type == "input":
            assert len(input_list) == 0
            assert len(output_list) > 0
        elif layer_type == "output":
            assert len(input_list) > 0
            assert len(output_list) == 0
        else:
            assert len(input_list) > 0
            assert len(output_list) > 0

        layer_def["input"] = []
        for data_info in input_list:
            assert isinstance(data_info, ActivationInfo)
            #assert data_info.name in self.all_tensor_usage.keys(), "Tensor {} is not registered in IR".format(data_info.name)
            #self.all_tensor_usage[data_info.name] += 1
            layer_def["input"].append(data_info.to_dict())

        layer_def["output"] = []
        for data_info in output_list:
            assert isinstance(data_info, ActivationInfo)
            # if data_info.name not in self.all_tensor_usage:
            #     self.all_tensor_usage[data_info.name] = 0
            # else:
            #     assert layer_type == "input"  # input可能重复出现KV cache
            layer_def["output"].append(data_info.to_dict())
        
        layer_def["param"] = dict()
        for data_info in param_list:
            assert isinstance(data_info, ParamInfo) or isinstance(data_info, ValueInfo)
            if data_info.name not in self.all_param_name:
                self.all_param_name.append(data_info.name)
            layer_def["param"].update(data_info.to_dict())

        layer_def["granularity"] = granularity
        layer_def["structure"] = structure
        
        self.ir.append(layer_def)

    @staticmethod
    def _get_mm_mv_type(
        matrix_A_shape,
        matrix_B_shape,
        linear_flag: bool = True,
    ):
        if len(matrix_A_shape) == 2:
            m, k_A = matrix_A_shape  # input_A: [M, K]
        elif len(matrix_A_shape) == 3:
            _, m, k_A = matrix_A_shape  # input_A: [batch, M, K]
        elif len(matrix_B_shape) == 4:
            _,_,m,k_A = matrix_A_shape
        else:
            raise ValueError("Matrix A shape {} is not supported".format(matrix_A_shape))
        if len(matrix_B_shape) == 2:
            n, k_B = matrix_B_shape  # input_B: [K, N]
        elif len(matrix_B_shape) == 3:
            _, n, k_B = matrix_B_shape  # input_B: [batch, K, N]
        elif len(matrix_B_shape) == 4:
            _,_,n,k_B = matrix_B_shape
        else:
            raise ValueError("Matrix B shape {} is not supported".format(matrix_B_shape))
        # if linear_flag:
        #assert k_A == k_B
        # else:
        #     #assert k_A == n

        op_type = "MV" if (m == 1) else "MM"
        
        return op_type

    def input_layer(
        self,
        # name
        layer_name,
        # input act
        input_act_name,
        input_act_data,
        input_act_dtype,
        # k cache list
        k_cache_name_list,
        k_cache_tag_list,
        k_cache_data_list,
        k_cache_dtype_list,
        k_cache_data_type,
        # v cache list
        v_cache_name_list,
        v_cache_tag_list,
        v_cache_data_list,
        v_cache_dtype_list,
        v_cache_data_type,
    ):
        # #assert input_act_dtype == "float16"
        #assert k_cache_data_type == "k_cache"
        #assert v_cache_data_type == "v_cache_t"

        output_list = list()
        output_list.append(ActivationInfo(input_act_name, input_act_data.shape, input_act_dtype, "sequence", input_act_data))  # input need dump data
        self.all_activation_list.add(ActivationInfo(input_act_name, input_act_data.shape, input_act_dtype, "sequence", input_act_data))
        for data_name, data_tag, data, data_dtype in zip(k_cache_name_list, k_cache_tag_list, k_cache_data_list, k_cache_dtype_list):
            #assert data_dtype in ("int8", "float16")
            tmp_act = ActivationInfo(data_name, data.shape, data_dtype, k_cache_data_type, keepdim=False)
            tmp_act.add_data_tag(data_tag)
            output_list.append(tmp_act)  # k_cache need dump data and tag
            self.all_activation_list.add(tmp_act)
        for data_name, data_tag, data, data_dtype in zip(v_cache_name_list, v_cache_tag_list, v_cache_data_list, v_cache_dtype_list):  # input need dump data
            #assert data_dtype in ("int8", "float16")
            tmp_act = ActivationInfo(data_name, data.shape, data_dtype, v_cache_data_type, keepdim=False)
            tmp_act.add_data_tag(data_tag)
            output_list.append(tmp_act)  # v_cache need dump data and tag
            self.all_activation_list.add(tmp_act)

        self._set_layer_def(layer_name, "input", list(), output_list, list(), "memory", dict())

    def output_layer(
        self,
        # name
        layer_name,
        # output act
        output_act_name,
        output_act_shape,
        output_act_dtype,
        # k cache list
        k_cache_name_list,
        k_cache_tag_list,
        k_cache_shape_list,
        k_cache_dtype_list,
        k_cache_data_type,
        # v cache list
        v_cache_name_list,
        v_cache_tag_list,
        v_cache_shape_list,
        v_cache_dtype_list,
        v_cache_data_type,
    ):
        #assert output_act_dtype == "float16"
        #assert k_cache_data_type == "k_cache"
        #assert v_cache_data_type == "v_cache_t"

        input_list = list()
        input_list.append(ActivationInfo(output_act_name, output_act_shape, output_act_dtype, "sequence"))
        # self.all_activation_list.add(ActivationInfo(output_act_name, output_act_shape, output_act_dtype, "sequence"))
        for data_name, data_tag, data_shape, data_dtype in zip(k_cache_name_list, k_cache_tag_list, k_cache_shape_list, k_cache_dtype_list):
            #assert data_dtype in ("int8", "float16")
            tmp_act = ActivationInfo(data_name, data_shape, data_dtype, k_cache_data_type)
            tmp_act.add_data_tag(data_tag)
            input_list.append(tmp_act)
            # self.all_activation_list.add(tmp_act)
        for data_name, data_tag, data_shape, data_dtype in zip(v_cache_name_list, v_cache_tag_list, v_cache_shape_list, v_cache_dtype_list):
            #assert data_dtype in ("int8", "float16")
            tmp_act = ActivationInfo(data_name, data_shape, data_dtype, v_cache_data_type)
            tmp_act.add_data_tag(data_tag)
            input_list.append(tmp_act)
            # self.all_activation_list.add(tmp_act)
        self._set_layer_def(layer_name, "output", input_list, list(), list(), "memory", dict())

    def linear_layer(  # mm or mv
        self,
        # name
        layer_name,
        # input
        input_name,
        input_shape,
        # output
        output_name,
        output_shape,
        output_dtype,
        # param
        param_name_prefix,
        weight,
        bias,  # may be None
        # structure
        relu_flag,
        granularity,
    ):
        bias_flag = bias is not None
        #assert output_dtype in ("int8", "float16")

        input_list = list()
        input_list.append(ActivationInfo(input_name, input_shape, output_dtype, "sequence"))
        # self.all_activation_list.add(ActivationInfo(input_name, input_shape, "int8", "sequence"))
        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))
        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))
        #assert len(list(weight.shape)) == 2
        
        param_list = list()
        param_list.append(ParamInfo(param_name_prefix + ".weight", weight, "float16", "linear_weight"))
        
        self.all_param_list.add(ParamInfo(param_name_prefix + ".weight", weight, "float16", "linear_weight"))

        if bias_flag:
            param_list.append(ParamInfo(param_name_prefix + ".bias", bias, "float16", "linear_bias"))
            self.all_param_list.add(ParamInfo(param_name_prefix + ".bias", bias, "float16", "linear_bias"))

        op_type = self._get_mm_mv_type(input_shape, param_list[0].shape, linear_flag=True)  # weight shape
        layer_type = op_type
        
        structure_dict = {
            "bias_flag": bias_flag,
            "relu_flag": relu_flag,
            "mask_mode": None,
        }
        self._set_layer_def(layer_name, layer_type, input_list, output_list, param_list, granularity, structure_dict)

    def attention_layer(  # mm or mv
        self,
        # name
        layer_name,
        # input
        input_A_name,
        input_A_shape,
        input_B_name,
        input_B_tag,
        input_B_shape,
        # output
        output_name,
        output_shape,
        output_dtype,
        # param
        param_name_prefix,
        # structure
        mask_id,
        mask_mode,
        granularity,
    ):
        # qkt for SDDMM block masked sparse attention in QKt 
        # qktv for 0/1 block masked sparse of softmax(QKt) in softmax(QKt)V
        #assert mask_mode in ("qkt", "qktv")
        if mask_mode == "qkt":
            input_KV_type = "k_cache"
        elif mask_mode == "qktv":
            input_KV_type = "v_cache_t"
        else:
            raise ValueError("mask_mode {} is not supported".format(mask_mode))
        #assert output_dtype in ("int8", "float16")

        input_list = list()
        input_list.append(ActivationInfo(input_A_name, input_A_shape, output_dtype, "sequence",keepdim=True))
        tmp_kv_act = ActivationInfo(input_B_name, input_B_shape, output_dtype, input_KV_type,keepdim=True)
        if input_KV_type in ("k_cache", "v_cache_t"):
            assert input_B_tag is not None
            tmp_kv_act.add_data_tag(input_B_tag)
        else:
            assert input_B_tag is None
        input_list.append(tmp_kv_act)

        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "sequence",keepdim=True))
        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))
    
        param_list = list()

        op_type = self._get_mm_mv_type(input_A_shape, input_B_shape, linear_flag=False)
        if op_type == "MV" and mask_mode in ("qkt", "qktv"):
            mask_mode = "none"  # decode阶段不做attention mask
        layer_type = "attention" + op_type

        #assert 0 <= mask_id < 32
        structure_dict = {
            "bias_flag": False,
            "relu_flag": False,
            "mask_mode": mask_mode,
            "attention_mask_id": mask_id,
        }

        self._set_layer_def(layer_name, layer_type, input_list, output_list, param_list, granularity, structure_dict)
        #assert mask_mode in ("qkt", "qktv", "none")

    def conv_layer(
            self,
            # name
            layer_name,
            # input
            input_name,
            input_shape,
            # output
            output_name,
            output_shape,
            output_dtype,
            # param
            param_name_prefix,
            kernel,
            bias,  # may be None
            # structure
            stride,
            padding,
            relu_flag,
    ):
        bias_flag = bias is not None
        #assert output_dtype in ("int8", "float16")

        input_list = list()
        input_list.append(ActivationInfo(input_name, input_shape, output_dtype, "figure"))

        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "figure"))

        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "figure"))

        #assert len(list(kernel.shape)) == 4
        param_list = list()
        param_list.append(ParamInfo(param_name_prefix + ".weight", kernel, "uint4", "linear_weight"))

        self.all_param_list.add(ParamInfo(param_name_prefix + ".weight", kernel, "uint4", "linear_weight"))

        if bias_flag:
            param_list.append(ParamInfo(param_name_prefix + ".bias", bias.shape, "float16", "linear_bias"))
            self.all_param_list.add(ParamInfo(param_name_prefix + ".bias", bias.shape, "float16", "linear_bias"))

        structure_dict = {
            "stride": stride,
            "padding": padding,
            "bias_flag": bias_flag,
            "relu_flag": relu_flag,
        }
        self._set_layer_def(layer_name, "conv", input_list, output_list, param_list, "MM", structure_dict)

    def eltwise_layer(
        self,
        # name
        layer_name,
        # input
        input_A_name,
        input_A_shape,
        input_B_name,
        input_B_shape,
        # output
        output_name,
        output_shape,
        output_dtype,
        # param
        param_name_prefix,
        # structure
        eltwise_type,  # eltwise_add, eltwise_mul
        granularity,
    ):
        #assert input_A_shape == input_B_shape == output_shape
        #assert eltwise_type in ("eltwiseadd", "eltwisemul")
        #assert output_dtype in ("int8", "float16")
        
        input_list = list()
        input_list.append(ActivationInfo(input_A_name, input_A_shape, "float16", "sequence"))
        input_list.append(ActivationInfo(input_B_name, input_B_shape, "float16", "sequence"))

        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))
        
        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))

        param_list = list()
        
        structure_dict = {}
        
        self._set_layer_def(layer_name, eltwise_type, input_list, output_list, param_list, granularity, structure_dict)

    def rotary_layer(
        self,
        # name
        layer_name,
        # input
        input_name,
        input_shape,
        # output
        output_name,
        output_shape,
        output_dtype,
        # param
        param_name_prefix,
        sin_weight,
        cos_weight,
        # structure
        granularity,  # only misc layer has dynamic_scale_flag
    ):
        # #assert not dynamic_scale_flag
        #assert output_dtype in ("int8", "float16")
        #assert input_shape == output_shape

        input_list = list()
        input_list.append(ActivationInfo(input_name, input_shape, "float16", "sequence"))
        
        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))
        
        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))

        param_list = list()
        sin_cos_table = torch.cat((sin_weight, cos_weight), dim=2)  # 2 * (1, seq_len, 128) -> (1,  seq_len, 256)
        # sin_cos_table = sin_cos_table.transpose(1, 2)  # (1, seq_len, 256) -> (1, 256, seq_len)
        # sin_cos_table = torch.flatten(sin_cos_table)  # (1, 128, 2) -> (256,)
        param_list.append(ParamInfo(param_name_prefix + ".sin_cos_table", sin_cos_table, "float16", "rotary_sin_cos_table"))
        
        self.all_param_list.add(ParamInfo(param_name_prefix + ".sin_cos_table", sin_cos_table, "float16", "rotary_sin_cos_table"))
        
        structure_dict = {}

        self._set_layer_def(layer_name, "rotary", input_list, output_list, param_list, granularity, structure_dict)

    def layernorm_layer(
        self,
        # name
        layer_name,
        # input
        input_name,
        input_shape,
        # output
        output_name,
        output_shape,
        output_dtype,
        # param
        param_name_prefix,
        weight,
        bias,  # may be None
        # structure
        rms_flag,
        granularity,  # only misc layer has dynamic_scale_flag
    ):
        # #assert not dynamic_scale_flag
        #assert output_dtype in ("int8", "float16")
        #assert input_shape == output_shape
        #assert isinstance(rms_flag, bool)

        input_list = list()
        input_list.append(ActivationInfo(input_name, input_shape, "float16", "sequence"))
        
        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))
        
        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))

        param_list = list()
        param_list.append(ParamInfo(param_name_prefix + ".weight", weight, "float16", "layernorm_weight"))
        
        self.all_param_list.add(ParamInfo(param_name_prefix + ".weight", weight, "float16", "layernorm_weight"))

        bias_flag = bias is not None
        if bias_flag:
            param_list.append(ParamInfo(param_name_prefix + ".bias", bias, "float16", "layernorm_bias"))
            self.all_param_list.add(ParamInfo(param_name_prefix + ".bias", bias, "float16", "layernorm_bias"))
        
        structure_dict = {
            "bias_flag": bias_flag,
        }

        self._set_layer_def(layer_name, "rmsnorm" if rms_flag else "layernorm", input_list, output_list, param_list, granularity, structure_dict)

    def silu_layer(
        self,
        # name
        layer_name,
        # input
        input_name,
        input_shape,
        # output
        output_name,
        output_shape,
        output_dtype,
        # param
        param_name_prefix,
        # output_int_scale,
        # structure
        granularity,  # only misc layer has dynamic_scale_flag
    ):
        # #assert not dynamic_scale_flag
        #assert output_dtype == "float16"
        #assert input_shape == output_shape

        input_list = list()
        input_list.append(ActivationInfo(input_name, input_shape, "float16", "sequence"))
        
        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))
        
        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))

        param_list = list()
        
        structure_dict = {}

        self._set_layer_def(layer_name, "silu", input_list, output_list, param_list, granularity, structure_dict)

    def gelu_layer(
            self,
            # name
            layer_name,
            # input
            input_name,
            input_shape,
            # output
            output_name,
            output_shape,
            output_dtype,
            # param
            param_name_prefix,
            # output_int_scale,
            # structure
            granularity,  # only misc layer has dynamic_scale_flag
    ):
        # #assert not dynamic_scale_flag
        #assert output_dtype == "float16"
        #assert input_shape == output_shape

        input_list = list()
        input_list.append(ActivationInfo(input_name, input_shape, "float16", "sequence"))

        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))

        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))

        param_list = list()

        structure_dict = {}

        self._set_layer_def(layer_name, "gelu", input_list, output_list, param_list, granularity, structure_dict)

    def softmax_layer(
        self,
        # name
        layer_name,
        # input
        input_name,
        input_shape,
        # output
        output_name,
        output_shape,
        output_dtype,
        # param
        param_name_prefix,
        # structure
        dim,
        granularity,  # only misc layer has dynamic_scale_flag
    ):
        # #assert not dynamic_scale_flag
        #assert dim == 2
        #assert output_dtype in ("int8", "float16")
        #assert input_shape == output_shape

        input_list = list()
        input_list.append(ActivationInfo(input_name, input_shape, "float16", "sequence",keepdim=True))
        
        output_list = list()
        output_list.append(ActivationInfo(output_name, output_shape, output_dtype, "sequence",keepdim=True))

        self.all_activation_list.add(ActivationInfo(output_name, output_shape, output_dtype, "sequence"))
        
        param_list = list()
        
        structure_dict = {}

        self._set_layer_def(layer_name, "softmax", input_list, output_list, param_list, granularity, structure_dict)

    def transpose_layer(
        self,
        # name
        layer_name,
        # input
        input_name,
        input_tag,  # may not exist
        input_shape,
        input_dtype,
        input_data_type,
        # output
        output_name,
        output_shape,
        output_data_type,
        # structure
        dim,
        dim_trans,
        granularity,
    ):
        # assert (input_data_type == output_data_type == "sequence") or \
        #        (input_data_type == "k_cache" and output_data_type == "k_cache_t")
        # assert dim == [1, 2]

        # assert len(input_shape) >= 3
        # assert input_shape[0] == output_shape[0]
        # assert input_shape[1] == output_shape[2]
        # assert input_shape[2] == output_shape[1]
        # assert input_shape[3:] == output_shape[3:] if len(input_shape) > 3 else True
        # assert dim_trans in ("up", "down", "keep")

        input_list = list()
        output_list = list()
        if dim_trans == "up":
            input_act = ActivationInfo(input_name, input_shape, input_dtype, input_data_type, keepdim=False)
            output_act = ActivationInfo(output_name, output_shape, input_dtype, output_data_type, keepdim=True)
        elif dim_trans == "down":
            input_act = ActivationInfo(input_name, input_shape, input_dtype, input_data_type, keepdim=True)
            output_act = ActivationInfo(output_name, output_shape, input_dtype, output_data_type, keepdim=True)
        else:
            input_act = ActivationInfo(input_name, input_shape, input_dtype, input_data_type)
            output_act = ActivationInfo(output_name, output_shape, input_dtype, output_data_type)

        if input_tag is not None:
            input_act.add_data_tag(input_tag)

        input_list.append(input_act)
        output_list.append(output_act)

        self.all_activation_list.add(output_act)
        
        param_list = list()
        structure_dict = {
            "dim": dim,
            "dim_trans": dim_trans,
        }
        self._set_layer_def(layer_name, "transpose", input_list, output_list, param_list, "memory", structure_dict)

    def concat_layer(
        self,
        # name
        layer_name,
        # input
        input_seq_name,
        input_seq_shape,
        input_seq_dtype,
        input_kv_cache_name,
        input_kv_cache_tag,  # k_cache_x / v_cache_x
        input_kv_cache_shape,
        input_kv_cache_dtype,
        # output
        output_name,
        output_shape,
        output_dtype,
        # structure
        data_type,  # k_cache or v_cache
        dim,
        granularity,
    ):
        #assert input_seq_dtype == input_kv_cache_dtype == output_dtype
        #assert input_seq_dtype in ("int8", "float16")
        #assert data_type in ("k_cache", "v_cache_t")
        #assert dim in (2, 3)
        dim_0_seq, dim_1_seq, dim_2_seq, dim_3_seq = input_seq_shape
        dim_0_B, dim_1_B, dim_2_B, dim_3_B = input_kv_cache_shape
        dim_0_o, dim_1_o, dim_2_o, dim_3_o = output_shape

        # assert dim_0_seq == dim_0_B == dim_0_o
        # assert dim_1_seq == dim_1_B == dim_1_o

        # if dim == 2:
        #     assert dim_2_seq == 1
        #     assert dim_2_seq + dim_2_B == dim_2_o
        #     assert dim_3_seq == dim_3_B == dim_3_o
        # elif dim == 3:
        #     assert dim_3_seq == 1
        #     assert dim_2_seq == dim_2_B == dim_2_o
        #     assert dim_3_seq + dim_3_B == dim_3_o
        # else:
        #     raise ValueError

        input_list = list()
        input_list.append(ActivationInfo(input_seq_name, input_seq_shape, input_kv_cache_dtype, "sequence",keepdim=True))
        
        kv_cache_act = ActivationInfo(input_kv_cache_name, input_kv_cache_shape, input_kv_cache_dtype, data_type,keepdim=True)
        kv_cache_act.add_data_tag(input_kv_cache_tag)
        input_list.append(kv_cache_act)
        
        output_list = list()
        new_kv_cache_act = ActivationInfo(output_name, output_shape, output_dtype, data_type,keepdim=True)
        new_kv_cache_act.add_data_tag(input_kv_cache_tag)
        output_list.append(new_kv_cache_act)
        
        self.all_activation_list.add(new_kv_cache_act)

        param_list = list()
        structure_dict = {
            "dim": dim - 1,
        }
        self._set_layer_def(layer_name, "concat", input_list, output_list, param_list, "memory", structure_dict)

    def rename_tensor(
        self,
        old_name: str,
        new_name: str,
    ):
        #assert new_name != old_name
        #assert old_name in self.all_tensor_usage.keys()
        new_name_exist_flag = new_name in self.all_tensor_usage.keys()
        prev_new_name_cnt = self.all_tensor_usage[new_name] if new_name_exist_flag else 0  # 如果new_name原来就存在，那么需要把new_name的usage加到old_name上
        old_name_cnt = self.all_tensor_usage.pop(old_name)
        change_cnt = 0
        output_flag = False
        for layer in self.ir:
            for each_input in layer["input"]:
                if each_input["name"] == old_name:
                    if not new_name_exist_flag:
                        assert not output_flag
                    each_input["name"] = new_name
                    change_cnt += 1
            for each_output in layer["output"]:
                #assert each_output["name"] != new_name or layer["type"] == "input", layer  # 不能已经是某层的输出
                if each_output["name"] == old_name:
                    each_output["name"] = new_name
                    output_flag = True
        #assert output_flag
        if new_name_exist_flag:
            assert old_name_cnt == change_cnt + 1  # + 1 是因为原来transpose层用了一次，但是现在删掉了
        self.all_tensor_usage[new_name] = change_cnt + prev_new_name_cnt

    def remove_transpose_layer(self):
        layer_id = 0
        while layer_id < len(self.ir):
            if self.ir[layer_id]["type"] == "transpose":
                transpose_layer = self.ir[layer_id]
                #assert len(transpose_layer["input"]) == 1
                #assert len(transpose_layer["output"]) == 1
                dim_0, dim_1 = transpose_layer["structure"]["dim"]
                #assert dim_1 == dim_0 + 1  # 只支持相邻维度的transpose
                #assert transpose_layer["input"][0]["shape"][dim_0] == transpose_layer["output"][0]["shape"][dim_1]
                #assert transpose_layer["input"][0]["shape"][dim_1] == transpose_layer["output"][0]["shape"][dim_0]
                #assert np.prod(transpose_layer["input"][0]["shape"]) == np.prod(transpose_layer["output"][0]["shape"])
                if transpose_layer["input"][0]["shape"][dim_0] == 1 or transpose_layer["input"][0]["shape"][dim_1] == 1:  # 不改变数据排布，不需要transpose
                    self.ir.pop(layer_id)
                    self.rename_tensor(transpose_layer["input"][0]["name"], transpose_layer["output"][0]["name"])
                    continue
                else:
                    layer_id += 1
            else:
                layer_id += 1

    def check_ir(self):
        # 各层层内参数的正确性由注册层ir时保证，本函数负责检查层间连接关系的正确性
        # 由于考虑了input层和output层，所以每个tensor的出现次数应>=1
        # 即每个tensor的名称应在output中仅出现1次，在input中出现>=1次
        check_all_tensor_usage = dict()
        all_tensor_dtype = dict()
        all_tensor_shape = dict()
        #assert self.ir[0]["type"] == "input"
        #assert self.ir[-1]["type"] == "output"

        for layer in self.ir[1:-1]:
            assert layer["type"] not in ("input", "output")

        for layer in self.ir:
            #assert layer["type"] in ALL_HW_LAYER_TYPE, layer["type"]
            for each_input in layer["input"]:
                #assert each_input["name"] in check_all_tensor_usage.keys()
                #assert each_input["name"] in all_tensor_dtype.keys()
                check_all_tensor_usage[each_input["name"]] += 1
                #assert each_input["dtype"] == all_tensor_dtype[each_input["name"]]
                # 检查数据排布是否相同
                if each_input["shape"] != all_tensor_shape[each_input["name"]]:  # 若形状不同，则数据排布必须相同
                    squeeze_input_shape = [i for i in each_input["shape"] if i != 1]  # 删掉所有为1的维度
                    squeeze_record_shape = [i for i in all_tensor_shape[each_input["name"]] if i != 1]
                    if len(squeeze_input_shape) == len(squeeze_record_shape):  # 如果长度一样，那么形状必须相同
                        assert squeeze_input_shape == squeeze_record_shape
                    else:
                        # example: (32, 128) -> (4096)
                        # #assert len(squeeze_input_shape) == 1 or len(squeeze_record_shape) == 1
                        assert np.prod(squeeze_input_shape) == np.prod(squeeze_record_shape)

            for each_output in layer["output"]:
                #assert each_output["name"] not in check_all_tensor_usage.keys()
                #assert each_output["name"] not in all_tensor_dtype.keys()
                check_all_tensor_usage[each_output["name"]] = 0
                all_tensor_dtype[each_output["name"]] = each_output["dtype"]
                all_tensor_shape[each_output["name"]] = each_output["shape"]

        if len(self.ir) == 0:
            raise ValueError("IR is empty")
        for key, value in check_all_tensor_usage.items():
            #assert key in self.all_tensor_usage.keys()
            # value 可能不等于 self.all_tensor_usage[key]，因为只取了decode部分的IR
            if value < 1:
                raise ValueError("Tensor {} is not used in IR".format(key))
        return True
