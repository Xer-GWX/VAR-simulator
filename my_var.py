import torch
from torch import nn
import numpy as np
from loguru import logger
from typing import List, Optional
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ir_generator import IRGenerator
from functools import partial

# define parameter
# WORD_EMBED_PROJ_DIM = 1024
# VOCAB_SIZE = 28996  # 30522
# PADDING_IDX = 1
# MAX_POSITION_EMBEDDINGS = 512
# HIDDEN_SIZE = 1024
# EMBED_DIM = 1024
# FFN_DIM = 4096
# NUM_HEADS = 16
# NUM_KEY_VALUE_HEADS = 16
# HEAD_DIM = EMBED_DIM // NUM_HEADS  # 64
# KEY_VALUE_GROUP = NUM_HEADS // NUM_KEY_VALUE_HEADS
# SCALING = HEAD_DIM ** -0.5
# NUM_HIDDEN_LAYERS = 1  # 24 layers

# BERT_MAX_LEN = 512
# TYPE_VOCAB_SIZE = 2
# ATTENTION_MASK_BLOCK_SIZE = 16
# define parameter
CVAE = 32
EMBED_DIM=1024
STAGE_TOTAL = 10
HEAD_NUM = 16
HEAD_DIM = int(EMBED_DIM / HEAD_NUM)
W = H = 16
L = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, padding_idx=PADDING_IDX)
        self.position_embeddings = nn.Embedding(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(TYPE_VOCAB_SIZE, HIDDEN_SIZE)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = "absolute"
        self.register_buffer(
            "position_ids", torch.arange(MAX_POSITION_EMBEDDINGS).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        return embeddings


class MyVAR:
    def __init__(self,
                 model_ir_generator: IRGenerator,
                 NORM_VARIANCE_EPS: float = 1e-06,) -> None:
        logger.info("Init VARModel")
        #self.call_forward_count = 0
        #self.now_token_id = None
        self.generator = model_ir_generator
        self.cached_k, self.cached_v = None, None
        
        #self.pool = pool

        # define layers
        #self.embed_tokens_layer = BertEmbeddings()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.f_hat = torch.zeros((8,CVAE,W,H)).to(device)
        self.word_embed = nn.Linear(CVAE,EMBED_DIM,bias=True)
        #self.embed_layer_norm = nn.LayerNorm(EMBED_DIM, eps=NORM_VARIANCE_EPS,)
        self.ada_lin_silu = nn.SiLU().to(device)
        self.ada_lin_linear = nn.Linear(EMBED_DIM,EMBED_DIM*6,bias=True).to(device)
        
        norm_layer = partial(nn.LayerNorm, eps=NORM_VARIANCE_EPS)
        self.ln_wo_grad = norm_layer(EMBED_DIM,elementwise_affine=False)
 
        self.mat_qkv_linear = nn.Linear(EMBED_DIM,EMBED_DIM*3,bias=False).to(device)
        self.attn_proj = nn.Linear(EMBED_DIM,EMBED_DIM,bias=True).to(device)
        self.fc1_linear = nn.Linear(EMBED_DIM,EMBED_DIM*4,bias=True).to(device)
        self.act_fn_gelu = nn.GELU().to(device)
        self.fc2_linear = nn.Linear(EMBED_DIM*4,EMBED_DIM,bias=True).to(device)
        self.head_nm_linear = nn.Linear(EMBED_DIM,EMBED_DIM*2,bias=True).to(device)
        self.head_linear = nn.Linear(EMBED_DIM,EMBED_DIM*4,bias=True).to(device)

        self.Phi_conv2d = nn.Conv2d(CVAE, CVAE, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device)
        self.word_embed_linear = nn.Linear(CVAE,EMBED_DIM,bias=True).to(device)
        logger.info("load VAR weight done")

    
    @staticmethod
    def get_name_prefix(idx):
        return f"stage{idx}_"

    @staticmethod
    def get_name_suffix(idx):
        return f"_block{idx}"

    def forward(self,
                input_ids,
                cond_BD,
                ):
        logger.info(f"Encode stage")#, input token num: {input_ids.shape[-1]}
        
        """
        ### BERT-Large preprocess ###
        """
        # causal_attention_mask, hidden_states_tem = self.forward_preprocess(input_ids, attention_mask, past_key_values)
        # name_prefix = self.get_name_prefix(self.call_forward_count)#round_0_
        # next_round_name_prefix = self.get_name_prefix(self.call_forward_count + 1)
        # next_decoder_cache = ()
        # vector_flag = (hidden_states_tem.shape[1] == 1)

        # self.generator.input_layer(
        #     # name
        #     layer_name=name_prefix + "input_layer",
        #     # input act
        #     input_act_name=name_prefix + "hidden_states_before_layer_norm",
        #     input_act_data=hidden_states_tem,
        #     input_act_dtype="float16",
        #     # k cache list
        #     k_cache_name_list=list(),
        #     k_cache_tag_list=list(),
        #     k_cache_data_list=list(),
        #     k_cache_dtype_list=list(),
        #     k_cache_data_type="k_cache",
        #     # v cache list
        #     v_cache_name_list=list(),
        #     v_cache_tag_list=list(),
        #     v_cache_data_list=list(),
        #     v_cache_dtype_list=list(),
        #     v_cache_data_type="v_cache_t",
        # )

        # hidden_states = self.embed_layer_norm(hidden_states_tem)

        for stage_num in range(STAGE_TOTAL):
            name_prefix = self.get_name_prefix(stage_num)
            for idx in range(HEAD_NUM):
                name_suffix = self.get_name_suffix(idx)
                B = input_ids.shape[0]; L_s = input_ids.shape[1]
                hidden_silu = self.ada_lin_silu(cond_BD)
                self.generator.silu_layer(
                    # name
                    layer_name          = name_prefix + "ada_lin_silu" + name_suffix,
                    # input
                    input_name          = name_prefix + "cond_BD" + name_suffix,
                    input_shape         = cond_BD.shape,
                    # output
                    output_name         = name_prefix + "hidden_silu" + name_suffix,
                    output_shape        = hidden_silu.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "silu_layer_" + name_suffix,
                    # structure
                    granularity         = "element",
                )
                ada_lin_output = self.ada_lin_linear(hidden_silu)

                self.generator.linear_layer(
                    # name
                    layer_name          = name_prefix + "ada_lin_output" + name_suffix,
                    # input
                    input_name          = name_prefix + "hidden_silu" + name_suffix,
                    input_shape         = hidden_silu.shape,
                    # output
                    output_name         = name_prefix + "ada_lin_output" + name_suffix,
                    output_shape        = ada_lin_output.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "ada_lin_output_layer_" + name_suffix,
                    weight              = self.ada_lin_linear.weight,
                    bias                = self.ada_lin_linear.bias,
                    # structure
                    relu_flag           = False,
                    granularity         = "matrix" #"vector" if vector_flag else "matrix",
                )
                
                hidden_ln = self.ln_wo_grad(input_ids)
                self.generator.layernorm_layer(
                # name
                layer_name          = name_prefix + "input_layer_norm_layer" + name_suffix,
                # input
                input_name          = name_prefix + "input" + name_suffix,
                input_shape         = input_ids.shape,
                # output
                output_name         = name_prefix + "hidden_layernorm" + name_suffix,
                output_shape        = hidden_ln.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "input_layer_norm_layer_" + name_suffix,
                weight              = self.ln_wo_grad.weight,
                bias                = self.ln_wo_grad.bias,
                # structure
                rms_flag            = False,
                granularity         = "vector",
            )
                
                # [16,1920]
                gamma1, gamma2, scale1, scale2, shift1, shift2 = ada_lin_output.view(-1, 1, 6, EMBED_DIM).unbind(2)
                hidden_scale1_mul = hidden_ln.mul(scale1)
                self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "hidden_mul_scale1_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "hidden_layernorm" + name_suffix,
                input_A_shape       = hidden_ln.shape,
                input_B_name        = name_prefix + "scale1" + name_suffix,
                input_B_shape       = scale1.shape,
                # output
                output_name         = name_prefix + "hidden_scale1_mul" + name_suffix,
                output_shape        = hidden_scale1_mul.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "hidden_mul_scale1_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwisemul",
                granularity         = "element",
            )
                hidden_shift1_add = hidden_scale1_mul + shift1
                self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "hidden_add_shift1_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "hidden_scale1_mul" + name_suffix,
                input_A_shape       = hidden_scale1_mul.shape,
                input_B_name        = name_prefix + "shift1" + name_suffix,
                input_B_shape       = shift1.shape,
                # output
                output_name         = name_prefix + "hidden_shift1_add" + name_suffix,
                output_shape        = hidden_shift1_add.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "hidden_add_shift1_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwiseadd",
                granularity         = "element",
            )
                

                qkv = self.mat_qkv_linear(hidden_shift1_add)
                self.generator.linear_layer(
                    # name
                    layer_name          = name_prefix + "qkv_linear" + name_suffix,
                    # input
                    input_name          = name_prefix + "hidden_shift1_add" + name_suffix,
                    input_shape         = input_ids.shape,
                    # output
                    output_name         = name_prefix + "qkv" + name_suffix,
                    output_shape        = qkv.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "qkv_linear_output_layer_" + name_suffix,
                    weight              = self.mat_qkv_linear.weight,
                    bias                = self.mat_qkv_linear.bias,
                    # structure
                    relu_flag           = False,
                    granularity         = "matrix" #"vector" if vector_flag else "matrix",
                )
                
                qkv = qkv.view(B, L_s, 3, HEAD_NUM, HEAD_DIM)
                q,k,v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        # qkv: BL3Hc  16 1(L) 3 16 64
                #k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
                
                if idx == 0:
                    if self.cached_k is None:  k_concat = self.cached_k = k; v_concat = self.cached_v = v
                    else: k_concat = torch.cat((self.cached_k, k), dim=2); v_concat = torch.cat((self.cached_v, v), dim=2)
                if stage_num == 0:
                    pass
                else:
                    self.generator.concat_layer(
                    # name
                    layer_name              = name_prefix + "concat_cached_k" + name_suffix,
                    # input
                    input_seq_name          = name_prefix + "k" + name_suffix,
                    input_seq_shape         = k.shape,
                    input_seq_dtype         = "float16",
                    input_kv_cache_name     = name_prefix + "cached_k" + name_suffix,
                    input_kv_cache_tag      = "k_cache" + name_suffix,
                    input_kv_cache_shape    = self.cached_k.shape,
                    input_kv_cache_dtype    = "float16",
                    # output
                    output_name             = name_prefix + "k_concat" + name_suffix, # for next round
                    output_shape            = k_concat.shape,
                    output_dtype            = "float16",
                    # structure
                    data_type               = "k_cache",
                    dim                     = 2,
                    granularity             = None,
                )
                
                    self.generator.concat_layer(
                    # name
                    layer_name              = name_prefix + "concat_cached_v" + name_suffix,
                    # input
                    input_seq_name          = name_prefix + "v" + name_suffix,
                    input_seq_shape         = v.shape,
                    input_seq_dtype         = "float16",
                    input_kv_cache_name     = name_prefix + "cached_v" + name_suffix,
                    input_kv_cache_tag      = "v_cache" + name_suffix,
                    input_kv_cache_shape    = self.cached_v.shape,
                    input_kv_cache_dtype    = "float16",
                    # output
                    output_name             = name_prefix + "v_concat"+ name_suffix,  # for next round
                    output_shape            = v_concat.shape,
                    output_dtype            = "float16",
                    # structure
                    data_type               = "v_cache",
                    dim                     = 2,
                    granularity             = None,
                )
                
                k_concat_t = k_concat.transpose(-2, -1)
                self.generator.transpose_layer(
                    # name
                    layer_name          = name_prefix + "k_concat_transpose" + name_suffix,
                    # input
                    input_name          = name_prefix + "k_concat" + name_suffix,
                    input_tag           = None,
                    input_shape         = k_concat.shape,
                    input_dtype         = "float16",
                    input_data_type     = "sequence",
                    # output
                    output_name         = name_prefix + "k_concat_t" + name_suffix,
                    output_shape        = k_concat_t.shape,
                    output_data_type    = "sequence",
                    # structure
                    dim                 = [1, 2],
                    dim_trans           = "down",
                    granularity         = None,
                )
                qkt_mm = q @ k_concat_t
                self.generator.attention_layer(
                    # name
                    layer_name          = name_prefix + "attention_qkt" + name_suffix,
                    # input
                    input_A_name        = name_prefix + "q" + name_suffix,
                    input_A_shape       = q.shape,
                    input_B_name        = name_prefix + "k_concat_t" + name_suffix,
                    # input_B_tag=None,
                    input_B_tag         = "k_cache" + name_suffix,
                    input_B_shape       = k_concat_t.shape,  # 硬件的输入形状和torch MM输入形状不一致
                    # output
                    output_name         = name_prefix + "qkt_mm" + name_suffix,
                    output_shape        = qkt_mm.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "attention_qkt" + name_suffix,
                    # structure
                    mask_id             = idx,
                    mask_mode           = "qkt",
                    granularity         = "matrix"#"vector" if vector_flag else "matrix",
                )
                qkt_softmax = qkt_mm.softmax(dim=-1)
                #qkt_softmax = nn.functional.softmax(qkt_mm, dim=-1, dtype=torch.float32)  # .to(torch.float16)
                self.generator.softmax_layer(
                    # name
                    layer_name          = name_prefix + "softmax" + name_suffix,
                    # input
                    input_name          = name_prefix + "qkt_mm" + name_suffix,
                    input_shape         = qkt_mm.shape,
                    # output
                    output_name         = name_prefix + "qkt_softmax" + name_suffix,
                    output_shape        = qkt_softmax.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "_softmax_" + name_suffix,
                    # structure
                    dim                 = 2,
                    granularity         = "vector",
                )
                qktv_mm = qkt_softmax @ v_concat
                self.generator.attention_layer(
                    # name
                    layer_name          = name_prefix + "attention_qktv" + name_suffix,
                    # input
                    input_A_name        = name_prefix + "qkt_softmax" + name_suffix,
                    input_A_shape       = qkt_softmax.shape,
                    input_B_name        = name_prefix + "v_concat" + name_suffix,
                    input_B_tag         = "v_cache" + name_suffix,
                    input_B_shape       = v_concat.shape,  # 硬件的输入形状和torch MM输入形状不一致
                    # output
                    output_name         = name_prefix + "_qktv_mm" + name_suffix,
                    output_shape        = qktv_mm.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "attention_qktv" + name_suffix,
                    # structure
                    mask_id             = idx,
                    mask_mode           = "qktv",
                    granularity         = "matrix"#"vector" if vector_flag else "matrix",
                )
                qktv_mm = qktv_mm.view(B,L_s,EMBED_DIM)
                attn_output = self.attn_proj(qktv_mm)
                self.generator.linear_layer(
                    # name
                    layer_name          = name_prefix + "out_proj_layer" + name_suffix,
                    # input
                    input_name          = name_prefix + "qktv_mm" + name_suffix,
                    input_shape         = qktv_mm.shape,
                    # output
                    output_name         = name_prefix + "attn_output" + name_suffix,
                    output_shape        = attn_output.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "out_proj_layer" + name_suffix,
                    weight              = self.attn_proj.weight,
                    bias                = self.attn_proj.bias,
                    # structure
                    relu_flag           = False,
                    granularity         = "matrix"#"vector" if vector_flag else "matrix",
                )
                hidden_gamma1_mul = attn_output.mul(gamma1)
                self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "hidden_mul_gamma1_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "attn_output" + name_suffix,
                input_A_shape       = attn_output.shape,
                input_B_name        = name_prefix + "gamma1" + name_suffix,
                input_B_shape       = gamma1.shape,
                # output
                output_name         = name_prefix + "hidden_gamma1_mul" + name_suffix,
                output_shape        = hidden_gamma1_mul.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "hidden_mul_gamma1_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwisemul",
                granularity         = "element",
            )
                input_add_attnout = input_ids + hidden_gamma1_mul
                self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "hidden_add_attn_res_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "input" + name_suffix,
                input_A_shape       = input_ids.shape,
                input_B_name        = name_prefix + "hidden_gamma1_mul" + name_suffix,
                input_B_shape       = hidden_gamma1_mul.shape,
                # output
                output_name         = name_prefix + "input_add_attnout" + name_suffix,
                output_shape        = input_add_attnout.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "hidden_add_attn_res_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwiseadd",
                granularity         = "element",
            )
                
                hidden_ln_ffc = self.ln_wo_grad(input_add_attnout)
                self.generator.layernorm_layer(
                # name
                layer_name          = name_prefix + "input_add_attnout_layer_norm_layer" + name_suffix,
                # input
                input_name          = name_prefix + "input_add_attnout" + name_suffix,
                input_shape         = input_add_attnout.shape,
                # output
                output_name         = name_prefix + "hidden_layernorm_ffc" + name_suffix,
                output_shape        = hidden_ln_ffc.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "input_add_attnout_layer_norm_layer_" + name_suffix,
                weight              = self.ln_wo_grad.weight,
                bias                = self.ln_wo_grad.bias,
                # structure
                rms_flag            = False,
                granularity         = "vector",
            )
                
                hidden_scale2_mul = hidden_ln_ffc.mul(scale2)
                self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "hidden_mul_scale2_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "hidden_layernorm_ffc" + name_suffix,
                input_A_shape       = hidden_ln_ffc.shape,
                input_B_name        = name_prefix + "scale2" + name_suffix,
                input_B_shape       = scale2.shape,
                # output
                output_name         = name_prefix + "hidden_scale2_mul" + name_suffix,
                output_shape        = hidden_scale2_mul.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "hidden_mul_scale2_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwisemul",
                granularity         = "element",
            )
                hidden_shift2_add = hidden_scale2_mul + shift2
                self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "hidden_add_shift2_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "hidden_scale2_mul" + name_suffix,
                input_A_shape       = hidden_scale1_mul.shape,
                input_B_name        = name_prefix + "shift2" + name_suffix,
                input_B_shape       = shift1.shape,
                # output
                output_name         = name_prefix + "hidden_shift2_add" + name_suffix,
                output_shape        = hidden_shift2_add.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "hidden_add_shift2_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwiseadd",
                granularity         = "element",
            )
                fc1 = self.fc1_linear(hidden_shift2_add)
                self.generator.linear_layer(
                    # name
                    layer_name          = name_prefix + "fc1_layer" + name_suffix,
                    # input
                    input_name          = name_prefix + "input_add_attnout" + name_suffix,
                    input_shape         = input_add_attnout.shape,
                    # output
                    output_name         = name_prefix + "fc1" + name_suffix,
                    output_shape        = fc1.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "_fc1_layer_" + name_suffix,
                    weight              = self.fc1_linear.weight,
                    bias                = self.fc1_linear.bias,
                    # structure
                    relu_flag           = False,
                    granularity         = "matrix" #"vector" if vector_flag else "matrix",
                )
                fc1_gelu = self.act_fn_gelu(fc1)
                self.generator.gelu_layer(
                    # name
                    layer_name          = name_prefix + "activation_fn_layer" + name_suffix,
                    # input
                    input_name          = name_prefix + "fc1" + name_suffix,
                    input_shape         = fc1.shape,
                    # output
                    output_name         = name_prefix + "fc1_gelu" + name_suffix,
                    output_shape        = fc1_gelu.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = str(stage_num) + "_gelu_layer_" + str(idx),
                    # structure
                    granularity         = "element",
                )
                fc2 = self.fc2_linear(fc1_gelu)
                self.generator.linear_layer(
                    # name
                    layer_name          = name_prefix + "fc2_layer" + name_suffix,
                    # input
                    input_name          = name_prefix + "fc1_gelu" + name_suffix,
                    input_shape         = fc1_gelu.shape,
                    # output
                    output_name         = name_prefix + "fc2" + name_suffix,
                    output_shape        = fc2.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = str(stage_num) + "_fc2_layer_" + str(idx),
                    weight              = self.fc2_linear.weight,
                    bias                = self.fc2_linear.bias,
                    # structure
                    relu_flag           = False,
                    granularity         = "matrix" #"vector" if vector_flag else "matrix",
                )
                hidden_gamma2_mul = fc2.mul(gamma2)
                self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "hidden_mul_gamma2_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "fc2" + name_suffix,
                input_A_shape       = attn_output.shape,
                input_B_name        = name_prefix + "gamma2" + name_suffix,
                input_B_shape       = gamma1.shape,
                # output
                output_name         = name_prefix + "hidden_gamma2_mul" + name_suffix,
                output_shape        = hidden_gamma2_mul.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "hidden_mul_gamma2_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwisemul",
                granularity         = "element",
            )
                input_add_ffnout = input_add_attnout + hidden_gamma2_mul
                self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "hidden_add_ffn_res_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "input_add_attnout" + name_suffix,
                input_A_shape       = input_ids.shape,
                input_B_name        = name_prefix + "hidden_gamma2_mul" + name_suffix,
                input_B_shape       = hidden_gamma1_mul.shape,
                # output
                output_name         = name_prefix + "input_add_ffnout" + name_suffix,
                output_shape        = input_add_ffnout.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "hidden_add_ffn_res_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwiseadd",
                granularity         = "element",
            )
                
            self.cached_k = k_concat; self.cached_v = v_concat
            head_nm = self.head_nm_linear(fc2)
            self.generator.linear_layer(
                    # name
                    layer_name          = name_prefix + "head_nm_layer" + name_suffix,
                    # input
                    input_name          = name_prefix + "fc2" + name_suffix,
                    input_shape         = fc2.shape,
                    # output
                    output_name         = name_prefix + "head_nm" + name_suffix,
                    output_shape        = head_nm.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = str(stage_num) + "_head_nm_layer_" + str(idx),
                    weight              = self.head_nm_linear.weight,
                    bias                = self.head_nm_linear.bias,
                    # structure
                    relu_flag           = False,
                    granularity         = "matrix" #"vector" if vector_flag else "matrix",
                )
            head = self.head_linear(fc2)
            self.generator.linear_layer(
                    # name
                    layer_name          = name_prefix + "head_layer" + name_suffix,
                    # input
                    input_name          = name_prefix + "fc2" + name_suffix,
                    input_shape         = fc2.shape,
                    # output
                    output_name         = name_prefix + "head" + name_suffix,
                    output_shape        = head.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "_head_layer_" + name_suffix,
                    weight              = self.head_linear.weight,
                    bias                = self.head_linear.bias,
                    # structure
                    relu_flag           = False,
                    granularity         = "matrix" #"vector" if vector_flag else "matrix",
                )
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # 采样 embedding transpose interpolate 得到Phi_input
            Phi_input = torch.zeros((int(B/2),CVAE,W,H)).to(device)
            Phi_output = self.Phi_conv2d(Phi_input)
            self.generator.conv_layer(
                # name
                layer_name = name_prefix + "Phi_conv2d_layer" + name_suffix,
                # input
                input_name = name_prefix + "Phi_input" + name_suffix,
                input_shape = Phi_input.shape,
                # output
                output_name = name_prefix + "Phi_output" + name_suffix,
                output_shape = Phi_output.shape,
                output_dtype = "float16",
                # param
                param_name_prefix = name_prefix + "_Phi_conv2d_layer_" + name_suffix,
                kernel = self.Phi_conv2d.weight,
                bias = None ,#self.Phi_conv2d.bias,  # may be None
                # structure
                stride = self.Phi_conv2d.stride,
                padding = self.Phi_conv2d.padding,
                relu_flag = False,
    )
            self.f_hat = self.f_hat + Phi_output
            self.generator.eltwise_layer(
                # name
                layer_name          = name_prefix + "f_hat_add_layer" + name_suffix,
                # input
                input_A_name        = name_prefix + "f_hat" + name_suffix,
                input_A_shape       = self.f_hat.shape,
                input_B_name        = name_prefix + "Phi_output" + name_suffix,
                input_B_shape       = Phi_output.shape,
                # output
                output_name         = name_prefix + "f_hat" + name_suffix,
                output_shape        = self.f_hat.shape,
                output_dtype        = "float16",
                # param
                param_name_prefix   = name_prefix + "f_hat_add_layer_" + name_suffix,
                # structure
                eltwise_type        = "eltwiseadd",
                granularity         = "element",
            )
            # stage10的self.f_hat是最后的输出output_layer
            if stage_num+1 < 10:
                word_embed_input = torch.zeros((B,L[stage_num+1],CVAE)).to(device)
                word_embed_output = self.word_embed_linear(word_embed_input)
                self.generator.linear_layer(
                    # name
                    layer_name          = name_prefix + "word_embed_layer" + name_suffix,
                    # input
                    input_name          = name_prefix + "word_embed_input" + name_suffix,
                    input_shape         = word_embed_input.shape,
                    # output
                    output_name         = name_prefix + "word_embed_output" + name_suffix,
                    output_shape        = word_embed_output.shape,
                    output_dtype        = "float16",
                    # param
                    param_name_prefix   = name_prefix + "_word_embed_layer_" + name_suffix,
                    weight              = self.word_embed_linear.weight,
                    bias                = self.word_embed_linear.bias,
                    # structure
                    relu_flag           = False,
                    granularity         = "matrix" #"vector" if vector_flag else "matrix",
                )
            input_ids = word_embed_output

        #         # past_key_value = past_key_values[idx] if past_key_values is not None else None
        #         # if idx != 0:
        #         #     self.generator.rename_tensor(
        #         #         old_name        = name_prefix + "final_hidden_states" + self.get_name_suffix(idx - 1),
        #         #         new_name        = name_prefix + "hidden_states" + name_suffix,
        #         #     )
        #         #     self.generator.rename_tensor(
        #         #         old_name        = name_prefix + "final_hidden_states_FP" + self.get_name_suffix(idx - 1),
        #         #         new_name        = name_prefix + "hidden_states_FP" + name_suffix,
        #         #     )
        #         # else:
        #             # self.generator.rename_tensor(
        #             #     old_name        = name_prefix + "hidden_states_before_layer_norm",
        #             #     new_name        = name_prefix + "hidden_states" + name_suffix,
        #             # )

        #         # layer_head_mask=(head_mask[idx] if head_mask is not None else None) 没用到
        #         """
        #         ### start BERT-Large forward ###
        #         """
        #         #bsz, tgt_len, _ = hidden_states.size()  # [batch_size, seq_len, hidden_size]

        #         q_proj_out = self.q_proj_layer_list[idx](hidden_states)
        #         self.generator.linear_layer(
        #             # name
        #             layer_name          = name_prefix + "q_proj_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "hidden_states" + name_suffix,
        #             input_shape         = hidden_states.shape,
        #             # output
        #             output_name         = name_prefix + "q_proj_out" + name_suffix,
        #             output_shape        = q_proj_out.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "q_proj_layer_" + str(idx),
        #             weight              = self.q_proj_layer_list[idx].weight,
        #             bias                = self.q_proj_layer_list[idx].weight,
        #             # structure
        #             relu_flag=False,
        #             granularity         = "vector" if vector_flag else "matrix",
        #         )
        #         # q_proj_out = q_proj_out * SCALING
        #         q_proj_out = q_proj_out.view(bsz, tgt_len, NUM_HEADS, HEAD_DIM)
        #         q_proj_out_t = q_proj_out.transpose(1, 2).contiguous()

        #         self.generator.transpose_layer(
        #             # name
        #             layer_name          = name_prefix + "q_transpose" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "q_proj_out" + name_suffix,
        #             input_tag           = None,
        #             input_shape         = q_proj_out.shape,
        #             input_dtype         = "float16",
        #             input_data_type     = "sequence",
        #             # output
        #             output_name         = name_prefix + "q_proj_out_t" + name_suffix,
        #             output_shape        = q_proj_out_t.shape,
        #             output_data_type    = "sequence",
        #             # structure
        #             dim                 = [1, 2],
        #             dim_trans           = "down",
        #             granularity         = None,
        #         )

        #         k_proj_out = self.k_proj_layer_list[idx](hidden_states)
        #         self.generator.linear_layer(
        #             # name
        #             layer_name          = name_prefix + "k_proj_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "hidden_states" + name_suffix,
        #             input_shape         = hidden_states.shape,
        #             # output
        #             output_name         = name_prefix + "k_proj_out" + name_suffix,
        #             output_shape        = k_proj_out.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "k_proj_layer_" + str(idx),
        #             weight              = self.k_proj_layer_list[idx].weight,
        #             bias                = self.k_proj_layer_list[idx].bias,
        #             # structure
        #             relu_flag           = False,
        #             granularity         = "vector" if vector_flag else "matrix",
        #         )
        #         k_proj_out = k_proj_out.view(bsz, tgt_len, NUM_HEADS, HEAD_DIM)
        #         k_proj_out_t = k_proj_out.transpose(1, 2).contiguous()
        #         self.generator.transpose_layer(
        #             # name
        #             layer_name          = name_prefix + "k_transpose" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "k_proj_out" + name_suffix,
        #             input_tag           = None,
        #             input_shape         = k_proj_out.shape,
        #             input_dtype         = "float16",
        #             input_data_type     = "sequence",
        #             # output
        #             output_name         = name_prefix + "k_proj_out_t" + name_suffix,
        #             output_shape        = k_proj_out_t.shape,
        #             output_data_type    = "sequence",
        #             # structure
        #             dim                 = [1, 2],
        #             dim_trans           = "down",
        #             granularity         = None,
        #         )

        #         v_proj_out = self.v_proj_layer_list[idx](hidden_states)
        #         self.generator.linear_layer(
        #             # name
        #             layer_name          = name_prefix + "v_proj_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "hidden_states" + name_suffix,
        #             input_shape         = hidden_states.shape,
        #             # output
        #             output_name         = name_prefix + "v_proj_out" + name_suffix,
        #             output_shape        = v_proj_out.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "v_proj_layer_" + str(idx),
        #             weight              = self.v_proj_layer_list[idx].weight,
        #             bias                = self.v_proj_layer_list[idx].bias,
        #             # structure
        #             relu_flag           = False,
        #             granularity         = "vector" if vector_flag else "matrix",
        #         )
        #         v_proj_out = v_proj_out.view(bsz, tgt_len, NUM_HEADS, HEAD_DIM)
        #         v_proj_out_t = v_proj_out.transpose(1, 2).contiguous()
        #         self.generator.transpose_layer(
        #             # name
        #             layer_name          = name_prefix + "transpose_v" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "v_proj_out" + name_suffix,
        #             input_tag           = None,
        #             input_shape         = v_proj_out.shape,
        #             input_dtype         = "float16",
        #             input_data_type     = "sequence",
        #             # output
        #             output_name         = name_prefix + "v_proj_out_t" + name_suffix,
        #             output_shape        = v_proj_out_t.shape,
        #             output_data_type    = "sequence",
        #             # structure
        #             dim                 = [1, 2],
        #             dim_trans           = "down",
        #             granularity         = None,
        #         )
        #         """
        #             ### BERT-Large KV Cache ###
        #         """
        #         if past_key_value is not None:
        #             raise NotImplementedError("KV Cache is not implemented yet!")
        #         else:
        #             self.generator.rename_tensor(
        #                 old_name        = name_prefix + "k_proj_out_t" + name_suffix,
        #                 new_name        = next_round_name_prefix + "past_keys" + name_suffix,
        #             )
        #             self.generator.rename_tensor(
        #                 old_name        = name_prefix + "v_proj_out_t" + name_suffix,
        #                 new_name        = next_round_name_prefix + "past_values_t" + name_suffix,
        #             )
        #             past_key_value = (k_proj_out_t, v_proj_out_t)

        #             key = k_proj_out_t.view(bsz * NUM_HEADS, -1, HEAD_DIM)
        #             value = v_proj_out_t.view(bsz * NUM_HEADS, -1, HEAD_DIM)

        #         query = q_proj_out_t.view(bsz * NUM_HEADS, -1, HEAD_DIM)

        #         src_len = key.size(1)
        #         key_t = key.transpose(1, 2)

        #         self.generator.transpose_layer(
        #             # name
        #             layer_name          = name_prefix + "k_cache_transpose" + name_suffix,
        #             # input
        #             input_name          = next_round_name_prefix + "past_keys" + name_suffix,
        #             input_tag           = "k_cache" + name_suffix,
        #             input_shape         = value.shape,
        #             input_dtype         = "float16",
        #             input_data_type     = "k_cache",
        #             # output
        #             output_name         = name_prefix + "past_keys_ready" + name_suffix,
        #             output_shape        = value.transpose(1, 2).shape,  # 硬件的输入形状和torch MM输入形状不一致
        #             output_data_type    = "k_cache_t",
        #             # structure
        #             dim                 = [1, 2],
        #             dim_trans           = "keep",
        #             granularity         = None,
        #         )

        #         qkt_mm = torch.bmm(query, key_t)
        #         self.generator.attention_layer(
        #             # name
        #             layer_name          = name_prefix + "attention_qkt" + name_suffix,
        #             # input
        #             input_A_name        = name_prefix + "q_proj_out_t" + name_suffix,
        #             input_A_shape       = query.shape,
        #             input_B_name        = name_prefix + "past_keys_ready" + name_suffix,
        #             # input_B_tag=None,
        #             input_B_tag         = "k_cache" + name_suffix,
        #             input_B_shape       = key_t.transpose(1, 2).shape,  # 硬件的输入形状和torch MM输入形状不一致
        #             # output
        #             output_name         = name_prefix + "qkt_mm" + name_suffix,
        #             output_shape        = qkt_mm.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "attention_qkt_" + str(idx),
        #             # structure
        #             mask_id             = idx,
        #             mask_mode           = "qkt",
        #             granularity         = "vector" if vector_flag else "matrix",
        #         )

        #         qkt_mm = qkt_mm.view(bsz, NUM_HEADS, tgt_len, src_len) + causal_attention_mask
        #         qkt_mm = torch.max(qkt_mm, torch.tensor(torch.finfo(qkt_mm.dtype).min))
        #         qkt_mm = qkt_mm.view(bsz * NUM_HEADS, tgt_len, src_len)

        #         # 这里注意，CPU算不能用float16，所以都统一成FP32。而GPU运行版本实际上都是float16，这里可能会有差距
        #         qkt_softmax = nn.functional.softmax(qkt_mm, dim=-1, dtype=torch.float32)  # .to(torch.float16)
        #         self.generator.softmax_layer(
        #             # name
        #             layer_name          = name_prefix + "softmax" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "qkt_mm" + name_suffix,
        #             input_shape         = qkt_mm.shape,
        #             # output
        #             output_name         = name_prefix + "qkt_softmax" + name_suffix,
        #             output_shape        = qkt_softmax.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "softmax_" + str(idx),
        #             # structure
        #             dim                 = 2,
        #             granularity         = "vector",
        #         )

        #         bsz, tgt_len, _ = hidden_states.size()
        #         qktv_mm = torch.bmm(qkt_softmax, value)

        #         self.generator.attention_layer(
        #             # name
        #             layer_name          = name_prefix + "attention_qktv" + name_suffix,
        #             # input
        #             input_A_name        = name_prefix + "qkt_softmax" + name_suffix,
        #             input_A_shape       = qkt_softmax.shape,
        #             input_B_name        = next_round_name_prefix + "past_values_t" + name_suffix,
        #             input_B_tag         = "v_cache_t" + name_suffix,
        #             input_B_shape       = value.transpose(1, 2).shape,  # 硬件的输入形状和torch MM输入形状不一致
        #             # output
        #             output_name         = name_prefix + "qktv_mm" + name_suffix,
        #             output_shape        = qktv_mm.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "attention_qktv_" + str(idx),
        #             # structure
        #             mask_id             = idx,
        #             mask_mode           = "qktv",
        #             granularity         = "vector" if vector_flag else "matrix",
        #         )

        #         qktv_mm = qktv_mm.view(bsz, NUM_HEADS, tgt_len, HEAD_DIM)
        #         qktv_t = qktv_mm.transpose(1, 2)

        #         self.generator.transpose_layer(
        #             # name
        #             layer_name          = name_prefix + "qktv_transpose" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "qktv_mm" + name_suffix,
        #             input_tag           = None,
        #             input_shape         = qktv_mm.shape,
        #             input_dtype         = "float16",
        #             input_data_type     = "sequence",
        #             # output
        #             output_name         = name_prefix + "qktv_t" + name_suffix,
        #             output_shape        = qktv_t.shape,
        #             output_data_type    = "sequence",
        #             # structure
        #             dim                 = [1, 2],
        #             dim_trans           = "up",
        #             granularity         = None,
        #         )
        #         qktv_t = qktv_t.reshape(bsz, tgt_len, EMBED_DIM)
        #         attn_output = self.o_proj_layer_list[idx](qktv_t)
        #         self.generator.linear_layer(
        #             # name
        #             layer_name          = name_prefix + "out_proj_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "qktv_t" + name_suffix,
        #             input_shape         = qktv_t.shape,
        #             # output
        #             output_name         = name_prefix + "attn_output" + name_suffix,
        #             output_shape        = attn_output.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "out_proj_layer_" + str(idx),
        #             weight              = self.o_proj_layer_list[idx].weight,
        #             bias                = self.o_proj_layer_list[idx].bias,
        #             # structure
        #             relu_flag           = False,
        #             granularity         = "vector" if vector_flag else "matrix",
        #         )

        #         hidden_sum = hidden_states + attn_output
        #         self.generator.eltwise_layer(
        #             # name
        #             layer_name          = name_prefix + "hidden_sum_layer" + name_suffix,
        #             # input
        #             input_A_name        = name_prefix + "hidden_states" + name_suffix,
        #             input_A_shape       = hidden_states.shape,
        #             input_B_name        = name_prefix + "attn_output" + name_suffix,
        #             input_B_shape       = attn_output.shape,
        #             # output
        #             output_name         = name_prefix + "hidden_sum" + name_suffix,
        #             output_shape        = hidden_sum.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "hidden_sum_layer_" + str(idx),
        #             # structure
        #             eltwise_type        = "eltwiseadd",
        #             granularity         = "element",
        #         )
        #         hidden_sum_shape = hidden_sum.shape
        #         hidden_sum = hidden_sum.reshape(-1, hidden_sum.size(-1))

        #         hidden_layernorm = self.atten_norm_layer_list[idx](hidden_sum)
        #         self.generator.layernorm_layer(
        #             # name
        #             layer_name          = name_prefix + "atten_layer_norm_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "hidden_sum" + name_suffix,
        #             input_shape         = hidden_sum.shape,
        #             # output
        #             output_name         = name_prefix + "hidden_layernorm" + name_suffix,
        #             output_shape        = hidden_layernorm.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "atten_layer_norm_layer_" + str(idx),
        #             weight              = self.atten_norm_layer_list[idx].weight,
        #             bias                = self.atten_norm_layer_list[idx].bias,
        #             # structure
        #             rms_flag            = False,
        #             granularity         = "vector",
        #         )

        #         """
        #             ### end BERT-Large forward ###
        #         """

        #         """
        #             ### Start BERT-Large FFN forward ###
        #         """

        #         hidden_intermediate = self.intermediate_layer_list[idx](hidden_layernorm)
        #         self.generator.linear_layer(
        #             # name
        #             layer_name          = name_prefix + "intermediate_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "hidden_layernorm" + name_suffix,
        #             input_shape         = hidden_layernorm.shape,
        #             # output
        #             output_name         = name_prefix + "hidden_intermediate" + name_suffix,
        #             output_shape        = hidden_intermediate.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "intermediate_layer_" + str(idx),
        #             weight              = self.intermediate_layer_list[idx].weight,
        #             bias                = self.intermediate_layer_list[idx].bias,
        #             # structure
        #             relu_flag           = False,
        #             granularity         = "vector" if vector_flag else "matrix",
        #         )

        #         hidden_gelu = self.activation_fn_layer_list[idx](hidden_intermediate)
        #         self.generator.gelu_layer(
        #             # name
        #             layer_name          = name_prefix + "activation_fn_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "hidden_intermediate" + name_suffix,
        #             input_shape         = hidden_intermediate.shape,
        #             # output
        #             output_name         = name_prefix + "hidden_gelu" + name_suffix,
        #             output_shape        = hidden_gelu.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "gelu_layer_" + str(idx),
        #             # structure
        #             granularity         = "element",
        #         )

        #         hidden_output = self.output_layer_list[idx](hidden_gelu)
        #         self.generator.linear_layer(
        #             # name
        #             layer_name          = name_prefix + "output_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "hidden_gelu" + name_suffix,
        #             input_shape         = hidden_gelu.shape,
        #             # output
        #             output_name         = name_prefix + "hidden_output" + name_suffix,
        #             output_shape        = hidden_output.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "output_layer_" + str(idx),
        #             weight              = self.output_layer_list[idx].weight,
        #             bias                = self.output_layer_list[idx].bias,
        #             # structure
        #             relu_flag           = False,
        #             granularity         = "vector" if vector_flag else "matrix",
        #         )

        #         hidden_add = hidden_sum + hidden_output
        #         self.generator.eltwise_layer(
        #             # name
        #             layer_name          = name_prefix + "hidden_add_layer" + name_suffix,
        #             # input
        #             input_A_name        = name_prefix + "hidden_sum" + name_suffix,
        #             input_A_shape       = hidden_sum.shape,
        #             input_B_name        = name_prefix + "hidden_output" + name_suffix,
        #             input_B_shape       = hidden_output.shape,
        #             # output
        #             output_name         = name_prefix + "hidden_add" + name_suffix,
        #             output_shape        = hidden_add.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "hidden_add_layer_" + str(idx),
        #             # structure
        #             eltwise_type        = "eltwiseadd",
        #             granularity         = "element",
        #         )

        #         final_hidden_states = self.final_layer_norm_layer_list[idx](hidden_add)
        #         self.generator.layernorm_layer(
        #             # name
        #             layer_name          = name_prefix + "final_layer_norm_layer" + name_suffix,
        #             # input
        #             input_name          = name_prefix + "hidden_add" + name_suffix,
        #             input_shape         = hidden_add.shape,
        #             # output
        #             output_name         = name_prefix + "final_hidden_states" + name_suffix,
        #             output_shape        = final_hidden_states.shape,
        #             output_dtype        = "float16",
        #             # param
        #             param_name_prefix   = "final_layer_norm_layer",
        #             weight              = self.final_layer_norm_layer_list[idx].weight,
        #             # 如果是最后一个，则下一个操作是pooler，否则下一个操作是生成QKV
        #             bias                = self.final_layer_norm_layer_list[idx].bias,
        #             # structure
        #             rms_flag            = False,
        #             granularity         = "vector",
        #         )

        #         hidden_reshape_2 = final_hidden_states.view(hidden_sum_shape)

        #         hidden_states = hidden_reshape_2  # next hidden states
        #         next_decoder_cache += (past_key_value,)

        # """
        #     ### end BERT-Large FFN forward ###
        # """
        # if self.pool:
        #     final_hidden_states = self.pool_layer(hidden_states)

        #     self.seq_relationship_new = nn.Linear(final_hidden_states.shape[-1], 32 * 3)
        #     new_seq_relationship_weight = self.seq_relationship.weight.repeat(32 * 3 // 2, 1)
        #     new_seq_relationship_bias = self.seq_relationship.bias.repeat(32 * 3 // 2)
        #     self.seq_relationship_new.weight = nn.Parameter(new_seq_relationship_weight)
        #     self.seq_relationship_new.bias = nn.Parameter(new_seq_relationship_bias)

        #     result = self.seq_relationship_new(final_hidden_states)

        # else:
        #     raise NotImplementedError
        # """
        #     ### end BERT-Large Encode ###
        # """
        # assert len(next_decoder_cache) == NUM_HIDDEN_LAYERS

        # self.generator.output_layer(
        #     # name
        #     layer_name=name_prefix + "output_layer",
        #     # output act
        #     output_act_name=name_prefix + "final_hidden_states" + self.get_name_suffix(NUM_HIDDEN_LAYERS - 1),  # name_prefix + "result",
        #     output_act_shape=result.shape,
        #     output_act_dtype="float16",
        #     # k cache list
        #     k_cache_name_list=list(),
        #     k_cache_tag_list=list(),
        #     k_cache_shape_list=list(),
        #     k_cache_dtype_list=list(),
        #     k_cache_data_type="k_cache",
        #     # v cache list
        #     v_cache_name_list=list(),
        #     v_cache_tag_list=list(),
        #     v_cache_shape_list=list(),
        #     v_cache_dtype_list=list(),
        #     v_cache_data_type="v_cache_t",
        # )
        # self.call_forward_count += 1

        # return CausalLMOutputWithPast(
        #     logits=final_hidden_states,
        #     past_key_values=next_decoder_cache,
        # )

        """
            ### end BERT-Large forward ###
        """
