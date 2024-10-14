from my_var import MyVAR
import torch
from ir_generator import IRGenerator
model_ir_generator = IRGenerator(stage='decode')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_ids = torch.zeros((16,1,1024)).to(device)
cond_BD = torch.zeros((16,1024)).to(device)
f_hat = torch.zeros((8,32,16,16)).to(device)
my_model = MyVAR(model_ir_generator)
decode_flag = True
my_model.forward(input_ids,cond_BD,f_hat,decode_flag)
#model_ir_generator.delete_head_tail_layers()
model_ir_generator.dump_json()
#model_ir_generator.ir