from my_var import MyBertLargeModel
import torch
from ir_generator import IRGenerator
model_ir_generator = IRGenerator()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_ids = torch.zeros((16,1,1024)).to(device)
my_model = MyBertLargeModel(model_ir_generator)

my_model.forward(input_ids)
model_ir_generator.delete_head_tail_layers()
model_ir_generator.dump_json()
#model_ir_generator.ir