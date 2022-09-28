import torch
state_dict_original = torch.load("checkpoints/sparse_inst_r50_base_ff9809.pth")
state_dict_mmdet = torch.load("work_dirs/sparse-inst_r50_x1_coco/epoch_1.pth")
from files import *
covert_file = openjson('sparse_inst_r50_base_covert.json')
for module_name in covert_file:
    state_dict_mmdet['state_dict'][module_name] = state_dict_original[covert_file[module_name]]
torch.save(state_dict_mmdet, 'sparse_inst_r50_base_mmdet.pth')