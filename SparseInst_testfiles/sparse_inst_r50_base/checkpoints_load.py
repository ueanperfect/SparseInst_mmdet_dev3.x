import torch
state_dict_original = torch.load("checkpoints/sparse_inst_r50_base_ff9809.pth")
state_dict_mmdet = torch.load("work_dirs/sparse-inst_r50_x1_coco/epoch_1.pth")
state_dict_mmdet = state_dict_mmdet['state_dict']
ori_backbone_dic={}
ori_encoder_dic = {}
ori_decoder_dic = {}
for module_name in state_dict_original:
    if module_name[0] == 'b':
        ori_backbone_dic[module_name] = state_dict_original[module_name]
    elif module_name[0] == 'e':
        ori_encoder_dic[module_name] = state_dict_original[module_name]
    else:
        ori_decoder_dic[module_name] = state_dict_original[module_name]
mmdet_backbone_dic={}
mmdet_encoder_dic = {}
mmdet_decoder_dic = {}
for module_name in state_dict_mmdet:
    if module_name[0] == 'b':
        mmdet_backbone_dic[module_name] = state_dict_mmdet[module_name]
    elif module_name[:11] == 'mask_head.e':
        mmdet_encoder_dic[module_name] = state_dict_mmdet[module_name]
    else:
        mmdet_decoder_dic[module_name] = state_dict_mmdet[module_name]

mmdet_backbone_layer = [{},{},{},{}]
for module_name in mmdet_backbone_dic:
    if module_name[:15]=='backbone.layer1':
        mmdet_backbone_layer[0][module_name] = mmdet_backbone_dic[module_name]
    elif module_name[:15]=='backbone.layer2':
        mmdet_backbone_layer[1][module_name] = mmdet_backbone_dic[module_name]
    elif module_name[:15]=='backbone.layer3':
        mmdet_backbone_layer[2][module_name] = mmdet_backbone_dic[module_name]
    elif module_name[:15] == 'backbone.layer4':
        mmdet_backbone_layer[3][module_name] = mmdet_backbone_dic[module_name]

ori_backbone_layer = [{},{},{},{}]
for module_name in ori_backbone_dic:
    if module_name[:13]=='backbone.res2':
        ori_backbone_layer[0][module_name] = ori_backbone_dic[module_name]
    elif module_name[:13]=='backbone.res3':
        ori_backbone_layer[1][module_name] = ori_backbone_dic[module_name]
    elif module_name[:13]=='backbone.res4':
        ori_backbone_layer[2][module_name] = ori_backbone_dic[module_name]
    elif module_name[:13] == 'backbone.res5':
        ori_backbone_layer[3][module_name] = ori_backbone_dic[module_name]

layer_list = [[],[],[],[]]
for index,layer in enumerate(mmdet_backbone_layer):
    for module_name in layer:
        if module_name[-1]=='d':
            layer_list[index].append(module_name)

for index,layer in enumerate(layer_list):
    for module_name in layer:
        mmdet_backbone_layer[index].pop(module_name)

from files import *
for index,layer in enumerate(mmdet_backbone_layer):
    dic = {}
    for module_name in layer:
        dic[module_name] = 0
    write_json(str(index)+'mmdetbackbone.json',dic)

for index,layer in enumerate(ori_backbone_layer):
    dic = {}
    for module_name in layer:
        dic[module_name] = 0
    write_json(str(index)+'deteronbackbone.json',dic)

