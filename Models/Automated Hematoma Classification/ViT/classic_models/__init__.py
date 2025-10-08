from .resnet import  resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from .densenet import densenet121, densenet161, densenet169, densenet201
from .shufflenet_v2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from .vision_kansformer import kit_base_patch16_224,kit_base_patch16_224_in21k
from .vision_transformer import vit_base_patch16_224, vit_base_patch32_224, vit_large_patch16_224, vit_base_patch16_224_in21k
from .swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224
cfgs = {   
    'resnet_small': resnet34,
    'resnet': resnet50,
    'resnet_big': resnet101,
    'resnext': resnext50_32x4d,
    'resnext_big': resnext101_32x8d,
    'densenet_tiny': densenet121,
    'densenet_small': densenet161,
    'densenet': densenet169,
    'densenet_big': densenet121,
    'shufflenet_small':shufflenet_v2_x0_5,
    'shufflenet': shufflenet_v2_x1_0,
    'kansformer1': kit_base_patch16_224,
    'kansformer2': kit_base_patch16_224_in21k,
    'vision_transformer_small': vit_base_patch32_224,  
    'vision_transformer': vit_base_patch16_224,
    'vision_transformer2': vit_base_patch16_224_in21k,
    'vision_transformer_big': vit_large_patch16_224,
    'swin_transformer_tiny': swin_tiny_patch4_window7_224,
    'swin_transformer_small': swin_small_patch4_window7_224,
    'swin_transformer': swin_base_patch4_window7_224,
}

def find_model_using_name(model_name, num_classes):   
    return cfgs[model_name](num_classes)

 
