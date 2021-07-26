import torch
import torchvision.models as models

from model import Model
from config import ActivityConfig as cfg
from ModelZoo import *

def load_model_weights(pth_path):
    adjusted_weights = {}
    model_weights = torch.load(pth_path, map_location='cpu')

    for name, params in model_weights['model_dict'].items():
        if "module" in name:
            name = name[name.find('.') + 1:]
            # print(name)
        adjusted_weights[name] = params
    
    return adjusted_weights

def _load(pth_path):

    model = Model()
    model = model.select_model(cfg.MODEL_NAME)
    
    if model.load_state_dict(load_model_weights(pth_path)):
        print('Load state dict success!')
    else:
        print('Could not load state dict!')

    return model


if __name__ == '__main__':
    pth_path = "/data/guojie/ar_output/T_tsn_ssl_tsn/checkpoint/2021_07_25_20:10:07_best_NONE_NONE.pth.tar"
    model = _load(pth_path)

    print(model)