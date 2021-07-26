# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
from torch import nn

# self-defined file
from load_model import _load
from load_image import *
from config import ActivityConfig as cfg
from data import train_loader_local, val_loader_local
from utils import IO, accuracy, AverageMeter, synchronize
from summary import Summary

# input image
# LABELS_file = 'imagenet-simple-labels.json'

# load image which will be processed
# image_file = 'test.jpg'
class_name = 'walk'
root_path = "/data/guojie/HMDB51/VideoSeq"
image_file = load_image_from_HMDB_video(root_path, class_name, None)


# pre-define variable
pth_path = "/data/guojie/ar_output/T_tsn_ssl_tsn/checkpoint/2021_07_25_20:10:07_best_NONE_NONE.pth.tar"
net = _load(pth_path)
net = nn.Sequential(*list(net.children()))[:-1]
# feature layer
finalconv_name = 'layer4'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# for name, module in net.named_modules():
    # print('modules: ', name)
net._modules["0"]._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)



preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])



# load test image
img_pil = Image.open(image_file)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# load the imagenet category list
# with open(LABELS_file) as f:
    # classes = json.load(f)


h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
# for i in range(0, 5):
    # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
# print(len(idx))
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
CAMs = returnCAM(features_blobs[0], weight_softmax, [0])

# render the CAM and output
# print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cam_file_name = class_name + "_CAM.jpg"
cv2.imwrite(cam_file_name, result)