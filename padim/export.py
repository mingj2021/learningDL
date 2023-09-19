import torch
import torchvision
from torch import Tensor, nn
from torchvision.models import resnet18, ResNet18_Weights, \
resnet101,ResNet101_Weights,  \
resnet50,ResNet50_Weights,   \
mobilenet_v3_large,MobileNet_V3_Large_Weights,   \
mobilenet_v2, MobileNet_V2_Weights,swin_v2_t,Swin_V2_T_Weights,swin_v2_s,Swin_V2_S_Weights,   \
vgg16, VGG16_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights

from torchsummary import summary

class FeatureExtractorResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.block0 = model.layer1
        self.block1 = model.layer2
        self.block2 = model.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features_0 = self.block0(x)
        features_1 = self.block1(features_0)
        features_2 = self.block2(features_1)
        return (features_0, features_1, features_2)

class FeatureExtractorEfficientnet_v2_s(nn.Module):
    def __init__(self):
        super().__init__()
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.block0 = model.features[0]
        self.block1 = model.features[1]
        self.block2 = model.features[2]
        self.block3 = model.features[3]
        self.block4 = model.features[4]
        self.block5 = model.features[5]
        
    def forward(self, x):
        features_0 = self.block0(x)
        features_1 = self.block1(features_0)
        features_2 = self.block2(features_1)
        features_3 = self.block3(features_2)
        features_4 = self.block4(features_3)
        features_5 = self.block5(features_4)
        return (features_1,features_2,features_3,features_5)

class FeatureExtractorVgg16(nn.Module):
    def __init__(self):
        super().__init__()
        model = vgg16(weights=VGG16_Weights)
        self.block0 = model.features[0:4]
        self.block1 = model.features[4:9]
        self.block2 = model.features[9:16]
        self.block3 = model.features[16:23]

    def forward(self, x):
        features_0 = self.block0(x)
        features_1 = self.block1(features_0)
        features_2 = self.block2(features_1)
        features_3 = self.block3(features_2)
        return (features_0, features_1, features_2, features_3)

class FeatureExtractorMobilenet_v2(nn.Module):
    def __init__(self):
        super().__init__()
        model = mobilenet_v2(weights=MobileNet_V2_Weights)
        self.block0 = model.features[0:2]
        self.block1 = model.features[2:4]
        self.block2 = model.features[4:7]
        self.block3 = model.features[7:14]

    def forward(self, x):
        features_0 = self.block0(x)
        features_1 = self.block1(features_0)
        features_2 = self.block2(features_1)
        features_3 = self.block3(features_2)
        return (features_0, features_1, features_2, features_3)


if __name__ == '__main__':
    _all = ['resnet18', 'efficientnet_v2_s', 'vgg16', 'mobilenet_v2']

    for backbone in _all:
        if backbone == 'resnet18':
            model = FeatureExtractorResNet18()
        elif backbone == 'efficientnet_v2_s':
            model = FeatureExtractorEfficientnet_v2_s()
        elif backbone == 'vgg16':
            model = FeatureExtractorVgg16()
        elif backbone == 'mobilenet_v2':
            model = FeatureExtractorMobilenet_v2()
        
        dummy_input = torch.randn(1, 3, 512, 512)
        input_names = [ "images" ]
        output_names = []
        x = model(dummy_input)
        for i in range(len(x)):
            output_names += ['output{}'.format(i)]

        torch.onnx.export(model, dummy_input, "{}.onnx".format(backbone), verbose=True, input_names=input_names, output_names=output_names)
