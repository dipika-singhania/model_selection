from torchvision import models
import torch
import torch.nn as nn

class ModelsZoo():
    def __init__(self):
        self.models_dict = {
                'resnet': models.resnet18(pretrained=True),
                'alexnet': models.alexnet(pretrained=True),
                'vgg16': models.vgg16(pretrained=True),
                'squeezenet': models.squeezenet1_0(pretrained=True),
                'densenet161': models.densenet161(pretrained=True),
                'shufflenet': models.shufflenet_v2_x0_5(pretrained=True),
                'mobilenet': models.mobilenet_v2(pretrained=True),
                'resnext50': models.resnext50_32x4d(pretrained=True),
                'mnasnet': models.mnasnet0_5(pretrained=True)
                }

    def get_models_names_list(self):
        return self.models_dict.keys()

    def get_params(self, model):
       return sum(p.numel() for p in model.parameters())
    
    def get_modified_model(self, model_name, class_size):
        model = self.models_dict[model_name]
        print("Loading model with params = ", self.get_params(model))
        if model_name in ['mnasnet', 'mobilenet', 'alexnet', 'vgg16']:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, class_size)
        elif model_name in ['densenet161']:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, class_size)

        elif model_name in ['resnet', 'resnext50', 'shufflenet']:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, class_size)

        elif model_name in ['squeeze_net']:
            # Need to change conv and other after layers
            conv2d_l = model.classifier[1]
            model.classifier[1] = torch.nn.Conv2d(in_channels=conv2d_l.in_Channesls, out_channels=conv2d_l.out_channels, kernel_size=conv2d_l.kernel_size, stride=conv2d_l.stride)

        return model

