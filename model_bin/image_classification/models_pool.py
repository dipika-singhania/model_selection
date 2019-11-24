from torchvision import models
import torch
import torch.nn as nn

MODEL_DICTIONARY = {'resnet18': {'pretrained': False},
                    'alexnet': {'pretrained': False},
                    'vgg16': {'pretrained': True},
                    'shufflenet_v2_x0_5': {'pretrained': False},
                    'mobilenet_v2': {'pretrained': True},
                    'mnasnet0_5': {'pretrained': False},
                    'mnasnet1_0': {'pretrained': False},
                    'resnet34': {'pretrained': False}}


class ModelsZoo():
    def __init__(self):
        """
        self.models_dict = {
                'resnet': models.resnet18(pretrained=True),
                'alexnet': models.alexnet(pretrained=True),
                'vgg16': models.vgg16(pretrained=True),
                'squeezenet': models.squeezenet1_0(pretrained=True),
                'densenet161': models.densenet161(pretrained=True),
                'shufflenet': models.shufflenet_v2_x0_5(pretrained=True),
                'mobilenet': models.mobilenet_v2(pretrained=True),
                'resnext50': models.resnext50_32x4d(pretrained=True),
                'mnasnet': models.mnasnet0_5(pretrained=True),
                'mnasnet0_75':models.mnasnet0_75(pretrained=True),
                'mnasnet1_0':models.mnasnet1_0(pretrained=True),
                'mnasnet1_3':models.mnasnet1_3(pretrained=True),
                'resnet34':models.resnet34(pretrained=True),
                'resnet50':models.resnet50(pretrained=True),
                'inception_v3':models.inception_v3(pretrained=True),
                'resnet101':models.resnet101(pretrained=True),
                'resnet152':models.resnet152(pretrained=True)
                }
        """

    def get_models_names_list(self):
        return self.models_dict.keys()

    def get_params(self, model):
       return sum(p.numel() for p in model.parameters())

    def extract_resnet50_features(self):
        res50_model = models.resnet50(pretrained=True)
        res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
        import pdb
        pdb.set_trace()
        for param in res50_conv.parameters():
            param.requires_grad = False
        return res50_conv 
    
    def get_modified_model(self, model_name, class_size, train_full=True):
        method_to_call = getattr(models, model_name)
        model = method_to_call(pretrained=True)

        if train_full is False:
            for param in model.parameters():
                param.requires_grad = False

        print("Loading model with params = ", self.get_params(model))
        if model_name in ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'alexnet', 'vgg16']:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, class_size)
        elif model_name in ['densenet161']:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, class_size)

        elif model_name in ['resnet18', 'resnet50', 'resnet101', 'resnet152','resnet34', 'resnext50', 'shufflenet', 'inception_v3']:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, class_size)

        elif model_name in ['squeeze_net']:
            # Need to change conv and other after layers
            conv2d_l = model.classifier[1]
            model.classifier[1] = torch.nn.Conv2d(in_channels=conv2d_l.in_Channesls, out_channels=conv2d_l.out_channels, kernel_size=conv2d_l.kernel_size, stride=conv2d_l.stride)

        return model

