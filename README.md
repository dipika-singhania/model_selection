# model_selection

Automatic Model Selection



import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
shufflenet = models.shufflenet_v2_x1_0()
mobilenet = models.mobilenet_v2()
resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
mnasnet = models.mnasnet1_0()


export PYTHONPATH=$PWD
 
python models/image_classification/pytorch_models_run.py --final_log logs/params_resnet18_db.json --model_name googlenet |& tee googlenet
python models/image_classification/pytorch_models_run.py --final_log logs/params_resnet18_db.json --model_name mobilenet_v2 |& tee mobilenet_v2_log



