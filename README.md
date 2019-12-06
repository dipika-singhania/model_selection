# model_selection

Automatic Model Selection

For details of running codes, please refer to "CS6203.pdf"

1. Initial Environment creation, one time

cd model_selection

conda env create -f environment.yml

conda activate mdl

1. First Step before running any command

cd model_selection

export PYTHONPATH=$PWD

2. To get the list of suggested model for a dataset

python run_scripts/suggest_model.py --dataset=cifar10

3. To get inference score of a model on a particular dataset

CUDA_VISIBLE_DEVICES=0 python run_scripts/run_model_on_dataset.py --model_name resnet18 --dataset cifar10 --infer 

4. To train a model on a particular dataset

CUDA_VISIBLE_DEVICES=0 python run_scripts/run_model_on_dataset.py --model_name resnet18 --dataset cifar10 --train

