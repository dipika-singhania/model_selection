import json
import pandas as pd
import csv
import numpy as np

class BuildDatasetModelCSV():
    def __init__(self, json_file, dataset_features_file, dataset_model_train_file):
        self.json_file = json_file
        self.dataset_features_file = dataset_features_file
        self.dataset_model_train_file = dataset_model_train_file

    def calculate_train_frames_per_sec(self, json_obj):
        # train_data_szie * num_of_epochs / total_train_time
        return json_obj['train_data_size'] * json_obj['total_epoch_run'] / json_obj['total_train_time']

    def calculate_eval_frames_per_sec(self, json_obj):
        # eval_data_size / total_eval_time
        return json_obj['eval_data_size'] / json_obj['eval_time']

    def calculate_score(self, json_obj):
        # return (json_obj['top_5_acc'] + json_obj['top_5_recall'] + json_obj['top_1_recall'] + json_obj['top_1_acc']) / 4
        return json_obj['top_1_recall']

    def load_and_populate_json_dump(self):
        self.datasets_name = set()
        self.models_name = set()
        self.table = {}
        with open(self.json_file, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                json_obj = json.loads(line.strip())
                d_name = json_obj['dataset']
                m_name = json_obj['model_name']
                if d_name not in self.datasets_name:
                    self.datasets_name.add(d_name)
                    self.table[d_name] = {}
                if m_name not in self.models_name:
                    self.models_name.add(m_name)
                    self.table[d_name][m_name] = []
                self.table[d_name][m_name] = [self.calculate_score(json_obj), 
                                         self.calculate_train_frames_per_sec(json_obj),
                                         self.calculate_eval_frames_per_sec(json_obj),
                                         json_obj['model_params']]


    def normalize_table(self):
        return       

                
    def get_datasets_names(self):
        print("Dataset name :")
        print(";".join(self.datasets_name))
        return self.datasets_name


    def get_models_names(self):
        print("Models name :")
        print(";".join(self.models_name))
        return self.models_name


    def get_target_table(self):
        target_table = []
        len_models = len(self.models_name)
        models_name = list(self.models_name)
        for d_name in self.datasets_name:
            for i in range(len_models):
                for j in range(i + 1, len_models):
                    m1 = models_name[i]
                    m2 = models_name[j]
                    if m1 == m2:
                        continue
                    row_ = [d_name, m1, m2, 1]
                    if self.table[d_name][m1][0] < self.table[d_name][m2][0]:
                        row_ = [d_name, m1, m2, 0]
                    target_table.append(row_)
        print("Total size of table =", len(target_table))
        return target_table
        

    def print_table(self, file_name, target_table):
        with open(file_name, 'w') as fp:
            for line in target_table:
                fp.write(",".join(map(str, line)) + "\n")

    def get_feature_target_table(self, target_table):
        target_feature_table = []
        self.get_dataset_feature()


    def get_model_features(self):
        return

    def get_dataset_feature(self, dataset_name):
        self.dataset_feature = {}
        with open(self.dataset_features_file, "r") as fp:
            lines = fp.read_lines()
            for line in lines[1:]: # First line contains dataset features
                line_split = line.strip().split(",")
                self.dataset_name[line_split[0]] = line_split[1:]
        return

    def get_models_features(self, models_name):
        return

    def get_ranked_list_of_model_params(self):
        return

    def get_ranked_list_of_model_accuracy(self, dataset_name):
        return

    def get_ranked_list_of_model_runtime(self, dataset_name):
        return

if __name__ == "__main__":
    json_dump_file = "logs/model_datasets_meta.json"
    dataset_features_file = "dataset_model_features/dataset.csv"
    dataset_model_train_file = "dataset_model_features/meta_model_learning.csv"
    bd = BuildDatasetModelCSV(json_dump_file, dataset_features_file, dataset_model_train_file)
    bd.load_and_populate_json_dump()
    target_table = bd.get_target_table()
    bd.print_table("dataset_model_features/dataset_model_name.csv", target_table)
    final_target_file = bd.get_feature_target_table()
