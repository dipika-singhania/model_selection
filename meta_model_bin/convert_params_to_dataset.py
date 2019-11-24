import json
import pandas as pd
import csv
import numpy as np


class BuildDatasetModelCSV():
    def __init__(self, json_file, dataset_features_file, models_features_file, dataset_model_train_file):
        self.json_file = json_file
        self.dataset_features_file = dataset_features_file
        self.models_features_file = models_features_file
        self.dataset_model_train_file = dataset_model_train_file

    def calculate_train_frames_per_sec(self, json_obj):
        # train_data_szie * num_of_epochs / total_train_time
        return json_obj['train_data_size'] * json_obj['total_epoch_run'] / json_obj['total_train_time']

    def calculate_eval_frames_per_sec(self, json_obj):
        # eval_data_size / total_eval_time
        return json_obj['eval_data_size'] / json_obj['eval_time']

    def calculate_score(self, json_obj):
        # return (json_obj['top_5_acc'] + json_obj['top_5_recall'] + json_obj['top_1_recall'] + json_obj['top_1_acc']) / 4
        return json_obj['top_1_acc']
    
    def dump_model_features_files(self):
        with open(self.models_features_file, "w") as fp:
            fp.write("model_name,model_params\n")
            for ele in self.model_features.keys():
                if ele == "header":
                    continue
                fp.write(ele + "," + ",".join(map(str, self.model_features[ele])) + "\n")

    def load_and_populate_json_dump(self):
        self.datasets_name = set()
        self.models_name = set()
        self.table = {}
        self.model_features = {"header":["model_name", "model_params"]}
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
                    self.model_features[m_name] = [json_obj['model_params']]
                self.table[d_name][m_name] = [self.calculate_score(json_obj), 
                                              self.calculate_train_frames_per_sec(json_obj),
                                              self.calculate_eval_frames_per_sec(json_obj),
                                              json_obj['model_params']]
        self.dump_model_features_files()

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

    def get_scores_target_table(self):
        header = ["dataset_name", "model_name_1", "model_name_2", "target"]
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
        return target_table, header

    def get_scores_target_table_permute(self):
        header = ["dataset_name", "model_name_1", "model_name_2", "target"]
        target_table = []
        len_models = len(self.models_name)
        models_name = list(self.models_name)
        for d_name in self.datasets_name:
            for i in range(len_models):
                for j in range(len_models):
                    if (i==j):
                        continue
                    m1 = models_name[i]
                    m2 = models_name[j]
                    if m1 == m2:
                        continue
                    row_ = [d_name, m1, m2, 1]
                    if self.table[d_name][m1][0] < self.table[d_name][m2][0]:
                        row_ = [d_name, m1, m2, 0]
                    target_table.append(row_)
        print("Total size of table =", len(target_table))
        return target_table, header

    def get_time_target_table(self):
        target_table = []
        header = ["dataset_name", "model_name_1", "model_name_2", "time_target"]
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
                    if self.table[d_name][m1][1] < self.table[d_name][m2][1]:
                        row_ = [d_name, m1, m2, 0]
                    target_table.append(row_)
        print("Total size of table =", len(target_table))
        return target_table, header

    def print_table(self, file_name, target_table, header):
        with open(file_name, 'w') as fp:
            fp.write(",".join(header) + "\n")
            for line in target_table:
                fp.write(",".join(map(str, line)) + "\n")
        print("Save table to ", file_name)

    def get_feature_target_table(self, target_table):
        target_feature_table = []
        self.load_dataset_feature()
        self.load_model_features()
        header = ['name'] + self.dataset_feature["header"][1:]
        model_feature_1 = [m1 + "_1" for m1 in self.model_features["header"][1:]]
        model_feature_2 = [m1 + "_2" for m1 in self.model_features["header"][1:]]
        header.extend(model_feature_1)
        header.extend(model_feature_2)
        header.extend(["target"])
        for ele in target_table:
            dataset_feature = ["-".join([ele[0], ele[1], ele[2]])]
            dataset_feature.extend(self.dataset_feature[ele[0]])
            dataset_feature.extend(self.model_features[ele[1]])
            dataset_feature.extend(self.model_features[ele[2]])
            dataset_feature.extend([ele[3]])
            target_feature_table.append(dataset_feature)
        print("Created target feature table ")
        return target_feature_table, header

    def load_model_features(self):
        if self.model_features is not None:
            return
        else:
            self.model_features = {}
            with open(self.models_features_file, "r") as fp:
                self.model_features["header"] = line[0].strip().split(",")
                lines = fp.readlines()
                for line in lines[1:]: # First line contains dataset features
                    line_split = line.strip().split(",")
                    self.model_features[line_split[0]] = line_split[1:]
            return

    def load_dataset_feature(self):
        self.dataset_feature = {}
        with open(self.dataset_features_file, "r") as fp:
            lines = fp.readlines()
            self.dataset_feature["header"] = lines[0].strip().split(",")
            for line in lines[1:]: # First line contains dataset features
                line_split = line.strip().split(",")
                self.dataset_feature[line_split[0]] = line_split[1:]
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
    dataset_features_file = "meta_learning_features/dataset.csv"
    dataset_model_train_file = "meta_learning_features/meta_model_learning.csv"
    models_features_file = "meta_learning_features/models.csv"
    bd = BuildDatasetModelCSV(json_dump_file, dataset_features_file, models_features_file, dataset_model_train_file)
    bd.load_and_populate_json_dump()
#     target_table, header = bd.get_scores_target_table()
    target_table, header = bd.get_scores_target_table_permute()
    bd.print_table("meta_learning_features/dataset_scores_model_name.csv", target_table, header)
    time_target_table, header = bd.get_time_target_table()
    bd.print_table("meta_learning_features/dataset_time_model_name.csv", target_table, header)
    final_target_file, header = bd.get_feature_target_table(target_table)
    bd.print_table(dataset_model_train_file, final_target_file, header)
