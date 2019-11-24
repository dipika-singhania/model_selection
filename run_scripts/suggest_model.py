import json
import pandas as pd
import csv
import numpy as np
import os
import argparse
import pickle
import meta_model_bin.convert_params_to_dataset as gen_files
from dataset_loaders.image_classification import data_config as data_cnf
from meta_model_bin.get_labels_to_vec import generate_PCA_for_target_embedding
from meta_model_bin.convert_params_to_dataset import BuildDatasetModelCSV
from dataset_loaders.image_classification.data_config import DATA_DICTIONARY
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def get_scores(names, targets):
    names = names.values
    scores = {}
    for i in range(len(names)):
        dataset, model1, model2 = names[i].split('-')
        scores[model1] = scores.get(model1, 0) + (targets[i] >= 0.5)
        scores[model2] = scores.get(model2, 0) + (targets[i] <= 0.5)
    return scores


def create_meta_model_and_suggest(data_name, meta_features_dir='meta_learning_features/',
                                  saved_model_loc='meta_model_pool/', save_model=True):

    target_embedding_dir = meta_features_dir + 'target_embeddings/'
    data_dir = 'dataset_pool/'

    existing_targets = [file.strip().split('_')[0] for file in os.listdir(target_embedding_dir)]
    for dataset in data_cnf.DATA_DICTIONARY.keys():
        if dataset not in existing_targets:
            print("%s dataset target embeddings does not exists and will be created"%dataset)
            _ = data_cnf.DATA_DICTIONARY[dataset](data_dir=data_dir, add_features=True)

    generate_PCA_for_target_embedding()
    json_dump_file = "logs/model_datasets_meta.json"
    dataset_features_file = meta_features_dir + "dataset.csv"
    dataset_model_train_file = meta_features_dir + "meta_model_learning.csv"
    models_features_file = meta_features_dir + "models.csv"
    bd = BuildDatasetModelCSV(json_dump_file, dataset_features_file, models_features_file, dataset_model_train_file)
    bd.load_and_populate_json_dump()
    #     target_table, header = bd.get_scores_target_table()
    target_table, header = bd.get_scores_target_table_permute()
    bd.print_table(meta_features_dir + "dataset_scores_model_name.csv", target_table, header)
    time_target_table, header = bd.get_time_target_table()
    bd.print_table(meta_features_dir + "dataset_time_model_name.csv", target_table, header)
    final_target_file, header = bd.get_feature_target_table(target_table)
    bd.print_table(dataset_model_train_file, final_target_file, header)

    feature_target_file = meta_features_dir + "meta_model_learning.csv"
    data_frame = pd.read_csv(feature_target_file)
    X = data_frame[data_frame.columns[1:-1]]
    y = data_frame[data_frame.columns[-1]]
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(X)
    train_ind = [data_name not in name for name in data_frame['name']]
    test_ind = [data_name in name for name in data_frame['name']]

    X_train = X[train_ind]
    y_train = y[train_ind]
    X_test = X[test_ind]
    y_test = y[test_ind]

    df_train = data_frame[train_ind]
    df_test = data_frame[test_ind]

    random_state = 0
    dtree_model = DecisionTreeClassifier(min_samples_leaf=5, random_state=random_state)
    dtree_clf = dtree_model.fit(X_train, y_train)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    y_pred_tree = dtree_clf.predict(X_test)
    y_pred_neigh = neigh.predict(X_test)

    tree_score = get_scores(df_test['name'], y_pred_tree)
    KNN_socre = get_scores(df_test['name'], y_pred_neigh)

    final_score = {}
    for key in tree_score:
        final_score[key] = tree_score[key] + KNN_socre[key]

    ranked_models = sorted(final_score, key=final_score.__getitem__, reverse=True)
    print("\033[92m")
    print("Suggested model is:", ranked_models[0])
    print("\033[00m")
    print('All the models and their ranking are:')
    for count, model in enumerate(ranked_models, 1):
        print('%d. %s'%(count, model))
    if save_model:
        save_dict = {'test': data_name,
                     'dtree': dtree_clf,
                     'KNN': neigh}
        with open(saved_model_loc + 'meta_model.pkl', 'wb') as openf:
            pickle.dump(save_dict, openf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATA_DICTIONARY.keys()), help="Name of the dataset")
    parser.add_argument('--meta_data_dir', type=str, default='meta_learning_features/', help="Meta data location")
    parser.add_argument('--save_meta_model', action='store_true', default=False,
                        help="Whether to save the trained meta model")
    parser.add_argument('--meta_model_save_loc', default='meta_model_pool/', type=str,
                        help="Where to save the trained meta model")

    (args, _) = parser.parse_known_args()

    create_meta_model_and_suggest(data_name=args.dataset, meta_features_dir=args.meta_data_dir,
                                  saved_model_loc=args.meta_model_save_loc, save_model=args.save_meta_model)