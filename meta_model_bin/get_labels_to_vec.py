import numpy as np
import csv
import os
import pickle
from sklearn.decomposition import PCA


def load_glove_embed():
    embeddings_dict = {}
    
    with open("/home/dipika16/CS6203/glove/glove.6B.300d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict
    

def get_word_list_embeddings(dataset_name, words_list, update_pca=False):
    embeddings_dict = load_glove_embed()
    vector_list = []
    for word in words_list:
        word = str(word)
        word = word.lower()
        sub_words = word.split()
        for sub_word in sub_words:
            if sub_word in embeddings_dict:
                vector_word = embeddings_dict[sub_word]
                vector_list.append(vector_word)

    target_embedding_file = '/home/dipika16/CS6203/meta_learning_features/target_embeddings/'
    target_embedding_file += dataset_name + '_target_embedding.pkl'
    with open(target_embedding_file, 'wb') as embedding_file:
        pickle.dump(vector_list, embedding_file)
        
    if update_pca or (not os.path.exists('/home/dipika16/CS6203/meta_learning_features/pca_embedding.pkl')):
        print('PCA generation will be performed')
        generate_PCA_for_target_embedding()
        
    with open('/home/dipika16/CS6203/meta_learning_features/pca_embedding.pkl', 'rb') as pcafile:
        pca_model = pickle.load(pcafile)
        
    pca_embedding = pca_model.transform(vector_list)
    max_feature = np.max(pca_embedding, axis=0).tolist()
    average_feature = np.mean(pca_embedding, axis=0).tolist()
    min_feature = np.min(pca_embedding, axis=0).tolist()
    final_feature = np.array(max_feature + average_feature + min_feature)
    print("Shape of vector =", final_feature.shape)

    return final_feature


def add_vector_to_file(name, img_height, img_width, total_train_size, num_of_classes, labels_features):
    
    final_list = [name,img_height, img_width, total_train_size,num_of_classes]
    final_list.extend(labels_features.tolist())
    csv_file_name = '/home/dipika16/CS6203/meta_learning_features/dataset.csv'
    with open(csv_file_name, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(final_list)
        
    
def generate_PCA_for_target_embedding():
    directory = '/home/dipika16/CS6203/meta_learning_features/target_embeddings/'
    files = [file for file in os.listdir(directory) if file.endswith('target_embedding.pkl')]
    embedding_list = []
    for file in files:
        with open(directory + file, 'rb') as openfile:
            embedding = pickle.load(openfile)
        embedding_list.extend(embedding)
    
    print('Total embeddings', len(embedding_list))
    pca_embedding = PCA(n_components=5)
    pca_embedding.fit(embedding_list)
    
    with open('/home/dipika16/CS6203/meta_learning_features/pca_embedding.pkl', 'wb') as pcafile:
        pickle.dump(pca_embedding, pcafile)
    
    print('PCA has been performed')
