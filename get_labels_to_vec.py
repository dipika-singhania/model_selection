import numpy as np
import csv

def load_glove_embed():
    embeddings_dict = {}
    
    with open("glove/glove.6B.50d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict
    

def get_word_list_embeddings(words_list):
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

    vector_list = np.max(vector_list, axis=0)
    print("Shape of vector =", vector_list.shape)

    return vector_list

def add_vector_to_file(name, img_height, img_width, total_train_size, num_of_classes, labels_features):
    final_list = [name,img_height, img_width, total_train_size,num_of_classes]
    final_list.extend(labels_features.tolist())
    csv_file_name = 'dataset_model_features/dataset.csv'
    with open(csv_file_name, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(final_list)
    
