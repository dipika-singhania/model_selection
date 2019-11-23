import pandas as pd
import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

class TrainMetaModel():
    def __init__(self):
        self.model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    def read_file_save(self, file_name):
        self.data_frame = pd.read_csv(file_name)
    
    def train_and_predict(self):
        X = self.data_frame[self.data_frame.columns[1:-1]]
        y = self.data_frame[self.data_frame.columns[-1]]
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
        
        # Train Decision Tree Classifer
        clf = self.model.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test) 
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        


if __name__ == "__main__":
    obj = TrainMetaModel()
    obj.read_file_save("dataset_model_features/meta_model_learning.csv")
    model = obj.train_and_predict()
    
