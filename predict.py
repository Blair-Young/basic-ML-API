import pickle
import numpy as np

class Model():
    def __init__(self):
        self.model = self.load_model()
        self.scaler = self.load_scaler()

    def load_model(self):
        with open('registry/clf.pk', 'rb') as f:
            return pickle.load(f)

    def load_scaler(self):
        with open('registry/scaler.pk', 'rb') as f:
            return pickle.load(f)

    def get_label(self, prediction):
        map = {0:'setosa',
               1:'versicolor',
               2:'virginica'}
        return map[prediction[0]]

    def preprocess(self, features):
        features = np.array(features)
        features = features.reshape(1, -1)
        print(f'features pre transform {features}')
        return self.scaler.transform(features)

    def predict(self, features):
        features = self.preprocess(features)
        prediction = self.model.predict(features)
        return self.get_label(prediction)




clf =  Model()
print(clf.predict(np.array([5.1, 3.5, 1.4, 0.2])))





