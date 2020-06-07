import joblib
import numpy as np

class Model():
    def __init__(self):
        self.model = self.load_model()
        self.scaler = self.load_scaler()

    def load_model(self):
        return joblib.load('registry/clf.save')

    def load_scaler(self):
        return joblib.load('registry/scaler.save')

    def get_label(self, prediction):
        map = {0:'setosa',
               1:'versicolor',
               2:'virginica'}
        return map[prediction[0]]

    def preprocess(self, features):
        print(f'features at start of prepross {features}')
        features = np.array(features)
        features = features.reshape(1, -1)
        print(f'!!!!!!!features {features}')
        return self.scaler.transform(features)

    def predict(self, features):
        features = self.preprocess(features)
        features = np.array(features).reshape(1, -1)
        print(f'features at before predict {features}')
        prediction = self.model.predict(features)
        return self.get_label(prediction)
