from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import recall_score, f1_score, precision_score
import pickle

data = load_iris()

X = data['data']
y = data['target']
labels = data['target_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=420)



clf = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

recall = recall_score(y_test, predictions, average='weighted')
precision = precision_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')


print(f'recall {recall}')
print(f'precision {precision}')
print(f'f1 score {f1}')


with open('registry/clf.pk', 'wb') as c:
    pickle.dump(clf, c)


print('ok')
