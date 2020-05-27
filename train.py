from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, precision_score
import pickle

data = load_iris()

X = data['data']
y = data['target']
labels = data['target_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=420)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

clf = LogisticRegression()
clf.fit(X_train, y_train)

X_test = scaler.transform(X_test)
predictions = clf.predict(X_test)

recall = recall_score(y_test, predictions, average='weighted')
precision = precision_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')


print(f'recall {recall}')
print(f'precision {precision}')
print(f'f1 score {f1}')


with open('registry/clf.pk', 'wb') as c:
    pickle.dump(clf, c)

with open('registry/scaler.pk', 'wb') as s:
    pickle.dump(scaler, s)

print('ok')