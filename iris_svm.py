from sklearn import svm
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
iris_dataset = datasets.load_iris()
X = iris_dataset.data[:, 2]
y = iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
X_train_mod = X_train.reshape(-1, 1)
X_test_mod = X_test.reshape(-1, 1)
y_train_mod = y_train.reshape(-1, 1)
y_test_mod = y_test.reshape(-1, 1)
model = svm.SVC(kernel='linear')
model.fit(X_train_mod, y_train_mod)
y_pred_mod = model.predict(X_test_mod)
print(accuracy_score(y_test_mod, y_pred_mod))
