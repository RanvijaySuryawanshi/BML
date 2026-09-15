import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as sc
from sklearn.neighbors import KNeighborsClassifier as kc
from sklearn.metrics import accuracy_score as acc

db = pd.read_csv('winequality-red.csv', sep=';')
X = db.drop('quality', axis=1)
y = db['quality']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=1, stratify=y)

scaler = sc()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_initial = 5
knn = kc(n_neighbors=k_initial)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

accuracy = acc(y_test, y_pred)
print(f"Accuracy: {accuracy}")
