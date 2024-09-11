from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow
#from keras.layers import LSTM, Permute, Multiply, Concatenate
import os
import json
#from keras.models import Model, Sequential
#from keras.layers import Input, Dense, Embedding, Flatten, Dot, Reshape
#from keras.optimizers import Adam
from collections import OrderedDict
#from keras.layers import Layer
#import keras.backend as K
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from knn_tensorflow import KNeighborsClassifier



# Загрузка данных из JSON
folder_path = 'C:\\Users\\Vladimir\\Desktop\\DotaPicker\\DotaPicker\\DotaProject\\bin\\Debug\\net6.0\\dataset\\test'

all_features = []  # Список всех матчей с id геров в команде
cbow_all_features = []  # Список всех матчей с бинарными представлениями команд
labels = []


def hero_id_to_index(hero_id):
    """Конвертировать hero_id в индекс для бинарного представления."""
    if hero_id == 126: return 0
    if hero_id == 128: return 24
    if hero_id == 129: return 115
    if hero_id == 135: return 116
    if hero_id == 136: return 117
    if hero_id == 137: return 118
    if hero_id == 138: return 122
    return hero_id


def heroes_to_binary_representation(heroes):
    """Преобразовать список героев в бинарное представление."""
    binary_repr = [False] * 124
    for hero in heroes:
        binary_repr[hero] = True
    return binary_repr

print('Get data...')
m = 0
v = 0
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            matches = json.load(f)
            v += len(matches.values())
            for match in matches.values():
                radiant_heroes_order = dict([(pick['hero_id'], pick['order']) for pick in
                                  match['PicksBans'] if
                                  pick['team'] == 0 and pick['is_pick']])
                dire_heroes_order = dict([(pick['hero_id'], pick['order']) for pick
                                in match['PicksBans']
                               if pick['team'] == 1 and pick['is_pick']])

                correct = True
                correct_radiant_picks_order = dict()
                correct_dire_picks_order = dict()
                for player in match['Players']:
                    id = player['hero_id']
                    if id in radiant_heroes_order:
                        correct_radiant_picks_order[radiant_heroes_order[id]] = id
                    elif id in dire_heroes_order:
                        correct_dire_picks_order[dire_heroes_order[id]] = id
                    else:
                        correct = False
                        break

                if not correct:
                    m += 1
                    continue

                radiant_picks = list(map(lambda x: hero_id_to_index(x[1]),
                                     sorted(correct_radiant_picks_order.items(),
                                            key=lambda x: x[0])))
                dire_picks = list(map(lambda x: hero_id_to_index(x[1]),
                                      sorted(correct_dire_picks_order.items(),
                                             key=lambda x: x[0])))

                if len(radiant_picks) != 5 or len(dire_picks) != 5:
                    continue

                all_features.append(radiant_picks + dire_picks)
                labels.append(1 if match['RadiantWin'] else 0)

for feature in all_features:
    binary_feature = heroes_to_binary_representation(feature[:5])\
        + heroes_to_binary_representation(feature[5:])
    cbow_all_features.append(binary_feature)


print(len(cbow_all_features))

k = 0
def custom_metric(q, X, **kwargs):
    global k
    k += 1
    NUM_IN_QUERY = 10000
    and_sum = 0
    for i in range(len(X)):
        if q[i] and X[i]:
            and_sum += 1
    #and_operation = np.logical_and(q, X).astype(int)
    #weight = (np.sum(and_operation) / NUM_IN_QUERY) ** 4
    weight = and_sum * and_sum * and_sum * and_sum / NUM_IN_QUERY
    dist = (1 / (weight + 1e-8)) - 1
    return dist


class KNNLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, **logreg_params):
        self.n_neighbors = n_neighbors
        self.logreg_params = logreg_params
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='manhattan')
        self.logreg = LogisticRegression(**logreg_params)

    def fit(self, X, y):
        # Train the KNN model
        self.knn.fit(X, y)

        # Find the K-nearest neighbors for each sample
        distances, indices, pred = self.knn._predict(X)
        print('distances all')
        # Use distances to generate new features
        # For example, take the inverse of the distance as weights for averaging class labels
        # Prevent division by zero in case of zero distance
        weights = list(map(lambda d: 1 / (d + 1), distances))

        # Calculate weighted average of class labels based on distances
        knn_features = np.array(
            [np.average(y[indices[i]], weights=weights[i]) for i in
             range(len(X))]).reshape(-1, 1)

        # Combine original features with KNN features
        combined_features = np.hstack((X, knn_features))

        # Train the logistic regression model on the combined features
        self.logreg.fit(combined_features, y)

    def predict(self, X):
        # Find the K-nearest neighbors for each sample in the test set
        distances, indices = self.knn.kneighbors(X)

        # Generate new features based on the K-nearest neighbors
        knn_features = np.array([y_train[indices[i]].mean() for i in range(len(X))]).reshape(-1, 1)

        # Combine original features with KNN features
        combined_features = np.hstack((X, knn_features))

        # Predict using the logistic regression model
        return self.logreg.predict(combined_features)




train_accuracies = []
test_accuracies = []

import time
import joblib
#n_ki = [100,500,2000,5000,10000]
n_ki = [100,500,2000,5000,10000]
for n in n_ki:
    features = np.array(cbow_all_features[:n])
    new_labels = np.array(labels[:n])
    print(len(features), len(new_labels))
    X_train, X_test, y_train, y_test = train_test_split(features, new_labels, test_size=0.1, random_state=42)
    start = time.time()

    model = KNNLogisticRegression(int(n * 0.9))
    model.fit(X_train, y_train)  # Обучаем модель
    # Вычисляем точность на тренировочном наборе
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    train_accuracies.append(train_accuracy)

    # Вычисляем точность на тестовом наборе
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    test_accuracies.append(test_accuracy)

    joblib.dump(model, f'trained_model_{n}.joblib')
    end = time.time()

    print(n)
    print(end - start)
    print('train_accuracy:\t', train_accuracy)
    print('test_accuracy:\t', test_accuracy)
    print('вычисления',k)
    print()
    k = 0

# Визуализация
plt.plot(n_ki, train_accuracies, label='Train Accuracy')
plt.plot(n_ki, test_accuracies, label='Test Accuracy')
plt.xlabel('n_ki')
plt.ylabel('Accuracy')
plt.title('Training Progress')
plt.legend()
plt.show()



#model.fit(X_train, y_train)
#predictions = model.predict(X_test)
#accuracy = accuracy_score(y_test, predictions)
#print(accuracy)
