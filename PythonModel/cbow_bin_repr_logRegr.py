from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Permute, Multiply, Concatenate
import os
import json
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dot, \
    Reshape
from tensorflow.keras.optimizers import Adam
from collections import OrderedDict
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from knn_tensorflow import KNeighborsClassifier
import random
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from itertools import combinations
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import GradientBoostingClassifier


NUM_HEROES = 124  # Общее количество героев в Dota 2
DIM = 248   # Количество контекстных героев
EMBEDDING_SIZE = 150




# Загрузка данных из JSON
folder_path = 'C:\\Users\\Vladimir\\Desktop\\DotaPicker\\DotaPicker\\DotaProject\\bin\\Debug\\net6.0\\dataset\\newIDs'

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
    binary_repr = [0] * 124
    for hero in heroes:
        binary_repr[hero] = 1
    return binary_repr


cbow_contexts = []
cbow_label_heroes = []


def get_context_and_label(team, contr_team):
#     contr_combs = combinations(team, 4)
    combs = list(
        map(lambda comb: list(comb),
        combinations(team, 4))
        )
#     contr_combs = list(map(lambda comb: list(comb), contr_combs))

    for i in range(len(combs)):
#         for j in range(len(contr_combs)):
        binary_feature = heroes_to_binary_representation(combs[i]) \
                     + heroes_to_binary_representation(contr_team)
        cbow_contexts.append(binary_feature)
        cbow_label_heroes.append(np.array(team[-(i + 1)]))




print('Get data...')
m = 0
v = 0
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            matches = json.load(f)
            v += len(matches.values())
            for match in matches.values():

                radiant_heroes_order = dict(
                    [(pick['hero_id'], pick['order']) for pick in
                     match['PicksBans'] if
                     pick['team'] == 0 and pick['is_pick']])
                dire_heroes_order = dict(
                    [(pick['hero_id'], pick['order']) for pick
                     in match['PicksBans']
                     if pick['team'] == 1 and pick['is_pick']])



                correct = True
                correct_radiant_picks_order = dict()
                correct_dire_picks_order = dict()
                for player in match['Players']:
                    id = player['hero_id']
                    if id in radiant_heroes_order:
                        correct_radiant_picks_order[
                            radiant_heroes_order[id]] = id
                    elif id in dire_heroes_order:
                        correct_dire_picks_order[dire_heroes_order[id]] = id
                    else:
                        correct = False
                        break

                if not correct:
                    m += 1
                    continue

                radiant_picks = list(map(lambda x: hero_id_to_index(x[1]),
                                         sorted(
                                             correct_radiant_picks_order.items(),
                                             key=lambda x: x[0])))
                dire_picks = list(map(lambda x: hero_id_to_index(x[1]),
                                      sorted(correct_dire_picks_order.items(),
                                             key=lambda x: x[0])))



                if len(radiant_picks) != 5 or len(dire_picks) != 5:
                    continue

                all_features.append(radiant_picks + dire_picks)
                labels.append(1 if match['RadiantWin'] else 0)


                if match['RadiantWin']:
                    get_context_and_label(radiant_picks, dire_picks)
                else:
                    get_context_and_label(dire_picks, radiant_picks)

#                 #get_context_and_label(radiant_picks, dire_picks)
#                 get_context_and_label(dire_picks, radiant_picks)

for feature in all_features:
    binary_feature = heroes_to_binary_representation(feature[:5]) \
                     + heroes_to_binary_representation(feature[5:])
    cbow_all_features.append(binary_feature)

'''
                if len(radiant_heroes) > 5 or len(dire_heroes) > 5:
                    k += 1
                    print('ATTTTTTEN', 'k', k)
                    print(match['MatchId'])
                    print(dire_heroes)
                    print(radiant_heroes)
                    continue

                if len(radiant_heroes) < 5 or len(dire_heroes) < 5:
                    continue
                match_features = dict()

                #for i in range(len(radiant_heroes)):
                #    radiant_hero = radiant_heroes.pop()
                #    dire_hero = dire_heroes.pop()
                #    match_features[i] = radiant_hero
                #    match_features[i + 5] = dire_hero

                for pick in match['PicksBans']:
                    if pick['is_pick']:
                            match_features[int(pick['order'])] = pick['hero_id']


                match_features = heroes_to_binary_representation(
                    radiant_heroes) + heroes_to_binary_representation(
                    dire_heroes)

                all_features.append(match_features)

                labels.append(1 if match['RadiantWin'] else 0)

                '''

print(v)
print(m)
features = np.array(all_features)
cbow_features = np.array(cbow_all_features)
labels = np.array(labels)

#num_heroes = 124


print('Create CBOW model...')


model = Sequential()
model.add(Embedding(input_dim=NUM_HEROES, output_dim=EMBEDDING_SIZE, input_length=DIM))
model.add(Flatten())
dropp = True
dense_num = 5
for i in range(dense_num):
    model.add(Dense(EMBEDDING_SIZE, activation='relu'))
    if dropp:
        model.add(Dropout(0.5))

model.add(Dense(NUM_HEROES, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


cbow_contexts = np.array(cbow_contexts)
cbow_label_heroes = np.array(cbow_label_heroes)

# Разделение данных на обучающий, валидационный и тестовый наборы
train_cbow_features, temp_cbow_features, \
    train_cbow_labels, temp_cbow_labels = train_test_split(
        cbow_contexts, cbow_label_heroes, test_size=0.11,
        random_state=42)  # Оставляем 10% для тестового и 1% для валидационного наборов

# Дальнейшее разделение для создания валидационного набора
val_cbow_features, test_cbow_features, val_cbow_labels, test_cbow_labels = train_test_split(
    temp_cbow_features, temp_cbow_labels, test_size=0.91,
    random_state=42)  # 10% от изначального размера для тестирования, остальное - для валидации



epochs = 2
shuffle = True
context_len = len(cbow_contexts)
val = True
workers = 8
model.fit(cbow_contexts, cbow_label_heroes, validation_data=(val_cbow_features, val_cbow_labels),
          epochs=epochs, batch_size=2048, shuffle=True, workers=workers,
          use_multiprocessing=True)

loss, accuracy = model.evaluate(test_cbow_features, test_cbow_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
name = f'cbow_model_dim_{DIM}_epoch_{epochs}_emb_{EMBEDDING_SIZE}_workers_{workers}_val_{val}_con_len_{context_len}'

if shuffle:
    name += '_shuffle'
if dropp:
    name += '_dropout'
model.save(name + '.h5')

embeddings = model.layers[0].get_weights()[0]


embedding_weights_path = 'C:\\Users\\Vladimir\\Desktop\\DotaPicker\\embedding_weights.npy'
np.save(embedding_weights_path, embeddings)
print(f"Embedding weights saved to {embedding_weights_path}")

# for el in embeddings:
#     print(el)

heroes = ['Techies', 'Alchemist', 'Chaos Knight', 'Nature\'s Prophet', 'Enigma', 'Vengeful Spirit', 'Sand King', 'Undying', 'Doom', 'Bloodseeker', 'Mars', 'Skywrath Mage', 'Grimstroke', 'Lina', 'Dragon Knight', 'Juggernaut', 'Magnus', 'Naga Siren', 'Phantom Lancer', 'Omniknight', 'Warlock', 'Batrider', 'Faceless Void', 'Slark', 'Lycan', 'Terrorblade', 'Tidehunter', 'Broodmother', 'Arc Warden']

with open('name_id.json') as file:
    name_id_dict = json.load(file)
with open('id_name.json') as file:
    id_name_dict = json.load(file)
heroes = name_id_dict.keys()
hero_ids = list(map(lambda name: name_id_dict[name], heroes))
hero_embeddings = list(map(lambda id: embeddings[id], hero_ids))


tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = list(embeddings)

data_2d = tsne.fit_transform(embeddings)

plt.scatter(data_2d[:, 0], data_2d[:, 1])

print(id_name_dict)

for id, vector in enumerate(embeddings):
    plt.annotate(id_name_dict[str(id)], (data_2d[id, 0], data_2d[id, 1]))

plt.title("Визуализация 50-мерных данных в 2D с помощью t-SNE")
plt.xlabel("Компонент 1")
plt.ylabel("Компонент 2")
plt.show()




def change_features_to_train_data(features):
    embedding_features = np.zeros((len(features), 10, EMBEDDING_SIZE))
    for i in range(len(features)):
        for j in range(10):
            embedding_features[i, j] = embeddings[features[i][j]]
    return embedding_features


def create_coefs(features):
    contr_coefs_radiant = np.zeros(
        (len(features), 5, 1))  # Make sure to have the right shape
    contr_coefs_dire = np.zeros(
        (len(features), 5, 1))  # Make sure to have the right shape

    with open('name_id.json') as file:
        name_id_dict = json.load(file)
    with open('id_name.json') as file:
        id_name_dict = json.load(file)
    with open('HeroList.json') as file:
        hero_list = json.load(file)

    for i in range(len(features)):
        for j in range(5):
#             name_radiant = id_name_dict[str(features[i][j])]
#             name_dire = id_name_dict[str(features[i][5 + j])]
#             percents = hero_list[name_radiant]['versus'][name_dire]
#             radiant_coeff = float(percents[:-1]) / 100.0
#             contr_coefs_radiant[i, j, 0] = radiant_coeff
#             contr_coefs_dire[i, j, 0] = 1 - radiant_coeff

            contr_coefs_radiant[i, j, 0] = 1
            contr_coefs_dire[i, j, 0] = 1

    return contr_coefs_radiant, contr_coefs_dire



# Разделение данных на обучающий, валидационный и тестовый наборы
train_features, temp_features, train_labels, temp_labels = train_test_split(
    features, labels, test_size=0.11,
    random_state=42)  # Оставляем 10% для тестового и 1% для валидационного наборов

# Дальнейшее разделение для создания валидационного набора
val_features, test_features, val_labels, test_labels = train_test_split(
    temp_features, temp_labels, test_size=0.91,
    random_state=42)  # 10% от изначального размера для тестирования, остальное - для валидации

# Тренировка модели на обучающем наборе
# model, callbacks = create_model()

train_data = change_features_to_train_data(
    features=train_features)

val_data = change_features_to_train_data(features=val_features)

'''
def create_logistic_regression_model():
    input_shape = (10, EMBEDDING_SIZE)
    model_input = Input(shape=input_shape, name='input_heroes')
    flatten_input = Flatten()(model_input)
    hidden_layer1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(flatten_input)
    dropout1 = Dropout(0.5)(hidden_layer1)
    hidden_layer2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dropout1)
    dropout2 = Dropout(0.5)(hidden_layer2)
    output = Dense(1, activation='sigmoid', name='output')(dropout2)
    model = Model(inputs=model_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
'''


def create_logistic_regression_model():
    input_shape = (10, EMBEDDING_SIZE)
    model_input = Input(shape=input_shape, name='input_heroes')
    flatten_input = Flatten()(model_input)

    #hidden_layer1 = Dense(8, activation='relu')(flatten_input)
    #dropout1 = Dropout(0.5)(hidden_layer1)
    #hidden_layer2 = Dense(4, activation='relu')(dropout1)
    #dropout2 = Dropout(0.5)(hidden_layer2)
    #softmax
    output = Dense(1, activation='sigmoid', name='output')(flatten_input)

    model = Model(inputs=model_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

#
# train_data = []
# for i in range(0, len(train_radiant)):
#     train_data.append(train_radiant[i] + train_dire[i])
#
# val_data = []
# for i in range(0 , len(val_radiant)):
#     val_data.append(val_radiant[i] + val_dire[i])


#train_labels = np.squeeze(train_labels)
#val_labels = np.squeeze(val_labels)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
train_data = np.array(train_data)
val_data = np.array(val_data)

model = create_logistic_regression_model()

print(len(train_data[5]))

model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
          epochs=1000, batch_size=2000, shuffle=True,
          use_multiprocessing=True)

model_save_path = "C:\\Users\\Vladimir\\Desktop\\DotaPicker\\trained_model3.h5"
model.save(model_save_path)

test_data = change_features_to_train_data(features=test_features)


loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

