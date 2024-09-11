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

num_heroes = 124
embedding_size = 50


print('Create CBOW model...')
# Создадим CBOW модель
input_layer = Input(shape=(num_heroes*2,))
embedding_layer = Embedding(input_dim=num_heroes, output_dim=embedding_size)(input_layer)
flatten_layer = Flatten()(embedding_layer)
hidden_layer = Dense(128, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(inputs=input_layer, outputs=output_layer)

# Компилируем модель
print('Compile CBOW model...')
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Обучаем модель
print('Train CBOW model...')
model.fit(cbow_features, labels, epochs=10, batch_size=512)

# Получаем веса слоя эмбеддингов, что соответствует векторным представлениям героев
print('Get CBOW results...')
embeddings = model.layers[1].get_weights()[0]

# Сохранение весов эмбеддингов
embedding_weights_path = 'C:\\Users\\Vladimir\\Desktop\\DotaPicker\\embedding_weights.npy'
np.save(embedding_weights_path, embeddings)
print(f"Embedding weights saved to {embedding_weights_path}")

for el in embeddings:
    print(el)

heroes = ['Techies', 'Alchemist', 'Chaos Knight', 'Nature\'s Prophet', 'Enigma', 'Vengeful Spirit', 'Sand King', 'Undying', 'Doom', 'Bloodseeker', 'Mars', 'Skywrath Mage', 'Grimstroke', 'Lina', 'Dragon Knight', 'Juggernaut', 'Magnus', 'Naga Siren', 'Phantom Lancer', 'Omniknight', 'Warlock', 'Batrider', 'Faceless Void', 'Slark', 'Lycan', 'Terrorblade', 'Tidehunter', 'Broodmother', 'Arc Warden']

with open('name_id.json') as file:
    name_id_dict = json.load(file)
with open('id_name.json') as file:
    id_name_dict = json.load(file)
heroes = name_id_dict.keys()
hero_ids = list(map(lambda name: name_id_dict[name], heroes))
hero_embeddings = list(map(lambda id: embeddings[id], hero_ids))


# Применяем t-SNE для уменьшения размерности до 2D
tsne = TSNE(n_components=2, random_state=0)
data_2d = tsne.fit_transform(embeddings)

# Визуализация 2D данных
plt.scatter(data_2d[:, 0], data_2d[:, 1])

print(id_name_dict)
# Добавление подписей к каждой точке
for id, vector in enumerate(embeddings):
    print(id_name_dict[str(id)])
    plt.annotate(id_name_dict[str(id)], (data_2d[id, 0], data_2d[id, 1]))

plt.title("Визуализация 50-мерных данных в 2D с помощью t-SNE")
plt.xlabel("Компонент 1")
plt.ylabel("Компонент 2")
plt.show()

'''
Z = linkage(hero_embeddings, method='ward')

# Построение дендрограммы
# Построение дендрограммы
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=heroes, orientation='right', leaf_rotation=0, leaf_font_size=10, color_threshold=0)

# Подпись осей
plt.xlabel('Distance')
plt.ylabel('Heroes')
plt.title('Heroes Dendrogram')

# Показать дендрограмму
plt.show()
'''




class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        output = K.sum(output, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        # Change the output shape to reflect the fact that you're returning a single vector per sample
        return (input_shape[0], input_shape[-1])


def create_model():
    # Определение входов модели
    input_1 = Input(shape=(5, 150), name='input_1')  # Radiant team LSTM
    input_2 = Input(shape=(5, 150), name='input_2')  # Dire team LSTM
    input_3 = Input(shape=(5, 1), name='input_3')  # Radiant team multiplier
    input_4 = Input(shape=(5, 1), name='input_4')  # Dire team multiplier

    # Определение LSTM слоев для обоих входов
    lstm_out_1 = LSTM(units=150, return_sequences=True, activation='tanh')(
        input_1)
    lstm_out_2 = LSTM(units=150, return_sequences=True, activation='tanh')(
        input_2)

    # Перестановка размерностей входных данных для подачи в Dense слои
    permute_1 = Permute((2, 1))(lstm_out_1)
    permute_2 = Permute((2, 1))(lstm_out_2)

    # Полносвязные слои с Softmax активацией для выходов LSTM
    dense_out_1 = Dense(5, activation='softmax')(permute_1)
    dense_out_2 = Dense(5, activation='softmax')(permute_2)

    # Повторная перестановка размерностей после Dense слоев
    permute_3 = Permute((2, 1))(dense_out_1)
    permute_4 = Permute((2, 1))(dense_out_2)

    # Поэлементное умножение выходов Dense и соответствующих множителей
    multiply_1 = Multiply()([permute_3, input_3])
    multiply_2 = Multiply()([permute_4, input_4])

    # Конкатенация результатов умножения
    concatenate_out = Concatenate(axis=1)([multiply_1, multiply_2])

    # Дополнительные полносвязные слои
    dense_1 = Dense(64, activation='relu')(concatenate_out)
    dense_2 = Dense(64, activation='relu')(dense_1)

    # Выравнивание выходных данных
    flatten_out = Flatten()(dense_2)

    # Финальный полносвязный слой с Sigmoid активацией для бинарной классификации
    final_output = Dense(1, activation='sigmoid')(flatten_out)#, name='dense_4')(flatten_out)

    # Создание модели
    model = Model(inputs=[input_1, input_2, input_3, input_4],
                  outputs=final_output)

    # Компиляция модели
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def change_features_to_train_data(features):
    radiant = np.zeros((len(features), 5, embedding_size))
    dire = np.zeros((len(features), 5, embedding_size))
    for i in range(len(features)):
        for j in range(5):
            radiant[i, j] = embeddings[features[i][j]]
            dire[i, j] = embeddings[features[i][j + 5]]
    return radiant, dire


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
            name_radiant = id_name_dict[str(features[i][j])]
            name_dire = id_name_dict[str(features[i][5 + j])]
            percents = hero_list[name_radiant]['versus'][name_dire]
            radiant_coeff = float(percents[:-1]) / 100.0
            contr_coefs_radiant[i, j, 0] = radiant_coeff
            contr_coefs_dire[i, j, 0] = 1 - radiant_coeff

    return contr_coefs_radiant, contr_coefs_dire


from sklearn.model_selection import train_test_split

# Разделение данных на обучающий, валидационный и тестовый наборы
train_features, temp_features, train_labels, temp_labels = train_test_split(
    features, labels, test_size=0.11,
    random_state=42)  # Оставляем 10% для тестового и 1% для валидационного наборов

# Дальнейшее разделение для создания валидационного набора
val_features, test_features, val_labels, test_labels = train_test_split(
    temp_features, temp_labels, test_size=0.91,
    random_state=42)  # 10% от изначального размера для тестирования, остальное - для валидации

# Тренировка модели на обучающем наборе
model = create_model()

train_radiant, train_dire = change_features_to_train_data(
    features=train_features)
train_radiant_coeffs, train_dire_coeffs = create_coefs(features=train_features)

val_radiant, val_dire = change_features_to_train_data(features=val_features)
val_radiant_coeffs, val_dire_coeffs = create_coefs(features=val_features)

# Данные должны быть в формате списка массивов numpy
train_data = [train_radiant, train_dire, train_radiant_coeffs,
              train_dire_coeffs]
val_data = [val_radiant, val_dire, val_radiant_coeffs, val_dire_coeffs]

train_labels = np.squeeze(train_labels)
val_labels = np.squeeze(val_labels)

# Make sure labels are 1D
train_labels = np.array(train_labels).flatten()
val_labels = np.array(val_labels).flatten()

print([x.shape for x in train_data])
print(train_labels.shape)
print([x.shape for x in val_data])
print(val_labels.shape)

# Используйте этот формат для обучения
model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
          epochs=50, batch_size=64, shuffle=True)

model_save_path = "C:\\Users\\Vladimir\\Desktop\\DotaPicker\\trained_model3.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Preprocess test_features
test_radiant, test_dire = change_features_to_train_data(features=test_features)
test_radiant_coeffs, test_dire_coeffs = create_coefs(features=test_features)

# Prepare the data in the format expected by the model
test_data = [test_radiant, test_dire, test_radiant_coeffs, test_dire_coeffs]

# Now evaluate the model with the correctly formatted data
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Предсказания для тестового набора
predictions = model.predict(test_features)

threshold = 0.5

correct_predictions = 0
predictions_count = 0
for i in range(len(test_labels)):
    radiant_win_probability = predictions[i][0]
    dire_win_probability = 1 - radiant_win_probability

    if radiant_win_probability > threshold:
        predicted_label = 1  # Предсказываем победу Radiant
    elif dire_win_probability > threshold:
        predicted_label = 0  # Предсказываем победу Dire
    else:
        predicted_label = -1  # Нейросеть не дает предсказания

    if predicted_label != -1:
        predictions_count += 1
        if predicted_label == test_labels[i]:
            correct_predictions += 1

accuracy = (correct_predictions / predictions_count) * 100
print(f"Accuracy on test data (with threshold {threshold}): {accuracy:.2f}%")
print(str(predictions_count) + ' ' + '/' + ' ' + str(len(test_labels)))

