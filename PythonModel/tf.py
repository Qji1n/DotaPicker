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
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dot, Reshape
from tensorflow.keras.optimizers import Adam
from collections import OrderedDict
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from knn_tensorflow import KNeighborsClassifier


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
embedding_size = 150


print('Create CBOW model...')
vector_dim = 150
window_size = 5  # Context window size
epochs = 10

# Define the Word2Vec Keras model
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(input_dim=num_heroes, output_dim=vector_dim, input_length=1, name='hero_embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# Setup a dot product operation to compare target and context vectors
dot_product = Dot([target, context], axes=1)
dot_product = Reshape((1,))(dot_product)
output = Dense(1, activation='sigmoid')(dot_product)

# Create the Word2Vec model
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train the model
model.fit([target, context], labels, epochs=epochs, batch_size=512, verbose=1)

# Extract embeddings
embeddings = model.get_layer('hero_embedding').get_weights()[0]

# Сохранение весов эмбеддингов
embedding_weights_path = 'C:\\Users\\Vladimir\\Desktop\\DotaPicker\\embedding_weights.npy'
np.save(embedding_weights_path, embeddings)
print(f"Embedding weights saved to {embedding_weights_path}")

for el in embeddings:
    print(el)


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
    # Входные данные
    radiant = Input(shape=(5, 150), name='input_1')
    dire = Input(shape=(5, 150), name='input_2')
    contr_coefs_radiant = Input(shape=(5, 1), name='input_3')
    contr_coefs_dire = Input(shape=(5, 1), name='input_4')

    # Обработка LSTM
    lstm_out_1 = LSTM(units=150, return_sequences=True)(
        radiant)  # Changed to False
    lstm_out_2 = LSTM(units=150, return_sequences=True)(
        dire)  # Changed to False

    # Механизм внимания
    attention_out_1 = AttentionLayer()(lstm_out_1)
    attention_out_2 = AttentionLayer()(lstm_out_2)

    # Умножение
    multiply_1 = Multiply()([attention_out_1, contr_coefs_radiant])
    multiply_2 = Multiply()([attention_out_2, contr_coefs_dire])

    # Объединение результатов
    concatenate_out = Concatenate(axis=-1)([multiply_1, multiply_2])
    flatten_out = Flatten()(concatenate_out)

    # Полносвязные слои
    dense_1 = Dense(64, activation='relu')(flatten_out)
    final_output = Dense(1, activation='sigmoid', name='final_output')(dense_1)

    print("Shape of attention_out_1:", attention_out_1.shape)
    print("Shape of contr_coefs_radiant:", contr_coefs_radiant.shape)
    print("Shape of attention_out_2:", attention_out_2.shape)
    print("Shape of contr_coefs_dire:", contr_coefs_dire.shape)

    # Создание модели
    model = Model(inputs=[radiant, dire, contr_coefs_radiant, contr_coefs_dire], outputs=final_output)

    # Компиляция модели
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
    contr_coefs_radiant = np.zeros((len(features), 5, 1)) # Make sure to have the right shape
    contr_coefs_dire = np.zeros((len(features), 5, 1)) # Make sure to have the right shape

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
    features, labels, test_size=0.11, random_state=42)  # Оставляем 10% для тестового и 1% для валидационного наборов

# Дальнейшее разделение для создания валидационного набора
val_features, test_features, val_labels, test_labels = train_test_split(
    temp_features, temp_labels, test_size=0.91, random_state=42)  # 10% от изначального размера для тестирования, остальное - для валидации


# Тренировка модели на обучающем наборе
model = create_model()

train_radiant, train_dire = change_features_to_train_data(features=train_features)
train_radiant_coeffs, train_dire_coeffs = create_coefs(features=train_features)

val_radiant, val_dire = change_features_to_train_data(features=val_features)
val_radiant_coeffs, val_dire_coeffs = create_coefs(features=val_features)

# Данные должны быть в формате списка массивов numpy
train_data = [train_radiant, train_dire, train_radiant_coeffs, train_dire_coeffs]
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
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=50, batch_size=64)



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
print(f"Test Accuracy: {accuracy*100:.2f}%")


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

