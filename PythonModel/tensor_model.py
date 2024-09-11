import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_logistic_regression_model(input_dim):
    model = Sequential()
    model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

#Main classification procedure
def classifyKNN (trainData, testData, k, numberOfClasses = 11):
    def dist(q, X, **kwargs):
        NUM_IN_QUERY = 10000
        and_sum = 0
        for i in range(len(X)):
            if q[i] and X[i]:
                and_sum += 1
        weight = and_sum * and_sum * and_sum * and_sum / NUM_IN_QUERY
        dist = (1 / (weight + 1e-8)) - 1
        return dist

    testLabels = []
    for testPoint in testData:
        #Claculate distances between test point and all of the train points
        testDist = [(dist(testPoint, trainData[i][0]), trainData[i][1])] for i in range(len(trainData))]
        #How many points of each class among nearest K
        stat = [0 for i in range(numberOfClasses)]
        for d in sorted(testDist)[0:k]:
            stat[d[1]] += 1
        #Assign a class with the most number of occurences among K nearest neighbours
        testLabels.append( sorted(zip(stat, range(numberOfClasses)), reverse=True)[0][1] )
    return testLabels

# Предполагая, что у вас уже есть подготовленные данные: X_train, X_test, y_train, y_test
input_dim = X_train.shape[1]  # Количество признаков
model = build_logistic_regression_model(input_dim)

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# Предсказание и оценка точности
_, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')
