import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json

numpy.random.seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Dense(800, input_dim=784, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=200, epochs=100, validation_split=0.2, verbose=2)

model_json = model.to_json()

"""
Сохранение сети в json
"""

with open('models/mnist_model.json', 'w+') as json_file:
    json_file.write(model_json)

"""
Сохранение весов модели
"""

model.save_weights('weights/mnist_weights.h5')

"""
Используем сохранёную сеть.
1) Загружаем данные об архитекртуре сети из файла json.
2) Загружаем сохранёные веса
"""

with open('models/mnist_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Создаем модель на основе загружаемых данных
loaded_model = model_from_json(loaded_model_json)

# Загружаем веса в модель
loaded_model.load_weights('weights/mnist_weights.h5')

# Компилируем модель
loaded_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# Проверяем модель на тестовых данных
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))


