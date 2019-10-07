import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

# Размер мини-выборки
batch_size = 32
# Количество классов изображений
nb_classes = 10
# Количество эпох для обучения
nb_epoch = 25
# Размер изображений
img_rows, img_cols = 32, 32
# Количество каналов в изображении: RGB
img_channels = 3

numpy.random.seed(42)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

"""
Необходимо выполнить предварительную обработку данных.
Данные об интенсивности пикселей изображения необходимо нормализовать, чтобы все они находились в диапазоне
от 0 до 1, для этого этого преобразуем их в тип float32 и делим на 255.
"""

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

"""
Метки классов необходимо преобразовать в категории. На выходе наша сеть имеет 10 нейронов и выходной сигнал
нейронов, соответствует вероятности того, что изображение принадлежит к данному классу, соответственно номера классов в 
метках мы должны преобразовать в представления по категориям.
"""

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

"""
Convolution2D - свёрточный слой, который работает с двухмерными данными, этот слой будет иметь 32 карты признаков
размер ядра свёртки на каждой карте 3 на 3, размер входных данных 3 на 32 на 32, что соответствует 3м каналам изображения
для кодов трёх цветов ргб, размер изображения 32 на 32.
"""

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
# Второй слой
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Слой подвыборки, pool_size - размер уменьшения размерности 2 на 2.
# MaxPooling2D из квадратика 2 на 2 выбирается макс. значение
model.add(MaxPooling2D(pool_size=(2, 2)))

"""
После каскада из двух свёрточных слоёв и слоя подвыборки, мы добавляем слой регуляризации.
"""

model.add(Dropout(0.25))
# Третий слой, больше карт признаков - 64
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# Четвёртый слой
model.add(Conv2D(64, (3, 3), activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())

"""
Классификатор, по признакам найденных свёрточной сетью, выполняет определение к какому конкретно классу
принадлежит объект на картинке.
Начало нужно преобразовать сеть из двумерного представления в плоское - слой Flatten
Затем добавляем два полно связных слоя Dense.
Суммарное значение всех 10 нейронов = 1
"""

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # выключает нейроны с вероятностью 50%
model.add(Dense(10, activation='softmax'))

# Компилируем сеть
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

# Обучаем сесть
# shuffle - перемешивать данные в начале каждой эпохи
model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_split=0.1,
    shuffle=True, verbose=2
)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))