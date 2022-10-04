import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D
import os
import numpy as np
import cv2

mapping = ["0","1","2","3","4","5","6","7","8","9",
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def train_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
    # image_index = 123 
    # print(y_train[image_index])
    # cv2.imshow("number", x_train[image_index])
    # cv2.waitKey()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255


    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(4,4), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.4))
    model.add(Dense(10,activation=tf.nn.softmax))


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=10)

    model.save('my_model')
    # print(model.evaluate(x_test, y_test))

    # image_index = 4444
    # cv2.imshow("image", x_test[image_index].reshape(28, 28))
    # pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
    # print(pred.argmax())


def train_numerical():
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = []
    y_train = []
    path = "./train/"
    for x in ["BadImg/Bmp/", "GoodImg/Bmp/"]:
        for y in list(os.listdir(path +x)):
            if int(y.replace("Sample","")) <= 10:
                full_path = path +x+ y+"/"
                for im in list(os.listdir(full_path)):
                    image = cv2.imread(full_path + im, cv2.IMREAD_GRAYSCALE)
                    im1 = cv2.resize(image, dsize=(28, 28))
                    im1_array = np.array(im1) / 255
                    x_train.append(im1_array)
                    y_train.append(int(y.replace("Sample","")) - 1)
    print("preprocessing complete")

    # image_index = 123 
    # print(y_train[image_index])
    # cv2.imshow("number", x_train[image_index])
    # cv2.waitKey()


    y_train = np.array(y_train)
    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(4,4), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.4))
    model.add(Dense(10,activation=tf.nn.softmax))


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=60)

    model.save('my_model_chars_74')

def train_cap_letters():
    x_train = []
    y_train = []
    path = "./train/"
    for x in ["BadImg/Bmp/", "GoodImg/Bmp/"]:
        for y in list(os.listdir(path +x)):
            if int(y.replace("Sample","")) > 10 and int(y.replace("Sample","")) <= 36:
                full_path = path +x+ y+"/"
                for im in list(os.listdir(full_path)):
                    image = cv2.imread(full_path + im, cv2.IMREAD_GRAYSCALE)
                    im1 = cv2.resize(image, dsize=(28, 28))
                    im1_array = np.array(im1) / 255
                    x_train.append(im1_array)
                    y_train.append(int(y.replace("Sample","")) - 11)
    print("preprocessing complete")

    y_train = np.array(y_train)
    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(4,4), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.4))
    model.add(Dense(26,activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=60)

    model.save('my_model_alphabet')

if __name__ == "__main__":
    # train_numerical()
    train_cap_letters()