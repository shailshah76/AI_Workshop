from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


def buildCNNModel(input_shape, num_classes,num_channels,
                  kernel_size,dropout,pool_size,stride):
    model = Sequential()
    model.add(Conv2D(num_channels,kernel_size,padding= 'valid',strides=stride,
                     input_shape = input_shape))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(num_channels,kernel_size))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
    

