
from keras.layers import Dense, Conv1D, ELU, Flatten, Dropout, BatchNormalization, MaxPooling1D,Activation, Add,GlobalAveragePooling1D
from keras.models import Model
from keras.layers import Input
from keras.optimizer_v2.adam import Adam
from tensorflow.keras import Input

def my_model():
    input_layer = Input(shape=(2048*4, 1))

    conv1 = Conv1D(32, kernel_size=32, strides=1)(input_layer)
    elu1 = ELU()(conv1)

    conv2 = Conv1D(32, kernel_size=32, strides=1)(elu1)
    max_pool1 = MaxPooling1D(pool_size=8, strides=8)(conv2)
    elu2 = ELU()(max_pool1)

    conv3 = Conv1D(64, kernel_size=16, strides=1)(elu2)
    elu3 = ELU()(conv3)

    conv4 = Conv1D(64, kernel_size=16, strides=1)(elu3)
    max_pool2 = MaxPooling1D(pool_size=4, strides=4)(conv4)
    elu4 = ELU()(max_pool2)

    conv5 = Conv1D(128, kernel_size=8, strides=1)(elu4)
    elu5 = ELU()(conv5)

    conv6 = Conv1D(128, kernel_size=8, strides=1)(elu5)
    max_pool3 = MaxPooling1D(pool_size=4, strides=4)(conv6)
    elu5 = ELU()(max_pool3)


    flatten = Flatten()(elu5)

    dense1 = Dense(64, activation='elu')(flatten)
    dropout1 = Dropout(0.5)(dense1)

    dense2 = Dense(32, activation='elu')(dropout1)
    out = Dense(2, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(optimizer=Adam(learning_rate=1.5e-5), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model