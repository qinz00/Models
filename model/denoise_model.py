from keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling1D, MaxPooling2D, Activation, Add, UpSampling1D, UpSampling2D, concatenate, Reshape
from keras.layers import Input
from model.transformer_model import EncoderModel_6
from keras.models import Model
from tensorflow.keras.optimizers import Adam

def unet_blovk(layer_input, filters, kernel_size):
    d = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(layer_input)
    d = BatchNormalization(momentum=0.8)(d)
    d = Activation('elu')(d)

    d = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(d)
    d = BatchNormalization(momentum=0.8)(d)
    d = Activation('elu')(d)

    match_layer = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(layer_input)
    d = Add()([match_layer, d])
    return d


def decoder_block(layer_input1, layer_input2, filters, kernel_size):
    input1 = UpSampling1D(size=4)(layer_input1)
    merge = concatenate([layer_input2, input1], axis=2)
    d = unet_blovk(layer_input=merge, filters=filters, kernel_size=kernel_size)
    return d



def mymodel(input_size=(2048 * 4, 1)):
    inputs = Input(input_size)
    u1 = unet_blovk(layer_input=inputs, filters=32, kernel_size=64)
    pool1 = MaxPooling1D(pool_size=4)(u1)
    u2 = unet_blovk(layer_input=pool1, filters=64, kernel_size=64)
    pool2 = MaxPooling1D(pool_size=4)(u2)
    u3 = unet_blovk(layer_input=pool2, filters=128, kernel_size=32)
    pool3 = MaxPooling1D(pool_size=4)(u3)
    u4 = unet_blovk(layer_input=pool3, filters=256, kernel_size=32)
    pool4 = MaxPooling1D(pool_size=4)(u4)
    u5 = unet_blovk(layer_input=pool4, filters=512, kernel_size=16)

    bridge1 = Conv1D(filters=64, kernel_size=1, strides=1, padding='same')(u5)
    bridge1 = BatchNormalization(momentum=0.8)(bridge1)
    bridge1 = Activation('elu')(bridge1)

    transformer_encoder1 = EncoderModel_6(3, 64, 4, 128, 3840, rate=0.1)
    trans1 = transformer_encoder1(bridge1, True, None)

    trans_merge = concatenate([trans1, u5], axis=2)
    bridge2 = Conv1D(filters=512, kernel_size=16, strides=1, padding='same')(trans_merge)
    bridge2 = BatchNormalization(momentum=0.8)(bridge2)
    bridge2 = Activation('elu')(bridge2)

    u6 = decoder_block(layer_input1=bridge2, layer_input2=u4, filters=128, kernel_size=32)
    u7 = decoder_block(layer_input1=u6, layer_input2=u3, filters=64, kernel_size=32)
    u8 = decoder_block(layer_input1=u7, layer_input2=u2, filters=32, kernel_size=64)
    u9 = decoder_block(layer_input1=u8, layer_input2=u1, filters=16, kernel_size=64)

    conv10 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(u9)
    conv10 = BatchNormalization(momentum=0.8)(conv10)
    conv10 = Activation('elu')(conv10)

    conv11 = Conv1D(filters=16, kernel_size=32, strides=1, padding='same')(conv10)
    conv11 = BatchNormalization(momentum=0.8)(conv11)
    conv11 = Activation('elu')(conv11)

    conv12 = Conv1D(filters=8, kernel_size=16, strides=1, padding='same')(conv11)
    conv12 = BatchNormalization(momentum=0.8)(conv12)
    conv12 = Activation('elu')(conv12)

    conv13 = Conv1D(filters=1, kernel_size=16, strides=1, padding='same')(conv12)
    out = Activation('tanh')(conv13)

    model = Model(inputs=inputs, outputs=out)

    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
    model.summary()

    return model