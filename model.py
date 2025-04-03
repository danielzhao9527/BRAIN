from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense, Permute, Concatenate, Conv2D, Add, Activation, Lambda, Input, \
    BatchNormalization, LayerNormalization, Flatten, DepthwiseConv2D, AveragePooling2D, Dropout, Average, \
    Bidirectional, GRU, Conv1D, MultiHeadAttention
from tensorflow.keras.models import Model
from keras.regularizers import L1, L2
import tensorflow as tf


def mha_block(input_feature, key_dim=8, num_heads=2, dropout=0.5):
    # layer normalization
    x = LayerNormalization(epsilon=1e-6)(input_feature)
    x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(0.3)(x)
    # skip connection
    mha_feature = Add()([input_feature, x])

    return mha_feature


def conv_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(
        input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout,
               weightDecay=0.009, activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1,
                      kernel_regularizer=L2(weightDecay),
                      padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       kernel_regularizer=L2(weightDecay),
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       kernel_regularizer=L2(weightDecay),
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out


def DDF(n_classes, Chans=22, Samples=1125):
    input = Input(shape=(1, Chans, Samples))
    input0 = Permute((3, 2, 1))(input)

    block1_1 = conv_block(input_layer=input0, F1=16, kernLength=16, dropout=0.3, poolSize=7, in_chans=Chans)
    block1_2 = conv_block(input_layer=input0, F1=16, kernLength=32, dropout=0.3, poolSize=7, in_chans=Chans)
    block1 = Concatenate()([block1_1, block1_2])
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)
    input2 = mha_block(block1)

    decay = 0.01
    block2 = TCN_block(input_layer=input2, input_dimension=64, depth=2,
                               kernel_size=4, filters=64, weightDecay=decay,
                               dropout=0.3, activation='elu')
    block2 = Lambda(lambda x: x[:, -1, :])(block2)
    block2 = Dense(n_classes, kernel_regularizer=L2(0.5))(block2)

    block3 = Bidirectional(GRU(units=64, kernel_regularizer=L2(1e-7), return_sequences=True))(input2)
    block3 = Conv1D(filters=64, kernel_size=1, kernel_regularizer=L2(0.8), activation='elu', padding='same')(block3)
    block3 = Add()([block3, input2])
    block3 = tf.expand_dims(block3, axis=-1)
    block3 = MultiHeadAttention(key_dim=8, num_heads=2, dropout=0.5)(block3, block3)
    block3 = Dropout(rate=0.5)(block3)
    block3 = Flatten()(block3)
    block3 = Dense(n_classes, kernel_regularizer=L2(0.5))(block3)

    block4 = Average()([block2, block3])
    block4 = Dense(n_classes, kernel_constraint=max_norm(.25))(block4)
    softmax = Activation('softmax', name='softmax')(block4)
    return Model(inputs=input, outputs=softmax)
