from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, GlobalAveragePooling2D, Dense, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def dense_block(x, num_layers, growth_rate, name):
    for i in range(num_layers):
        cb = conv_block(x, growth_rate, name=name + '_layer_' + str(i + 1))
        x = Concatenate(name=name + '_concat_' + str(i + 1))([x, cb])
    return x

def conv_block(x, growth_rate, name):
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(4 * growth_rate, kernel_size=(1, 1), use_bias=False, name=name + '_conv1')(x)
    x = BatchNormalization(name=name + '_bn2')(x)
    x = Activation('relu', name=name + '_relu2')(x)
    x = Conv2D(growth_rate, kernel_size=(3, 3), padding='same', use_bias=False, name=name + '_conv2')(x)
    return x

def transition_layer(x, reduction, name):
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(x.shape[-1] * reduction), kernel_size=(1, 1), use_bias=False, name=name + '_conv')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name=name + '_pool')(x)
    return x

def DenseNet(input_shape, num_classes, growth_rate=32, num_layers_per_block=[6, 12, 24, 16], reduction=0.5):
    inputs = Input(shape=input_shape)
    x = Conv2D(2 * growth_rate, kernel_size=(7, 7), strides=(2, 2), padding='same', use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu', name='relu_conv1')(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    # Dense Blocks
    for i, num_layers in enumerate(num_layers_per_block):
        x = dense_block(x, num_layers=num_layers, growth_rate=growth_rate, name='dense_block' + str(i + 1))
        if i != len(num_layers_per_block) - 1:  # No transition after the last dense block
            x = transition_layer(x, reduction=reduction, name='transition' + str(i + 1))

    # Final Layers
    x = BatchNormalization(name='bn_final')(x)
    x = Activation('relu', name='relu_final')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='fc')(x)

    model = Model(inputs, outputs, name='DenseNet_MRI')
    return model

input_shape = (256, 256, 1)
num_classes = 2
model = DenseNet(input_shape=input_shape, num_classes=num_classes)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
