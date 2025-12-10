from keras import layers, Model, initializers

#
# Helper for conv construction
#
def conv_block(
        model_in,
        kernel_size,
        pooling,
        conv_filters,
    ):

    conv1 = layers.Conv2D(filters= conv_filters , activation='tanh', kernel_size=kernel_size, kernel_initializer='he_normal')(model_in)

    if pooling:
        pooling = layers.MaxPool2D(pool_size=(3,3))(conv1)
        return pooling
    else:
        return conv1

# Downscale the initialisation of the glorot normal for the output later for performance improvement (Andrychowicz et al., 2020)
# https://datascience.stackexchange.com/questions/19019/custom-weight-initialization-in-keras
SCALE = 0.01
class ReducedGlorot(initializers.GlorotNormal):
    def __call__(self, shape, dtype = None, **kwargs):
        glorot = initializers.GlorotUniform()
        return glorot(shape,dtype) * SCALE # type: ignore

    
def define_model(
        dimensions, 
        regression,
        total_moves , 
        conv,
        conv_filters,
        dense_units,

    ):
    # CNN model
    model_in = layers.Input(dimensions)

    if conv:
        conv1 = conv_block(model_in , kernel_size=(3,3), pooling=False, conv_filters=conv_filters)
        conv2 = conv_block(conv1, kernel_size=(3,3), pooling=True, conv_filters=conv_filters)
        conv3 = conv_block(conv2, kernel_size=(3,3), pooling=False, conv_filters=conv_filters)
        dense_in = layers.Flatten()(conv3)
    else:
        dense_in = model_in

    dense_1 = layers.Dense(dense_units, activation='tanh', kernel_initializer='glorot_uniform')(dense_in)
    dense_2 = layers.Dense(dense_units, activation='tanh',kernel_initializer='glorot_uniform')(dense_1)

    if regression:
        output = layers.Dense(1 , activation='linear')(dense_2)

    else: 
        output = layers.Dense(total_moves, activation='softmax', kernel_initializer= ReducedGlorot())(dense_2) # type: ignore


    model = Model(model_in, output)

    return model