from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Activation, Input, Flatten


class DQN(Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.input_dim)
        x = Flatten()(input_layer)

        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.output_dim, activation='relu')(x)

        return Model(inputs=[input_layer], outputs=[x])

    def call(self, inputs):
        x = self.model.call(inputs)
        return x


class ConvDQN(Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.input_dim)
        x = Conv2D(filters=32, kernel_size=8, strides=4,
                   activation='relu')(input_layer)
        x = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(x)

        x = Flatten()(x)

        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.output_dim, activation='relu')(x)

        return Model(inputs=[input_layer], outputs=[x])

    def call(self, inputs):
        x = self.model.call(inputs)
        return x
