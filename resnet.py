import numpy as np
import keras
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.models import Model


class ResNet(object):
	''' 
		Creates a residual neural network as described in the AlphaGo Zero paper.

		Output: a tuple [v, p] of value and prior prob. over the action space.
	'''
	def __init__(self, input_shape, n_filter=256, kernel_size=(3, 3), n_blocks=20, bn_axis=3):
		self.input_shape = input_shape
		self.n_filter = n_filter #number of filters in convolutional layers
		self.kernel_size = kernel_size #kernel size of convolutional layers
		self.n_blocks = n_blocks #number of residual blocks
		self.bn_axis = bn_axis #batch normalization axis
		self.model = self.build_model()

	def build_model(self):
		input_ = Input(shape=self.input_shape)

		#Input layers
		x = Conv2D(self.n_filter, self.kernel_size, strides=(1, 1), padding='same')(input_)
		x = BatchNormalization(axis=self.bn_axis)(x)
		x = Activation('relu')(x)

		#residual tower
		for _ in range(self.n_blocks):
			cnn_1 = Conv2D(self.n_filter, self.kernel_size, padding='same')
			cnn_2 = Conv2D(self.n_filter, self.kernel_size, padding='same')
			bn_1 = BatchNormalization(axis=self.bn_axis)
			bn_2 = BatchNormalization(axis=self.bn_axis)
			relu = Activation('relu')
			y = bn_2(cnn_2(relu(bn_1(cnn_1(x)))))
			x = relu(layers.add([x, y]))

		#Policy part
		y = Activation('relu')(BatchNormalization(axis=self.bn_axis)(Conv2D(2, (1, 1), padding='same')(x)))
		y = Flatten()(y)
		y = Dense(83, activation="softmax", name="policy_output", kernel_initializer='random_uniform',
	                bias_initializer='ones')(y)

		#Value part
		x = Activation('relu')(BatchNormalization(axis=self.bn_axis)(Conv2D(1, (1, 1), padding='same')(x)))
		x = Flatten()(x)
		x = Dense(256, activation='relu', kernel_initializer='random_uniform',
	                bias_initializer='ones')(x)
		x = Dense(1, kernel_initializer='random_uniform',
	                bias_initializer='ones')(x)
		x = Activation('tanh', name="value_output")(x)

		model = Model(input_, [x, y])
		return model

	def compile(self):
		self.model.compile(loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mse'},
              loss_weights={'policy_output': 1., 'value_output': 1.}, optimizer='adam')


if __name__ == '__main__':
	resnet = ResNet((19,19, 17),  n_blocks=20)
	resnet.model.summary()
	resnet.compile()	