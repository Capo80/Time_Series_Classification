import tensorflow as tf


# TODO tune starting momentum & learning ratewith GridSearchCV


# this fucking sucks
def get_cnn_standard(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv1D(6, kernel_size=105, activation='relu',padding="same", input_shape=input_shape))
	model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Conv1D(6, kernel_size=7, activation='relu',padding="same"))
	model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

	return model


def shallow_cnn(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv1D(6, kernel_size=105, activation='relu',padding="same", input_shape=input_shape))
	model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
	model.add(tf.keras.layers.BatchNormalization())

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

	return model


def shallow_cnn2(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv1D(6, kernel_size=105, activation='relu',padding="same", input_shape=input_shape))
	model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
	model.add(tf.keras.layers.BatchNormalization())
	# (105, 6)
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(units=105, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

	return model


def simple_dnn(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=input_shape))
	model.add(tf.keras.layers.Dense(units=315, activation='relu'))
	model.add(tf.keras.layers.Dense(units=105, activation='relu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(units=50, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=105, activation='relu'))
	model.add(tf.keras.layers.Dense(units=315, activation='relu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# the most promising
def simple_mlp(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=input_shape))
	model.add(tf.keras.layers.Dense(units=315, activation='relu'))
	model.add(tf.keras.layers.Dense(units=105, activation='relu'))
	model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def super_simple_mlp(input_shape, n_classes):
	input_layer = tf.keras.layers.Input(input_shape)
	f = tf.keras.layers.Flatten(input_shape=input_shape)(input_layer)
	d1 = tf.keras.layers.Dense(units=315*3, activation='relu')(f)
	o = tf.keras.layers.Dense(units=n_classes, activation='softmax')(d1)
	model = tf.keras.models.Model(inputs=input_layer, outputs=o)
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# this sucks too
def hybrid_restnet(input_shape, n_classes):

	input_shape = input_shape[:]

	# The IDEA: conv layers + residual block
	input_layer = tf.keras.layers.Input(input_shape)

	# RESIDUAL BLOCK
	rb = tf.keras.layers.Conv1D(filters=3, kernel_size=8, padding='same', activation='relu')(input_layer)
	rb = tf.keras.layers.BatchNormalization()(rb)
	rb = tf.keras.layers.Activation('relu')(rb)
	rb = tf.keras.layers.Conv1D(filters=3, kernel_size=3, padding='same', activation='relu')(rb)
	rb = tf.keras.layers.BatchNormalization()(rb)
	rb = tf.keras.layers.Activation('relu')(rb)
	#model.add(tf.keras.layers.Dropout(0.2))

	# expand channels for the sum
	shortcut_y = tf.keras.layers.Conv1D(filters=3, kernel_size=1, padding='same', activation='relu')(input_layer)
	shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

	# shortcut
	output_block_1 = tf.keras.layers.add([shortcut_y, rb])
	output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

	# 27 - 32 - 22

	gap = tf.keras.layers.GlobalAveragePooling1D()(output_block_1)

	"""
	c = tf.keras.layers.Conv1D(3, kernel_size=7, activation='relu',padding="valid")(gap)
	c = tf.keras.layers.AveragePooling1D(pool_size=3)(c)
	c = tf.keras.layers.Conv1D(3, kernel_size=7, activation='relu',padding="valid")(c)
	c = tf.keras.layers.AveragePooling1D(pool_size=3)(c)
	c = tf.keras.layers.BatchNormalization()(c)
	"""

	out = tf.keras.layers.Dense(n_classes, activation='softmax')(gap)

	opt = tf.keras.optimizers.Adam()
	model = tf.keras.models.Model(inputs=input_layer, outputs=out)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

	return model
