import tensorflow as tf

def get_cnn_standard(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv1D(6, kernel_size=7, activation='sigmoid',padding="valid", input_shape=input_shape))
	model.add(tf.keras.layers.AveragePooling1D(pool_size=3))

	model.add(tf.keras.layers.Conv1D(12, kernel_size=7, activation='sigmoid',padding="valid", input_shape=input_shape))
	model.add(tf.keras.layers.AveragePooling1D(pool_size=3))

	model.add(tf.keras.layers.Flatten())

	model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

	print(input_shape)
	model.summary()

	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

	return model


def simple_dnn(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=input_shape))
	model.add(tf.keras.layers.Dense(units=315, activation='elu'))
	model.add(tf.keras.layers.Dense(units=157, activation='elu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(units=78, activation='elu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=157, activation='elu'))
	model.add(tf.keras.layers.Dense(units=315, activation='elu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def simple_mlp(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=input_shape))
	model.add(tf.keras.layers.Dense(units=315, activation='relu'))
	#model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=120, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def super_simple_mlp(input_shape, n_classes):
	input_layer = tf.keras.layers.Input(input_shape)
	f = tf.keras.layers.Flatten(input_shape=input_shape)(input_layer)
	d1 = tf.keras.layers.Dense(units=315, activation='relu')(f)
	o = tf.keras.layers.Dense(units=n_classes, activation='softmax')(d1)
	model = tf.keras.models.Model(inputs=input_layer, outputs=o)
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def hybrid_restnet(input_shape, n_classes):

	# The IDEA: conv layers -

	n_feature_maps = 64

	input_layer = tf.keras.layers.Input(input_shape)

	# RESICUAL BLOCK
	conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
	conv_x = tf.keras.layers.BatchNormalization()(conv_x)
	conv_x = tf.keras.layers.Activation('relu')(conv_x)

	conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
	conv_y = tf.keras.layers.BatchNormalization()(conv_y)
	conv_y = tf.keras.layers.Activation('relu')(conv_y)

	conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
	conv_z = tf.keras.layers.BatchNormalization()(conv_z)

	return model
