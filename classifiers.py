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

def simple_mlp(input_shape, n_classes):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=input_shape))
	model.add(tf.keras.layers.Dense(units=315, activation='tanh'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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

# this sucks
def rest_net(input_shape, n_classes):
	n_feature_maps = 3

	input_layer = tf.keras.layers.Input(input_shape)

	# BLOCK 1

	conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
	conv_x = tf.keras.layers.BatchNormalization()(conv_x)
	conv_x = tf.keras.layers.Activation('relu')(conv_x)

	conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
	conv_y = tf.keras.layers.BatchNormalization()(conv_y)
	conv_y = tf.keras.layers.Activation('relu')(conv_y)

	conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
	conv_z = tf.keras.layers.BatchNormalization()(conv_z)

	# expand channels for the sum
	shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
	shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

	output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
	output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

	# BLOCK 2

	conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
	conv_x = tf.keras.layers.BatchNormalization()(conv_x)
	conv_x = tf.keras.layers.Activation('relu')(conv_x)

	conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
	conv_y = tf.keras.layers.BatchNormalization()(conv_y)
	conv_y = tf.keras.layers.Activation('relu')(conv_y)

	conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
	conv_z = tf.keras.layers.BatchNormalization()(conv_z)

	# expand channels for the sum
	shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
	shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

	output_block_2 = tf.keras.layers.add([shortcut_y, conv_z])
	output_block_2 = tf.keras.layers.Activation('relu')(output_block_2)

	# BLOCK 3

	conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
	conv_x = tf.keras.layers.BatchNormalization()(conv_x)
	conv_x = tf.keras.layers.Activation('relu')(conv_x)

	conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
	conv_y = tf.keras.layers.BatchNormalization()(conv_y)
	conv_y = tf.keras.layers.Activation('relu')(conv_y)

	conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
	conv_z = tf.keras.layers.BatchNormalization()(conv_z)

	# no need to expand channels because they are equal
	shortcut_y = tf.keras.layers.BatchNormalization()(output_block_2)

	output_block_3 = tf.keras.layers.add([shortcut_y, conv_z])
	output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)

	# FINAL

	gap_layer = tf.keras.layers.GlobalAveragePooling1D()(output_block_3)

	output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(gap_layer)

	model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

	model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
				  metrics=['accuracy'])

	return model
