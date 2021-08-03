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