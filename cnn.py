import numpy as np
import tensorflow as tf
import random
import glob
import matplotlib.pyplot as plt

seed = 12345
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

#%cd /content/drive/My\ Drive/Didattica/ML\ 2020-2021\ Deep\ Learning/Esercitazione\ CNN/

n_classes = 10

train = np.loadtxt("mnist_rot/mnist_rot_train.amat", dtype='float32')
test = np.loadtxt("mnist_rot/mnist_rot_test.amat", dtype='float32')
print(train.shape)
print(test.shape)

x_tr = train[:, :-1]
y_tr = train[:, -1]
x_ts = test[:, :-1]
y_ts = test[:, -1]

y_tr = tf.keras.utils.to_categorical(y_tr, num_classes=n_classes)
y_ts = tf.keras.utils.to_categorical(y_ts, num_classes=n_classes)

if tf.keras.backend.image_data_format() == 'channels_first':
    x_tr = x_tr.reshape(x_tr.shape[0], 1, 28, -1)
    x_ts = x_ts.reshape(x_ts.shape[0], 1, 28, -1)
else: # if tf.keras.backend.image_data_format() == 'channels_last'
    x_tr = x_tr.reshape(x_tr.shape[0], 28, -1, 1)
    x_ts = x_ts.reshape(x_ts.shape[0], 28, -1, 1)

print(x_tr.shape)
print(x_ts.shape)

input_shape = (x_tr.shape[1],x_tr.shape[2],x_tr.shape[3])
print(input_shape)

num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num_row*num_col):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(np.squeeze(x_tr[i]), cmap='gray')
    ax.set_title('Label: {}'.format(np.where(y_tr[i]==1)[0][0]))
plt.tight_layout()
plt.show()

val_dim = int(.3*len(x_tr))
arr_mlp = np.arange(len(x_tr))
index_val_mlp = np.random.choice(arr_mlp, val_dim, replace=False)
index_train_mlp = np.setdiff1d(arr_mlp, index_val_mlp)
x_val = x_tr[index_val_mlp,]
y_val = y_tr[index_val_mlp,]
x_tr = x_tr[index_train_mlp,]
y_tr = y_tr[index_train_mlp,]

print(x_val.shape)
print(y_val.shape)
print(x_tr.shape)
print(y_tr.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128, kernel_size=(6,6), activation='relu',padding="same", input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu',padding="same"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',padding="same"))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',padding="same"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu',padding="same"))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu',padding="same"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding="same"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(.45))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(.25))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

print(input_shape)
model.summary()

opt = tf.keras.optimizers.Adam()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
history=model.fit(x_tr, y_tr, batch_size=64, epochs=200, validation_data=(x_val, y_val), callbacks=[callback])
  
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

performance = model.evaluate(x_val, y_val, verbose=0)
print("Validation performance")
print(performance)

performance = model.evaluate(x_ts, y_ts, verbose=0)
print("Test performance")
print(performance)
