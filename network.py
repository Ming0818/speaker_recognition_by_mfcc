from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard

def train(x, y, epoch=20):
    checkpoint_filename = 'checkpoints/params-at-{epoch:02d}-{acc:.2f}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filename, save_best_only=True, monitor='val_loss')
    tensorboard = TensorBoard(log_dir='./logs', write_grads=True)

    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=(20, 700, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,3))
    model.add(Dropout(rate=.2))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64,3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(rate=.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x, y, batch_size=32, epochs=epoch, verbose=2, validation_split=0.1, callbacks=[model_checkpoint, tensorboard])

    json_string = model.to_json()
    with open("./model.json", 'w') as json_file:
        json_file.write(json_string)
    model.save_weights("./params.hdf5")
