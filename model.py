from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import callbacks

def initialize_model():
    model = Sequential()

    model.add(layers.Input(shape=(141, 216, 1)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu", kernel_regularizer=l2(0.00001)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(256, activation="relu", kernel_regularizer=l2(0.00001)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128, activation="relu", kernel_regularizer=l2(0.00001)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(10, activation="softmax"))

    return model

def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = initialize_model()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0)

    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, lr_reducer])

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    return model, history
