import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = np.load('data.npy')
labels = np.load('labels.npy')

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(data.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training Started \n")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

model.save('sign_language_model.h5')
print("Model saved as 'sign_language_model.h5'\n")