import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pickle

# Load the data
data = pd.read_csv('final_gaze.csv')

# Select features and labels
feature_columns = ['eye_region_details', 'head_pose', 'head_x', 'head_y', 'head_z', 'iris_2d', 'light_intensity', 'ambient_intensity']
X = data[feature_columns]
y = data['psychological_state']

# Encode labels using one-hot encoding
y_encoded = pd.get_dummies(y)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)  # 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split temp into 50% val, 50% test

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y)

# Convert class weights to a dictionary to pass to model.fit
class_weight_dict = dict(enumerate(class_weights))
class_weight_dict

# Build the neural network model
# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
model.fit(X_train_scaled, y_train, epochs=20, validation_data=(X_val_scaled, y_val), class_weight=class_weight_dict)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save('psychological_state_predictor.h5')