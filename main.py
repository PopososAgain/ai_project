import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import wandb


def plot_regression_line(model, X, y_true):

    X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    y_pred = model.predict(X_pred).flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y_true, label='True Data', marker='o', edgecolors='k')
    plt.plot(X_pred, y_pred, label='Predicted Line', color='r')
    plt.title('Regression Predictions as a Line')
    plt.show()


wandb.init(project="ai_lab_project_test", entity="popososagainpl")
config = wandb.config
config.epochs = 500
config.batch_size = 15

# Step 1: Generate Sample Data for Regression
X, y = make_regression(n_samples=1000, n_features=1, noise=40, random_state=42)

# Step 2: Dataset Preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Neural Network Model Construction for Regression
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')  # For regression
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Step 4: Model Training
history = model.fit(X_train, y_train, epochs=100, batch_size=15, validation_split=0.1)

# Plot making
plot_regression_line(model, X_test, y_test)

for epoch in range(config.epochs):
    wandb.log({"Epoch": epoch, "Training Loss": history.history['loss'][epoch],
               "Validation Loss": history.history['val_loss'][epoch]})
wandb.finish()

# User input
while True:
    user_input = input("Enter a value for prediction (type 'exit' to stop): ")

    if user_input.lower() == 'exit':
        break

    try:
        user_input = float(user_input)
        user_input = np.array([[user_input]])
        prediction = model.predict(user_input)

        print(f"Predicted value for input {user_input[0][0]}: {prediction[0][0]}")
    except ValueError:
        print("Invalid input. Please enter a numerical value or 'exit'.")
