import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load data
df = pd.read_csv('data.csv')

# Add noise to the data
noise_factor = 0.1
df_noisy = df + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=df.shape)

# Define autoencoder architecture
input_shape = (df.shape[1],)
hidden_size = 64
latent_size = 32

input_layer = Input(shape=input_shape)
hidden_layer = Dense(hidden_size, activation='relu')(input_layer)
latent_layer = Dense(latent_size, activation='relu')(hidden_layer)
hidden_layer_2 = Dense(hidden_size, activation='relu')(latent_layer)
output_layer = Dense(input_shape[0], activation='sigmoid')(hidden_layer_2)

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder model
epochs = 100
batch_size = 32
autoencoder.fit(df_noisy, df, epochs=epochs, batch_size=batch_size)

# Use the autoencoder to denoise the data
df_denoised = autoencoder.predict(df_noisy)

# Define neural network architecture
nn_input_shape = (df_denoised.shape[1],)
nn_hidden_size = 64
nn_output_size = 1

nn_input_layer = Input(shape=nn_input_shape)
nn_hidden_layer = Dense(nn_hidden_size, activation='relu')(nn_input_layer)
nn_output_layer = Dense(nn_output_size, activation='linear')(nn_hidden_layer)

# Build the neural network model
nn_model = Model(inputs=nn_input_layer, outputs=nn_output_layer)

# Compile the neural network model
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model
nn_epochs = 100
nn_batch_size = 32
nn_model.fit(df_denoised, chemical_concentrations, epochs=nn_epochs, batch_size=nn_batch_size)

# Assume X_test is a preprocessed test set of infrared spectra
# with shape (num_samples, num_features)
y_pred = nn_model.predict(X_test)

# y_pred is an array of predicted elemental concentrations
# with shape (num_samples, num_outputs)

# Create evaluation functions
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# MSE, R² and STD
mse = mean_squared_error(y_true, y_pred) 
r2 = r2_score(y_true, y_pred) 
std_true = np.std(y_true, axis=0)
# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# or rmse = np.sqrt(mse)
# RPD
rpd = std_true / rmse
# Adjusted R²
r2_adj = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
# RPIQ
q75, q25 = np.percentile(y_true, [75, 25], axis=0)
iqr = q75 - q25
rpiq = iqr / rmse

