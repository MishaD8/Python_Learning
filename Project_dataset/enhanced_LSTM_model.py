import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam   # type: ignore

def load_and_preprocess_data(filepath, add_features=True):
    """
    Load and preprocess 5 years of lottery data.

    Args:
        filepath: Path to the CSV file with lottery data add features: Whether to add 
        additional time-based features

    Returns:
        Processed dataframe and numeric data for modeling
    """
    # Load data
    data = pd.read_csv(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_dataset\dataset.csv')

    # Convert date column to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['data'])

    # Sort by date (ascending order)
    if 'date' in data.columns:
        data = data.sort_values('date').reset_index(drop=True)

    # Add time-based features if requested
    if add_features and 'date' in data.columns:
        # Add day of week (0=Monday, 6=Sunday)
        data['day_of_week'] = data['date'].dt.dayofweek

        # Add day of year (1-366)
        data['day_of_year'] = data['date'].dt.dayofyear

        # Add month (1-12)
        data['month'] = data['date'].dt.month

        # Add quarter (1-4)
        data['quarter'] = data['date'].dt.quarter

        # Add year
        data['year'] = data['date'].dt.year

        # Add week of year
        data['week_of_year'] = data['date'].dt.isocalendar().week

        # Convert categorical features to one-hot encoding
        data = pd.get_dummies(data, columns=['day_of_week', 'month', 'quarter'], drop_first=False)
    
    # Select numeric columns for the model (exclude date column)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'date' in numeric_cols:
        numeric_cols.remove('date')

    numeric_data = data[numeric_cols]

    return data, numeric_data

def prepare_sequences(data, window_size=10):
    """
    Create sequences for time series prediction with a larger window size.

    Args:
        data: Preprocessed numeric data
        window_size: Number of previous draws to use for prediction

    Returns:
        x: Input sequences
        y: Target values (next draw numbers)
    """
    x, y = [], []

    # Get columns that represent lottery numbers
    lottery_cols = [col for col in data.columns if col.startswith('num')]

    # Create sliding window sequences
    for i in range(len(data) - window_size):
        # Input sequence (window_size previous draws with all features)
        x.append(data.iloc[i:i+window_size].values)

        # Target (only the lottery numbers from the next draw)
        if lottery_cols:
            y.append(data.iloc[i+window_size][lottery_cols].values)
        else:
            # If no columns start with 'num', use all columns (less ideal)
            y.append(data.iloc[i+window_size].values)
    return np.array(x), np.array(y)

def build_enhances_model(input_shape, output_shape, learning_rate=0.001):
    """
    Build an enhanced LSTM model suitable for 5 years of lottery data.

    Args:
        input_shape: Shape of input sequences (window_size, features)
        output_shape: Number of output values (typically 6 or 7 for lottery)
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First bidirectional LSTM layer with more units
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        # Second bidirectional LSTM layer
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),

        # Third LSTM layer
        LSTM(128),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(output_shape)
    ])

    # Use Adam optimizer with custom learning rate
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model

def train_with_validation(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=300):
    """
    Train the model with validation and advanced callbacks.

    Args:
        model: Compiled model
        x_train, y_train: Training data
        x_val, y_val: Validation data
        batch_size: Batch size for training
        epochs: Maximum number of epochs

    Returns:
        Training history and trained model
    """
    # Callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),

        # Model checkpoint to save best model
        ModelCheckpoint(
            'best_lottery_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # Reduce learning rate when plateau is reached 
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history, model

def visialize_training(history):
    """
    Create detailed visualizations of training progress.

    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(15, 10))

    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Plot training & validation MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history,history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.yscale('log')

    plt.tight_layout()
    plt.show()

