import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras.losses import MSE as mean_squared_error 
from keras.metrics import MAE as mean_absolute_error 



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
    if 'data' in data.columns:
        data['date'] = pd.to_datetime(data['data'])
    elif 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

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

def visualize_training(history):
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

def evaluate_predictions(model, x_test, y_test, scaler=None):
    """
    Evaluate model predictions against actual lottery numbers.

    Args:
        model: Trained model 
        x_test: Test input sequences
        y_test: Actual lottery numbers
        scaler: Scaler used for normalization (if any)

    Returns: 
        DataFrame comparing predicted vs actual numbers
    """

    # Make predictions
    y_pred = model.predict(x_test)

    # Inverse transform if scaler was used
    if scaler is not None:
        # Create dummy array with same shape as the original data
        dummy = np.zeros((y_pred.shape[0], scaler.n_features_in_))
        # Place the predictions in the correct columns
        dummy[:, :y_pred.shape[1]] = y_pred
        # Inverse transform
        y_pred = scaler.inverse_transform(dummy)[:, :y_test.shape[1]]

        # Do the same for actual values
        dummy = np.zeros((y_test.shape[0], scaler.n_features_in_))
        dummy[:, :y_test.shape[1]] = y_test
        y_test = scaler.inverse_transform(dummy)[:, :y_test.shape[1]]

    # Round predictions for lottery numbers
    y_pred_rounded = np.round(y_pred).astype(int)

    # Create DataFrame for comparison
    results = pd.DataFrame()
    for i in range(y_test.shape[1]):
        results[f'Actual_{i+1}'] = y_test[:, i]
        results[f'Predicted_{i+1}'] = y_pred_rounded[:, i]
        results[f'Difference_{i+1}'] = np.abs(results[f'Actual_{i+1}'] - results[f'Predicted_{i+1}'])


    # Add overall error metrics
    results['Mean_Difference'] = results[[f'Difference_{i+1}' for i in range(y_test.shape[1])]].mean(axis=1)

    return results

def predict_next_draw(model, data, window_size, scaler=None):
    """
    Predict the next lottery draw.

    Args:
        model: Trained model
        data: Full preprocessed dataset
        window_size: Window size used for model training
        scaler: Scaler used for normalization (if any)

    Returns:
        Predicted lottery numbers
    """
    # Get the most recent window of data
    last_window = data.iloc[-window_size:].values

    # Reshape for model input (adding batch dimension)
    x_pred = last_window.reshape(1, window_size, data.shape[1])

    # Generate prediction
    prediction = model.predict(x_pred)

    # Flatten the prediction if it's a multi-dimensional array
    prediction = prediction.flatten()

    # Inverse transform if scaler was used
    if scaler is not None:
        # Create dummy array with same shape as the the original data
        dummy = np.zeros((1, scaler.n_features_in_))
        # Place the prediction in the correct columns
        dummy[0, :len(prediction)] = prediction
        # Inverse transform
        prediction = scaler.inverse_transform(dummy)[0, :len(prediction)]

    # Round to integers
    rounded_prediction = np.round(prediction).astype(int)

    # Make sure predictions are within valid range (e.g., 1-49 for many lotteries)
    # Adjust min_val and max_val based on your specific lottery
    min_val, max_val = 1, 49
    rounded_prediction = np.clip(rounded_prediction, min_val, max_val)

    # Convert to a list of integers
    numbers_list = rounded_prediction.tolist()

    # Ensure all predicted numbers are unique 
    # Convert NumPy arrays to Python integers for set operations
    
    if len(numbers_list) != len(set(numbers_list)):
        # If duplicates exist, replace them with new numbers
        unique_nums = set(numbers_list)
        all_possible = set(range(min_val, max_val +1))
        remaining = list(all_possible - unique_nums)
        np.random.seed(int(pd.Timestamp.now().timestamp()))
        np.random.shuffle(remaining)

        # Replace duplicates
        unique_list = []
        for num in numbers_list:
            if num not in unique_list:
                unique_list.append(num)
            else:
                unique_list.append(remaining.pop())

        rounded_prediction = np.array(unique_list)

    return rounded_prediction

def main():
    """
    Main function to run the full lottery prediction process.
    """

    # 1. Load and preprocess 5 years of lottery data
    try:
        print("Loading and preprocessing data...")
        full_data, numeric_data = load_and_preprocess_data(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_dataset\dataset.csv')
        print(f"Loaded data with {len(full_data)} draws and {numeric_data.shape[1]} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)

    # 3. Create sequences
    print("Creating sequences...")
    window_size = 10 # Using 10 previous draws for prediction
    x, y = prepare_sequences(scaled_df, window_size)
    print(f"Created {len(x)} sequences with shape {x.shape}")

    # 4. Split data (time series - keep chronological order)
    # Use earlier 70% for training, middle 15% for validation, last 15% for testing
    train_size = int(0.7 * len(x))
    val_size = int(0.15 * len(x))

    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
    x_test, y_test = x[train_size+val_size:], y[train_size+val_size:]

    print(f"Training set: {x_train.shape}")
    print(f"Validation set: {x_val.shape}")
    print(f"Test set: {x_test.shape}")

    # 5. Build enhanced model
    print("Building model...")
    input_shape = (x_train.shape[1], x_train.shape[2])
    output_shape = y_train.shape[1]
    model = build_enhances_model(input_shape, output_shape)
    model.summary()

    # 6. Train model
    print("Training model...")
    history, trained_model = train_with_validation(
        model, x_train, y_train, x_val, y_val, batch_size=32, epochs=300
    )

    # 7. Visualize training
    print("Visualizing training progress...")
    visualize_training(history)  

    #8. Evaluate on test set
    print("Evaluating model...")
    results = evaluate_predictions(trained_model, x_test, y_test)
    print("\nSample of prediction results:")
    print(results.head())

    # Calculate and print average error
    mae = results['Mean_Difference'].mean()
    print(f"\nAverage prediction error: {mae:.2f}")

    # 9. Predict next draw
    print("\nPredicting next lottery draw...")
    next_draw = predict_next_draw(trained_model, scaled_df, window_size)
    print(f"Predicted numbers for next draw: {next_draw}")

    # 10. Optional: Save the model for future use
    trained_model.save('lottery_prediction_model.h5')
    print("Model saved as 'lottery_prediction_model.h5'")

# def load_model_with_custom_objects():
#     try:
#         # Define custom objects
#         custom_objects = {
#             'loss': 'mse', # Use string identifier
#             'metrics': ['mae'] # Use string identifier
#         }

#         # Load model with custom objects
#         model = load_model('lottery_prediction_model.h5', compile=False)

#         # Recompile the model
#         optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#         model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

#         print("Successfully loaded existing model")
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

def update_existing_model():
    """
    Update existing model with new lottery drawings by rebuilding the model and loading weights
    """

    # First, load and preprocess the data

    print("Loading and preprocessing updated data...")
    full_data, numeric_data = load_and_preprocess_data(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_dataset\dataset.csv')
    print(f"Loaded data with {len(full_data)} draws and {numeric_data.shape[1]} features")

    # Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)

    # Create sequences
    print("Creating sequences...")
    window_size = 10
    x, y = prepare_sequences(scaled_df, window_size)

    # Recreate the model with the same architecture
    print("Rebuilding model architecture...")
    input_shape = (window_size, scaled_df.shape[1])
    output_shape = y.shape[1]
    model = build_enhances_model(input_shape, output_shape)

    try:
        # Try to load the weights
        # First, try loading the full model (might work in some versions)
        try:
            loaded_model = load_model('lottery_prediction.h5', compile=False)
            model.set_weights(loaded_model.get_weights())  
            print("Successfully loaded model weights")
        except:
            # If that fails, try loading just the weights
            model.load_weights('lottery_prediction_model.h5')
            print("Successfully loaded weights from model file")
    except Exception as e:
        print(f"Warning: Could not load weights, using new model: {e}")
        # Continue with a fresh model if loading fails

    # Use a different random seed
    np.random.seed(int(pd.Timestamp.now().timestamp()))
    tf.random.set_seed(int(pd.Timestamp.now().timestamp()))

    #Split data
    train_size = int(0.85 * len(x))
    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:], y[train_size:]

    # Train/fine-tune the model
    print("Training model with updated data...")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # Save the updated model
    model.save('updated_lottery_model.h5')
    print("Updated model saved as 'updated_lottery_model.h5'")

    # Predict next draw
    print("Predicting next lottery draw with updated model...")
    next_draw = predict_next_draw(model, scaled_df, window_size, scaler)
    print(f"Predicted numbers for next draw: {next_draw}")

if __name__ == "__main__":
    # Set seeds for reproducibility - but use different seeds when updating
    np.random.seed(42)
    tf.random.set_seed(42)

    # Uncomment the option you want to run

    # Option 1: Run full training from scratch
    # main()

    # Option 2: Update existing model with new data
    update_existing_model()
