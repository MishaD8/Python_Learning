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
    # filepath = "G:\Мой диск\cybersecurity\Python for cybersecurity\Project_dataset\dataset.csv"
    data = pd.read_csv(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_lottery\dataset.csv')
    
    

    lottery_cols = [col for col in data.columns if col.startswith('num')]
    if len(lottery_cols) > 6:
        bonus_cols = lottery_cols[6:] # These are the columns to exclude
        data = data.drop(columns=bonus_cols)
        print(f"Excluded bonus number column(s): {bonus_cols}")

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

    # Get only the first 6 lottery number columns
    lottery_cols = [col for col in data.columns if col.startswith('num')]
    if len(lottery_cols) > 6:
        lottery_cols = lottery_cols[:6]

    # Create sliding window sequences
    for i in range(len(data) - window_size): 
        # Input sequence (window_size previous draws with all features)
        x.append(data.iloc[i:i+window_size].values)

        # Target (only the 6 lottery numbers from the next draw)
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
        plt.plot(history.history['lr'])
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
        Predicted lottery numbers (only the first 6)
    """
    # Get the most recent window of data
    last_window = data.iloc[-window_size:].values

    # Reshape for model input (adding batch dimension)
    x_pred = last_window.reshape(1, window_size, data.shape[1])
    
    # Generate prediction
    prediction = model.predict(x_pred)

    # Flatten the prediction if it's a multi-dimensional array
    prediction = prediction.flatten()

    # Ensure we only take the first 6 predictions if there are more
    if len(prediction) > 6:
        prediction = prediction[:6]

    # # Get the most recent window of data
    # last_window = data.iloc[-window_size:].values

    # # Reshape for model input (adding batch dimension)
    # x_pred = last_window.reshape(1, window_size, data.shape[1])

    # # Generate prediction
    # prediction = model.predict(x_pred)

    # # Flatten the prediction if it's a multi-dimensional array
    # prediction = prediction.flatten()

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

    return rounded_prediction[:6]

def main():
    """
    Main function to run the full lottery prediction process.
    """

    # 1. Load and preprocess 5 years of lottery data
    try:
        print("Loading and preprocessing data...")
        full_data, numeric_data = load_and_preprocess_data(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_lottery\dataset.csv')
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
    full_data, numeric_data = load_and_preprocess_data(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_lottery\dataset.csv')
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
    main()

    # Option 2: Update existing model with new data
    # update_existing_model()



# # Fix for visualize_training function
# def visualize_training(history):
#     """
#     Create detailed visualizations of training progress.
    
#     Args:
#         history: Training history from model.fit()
#     """
#     plt.figure(figsize=(15, 10))

#     # Plot training & validation loss
#     plt.subplot(2, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(loc='upper right')

#     # Plot training & validation MAE
#     plt.subplot(2, 2, 2)
#     plt.plot(history.history['mae'], label='Training MAE')
#     plt.plot(history.history['val_mae'], label='Validation MAE')
#     plt.title('Model MAE')
#     plt.ylabel('Mean Absolute Error')
#     plt.xlabel('Epoch')
#     plt.legend(loc='upper right')

#     # Plot learning rate if available
#     if 'lr' in history.history:
#         plt.subplot(2, 2, 3)
#         plt.plot(history.history['lr'])  # Fixed this line
#         plt.title('Learning Rate')
#         plt.ylabel('Learning Rate')
#         plt.xlabel('Epoch')
#         plt.yscale('log')

#     plt.tight_layout()
#     plt.show()

# # Improved data preprocessing with cyclical encoding for time features
# def load_and_preprocess_data(filepath, add_features=True):
#     """
#     Load and preprocess lottery data with enhanced feature engineering.
    
#     Args:
#         filepath: Path to the CSV file with lottery data
#         add_features: Whether to add additional time-based features
    
#     Returns:
#         Processed dataframe and numeric data for modeling
#     """
#     # Load data
#     data = pd.read_csv(filepath)
    
#     # Standardize column names
#     data.columns = [col.lower() for col in data.columns]
    
#     # Handle date column naming inconsistency
#     if 'data' in data.columns and 'date' not in data.columns:
#         data.rename(columns={'data': 'date'}, inplace=True)
    
#     # Filter only needed lottery number columns 
#     lottery_cols = [col for col in data.columns if col.startswith('num')]
#     if len(lottery_cols) > 6:
#         bonus_cols = lottery_cols[6:]  # These are the columns to exclude
#         data = data.drop(columns=bonus_cols)
#         print(f"Excluded bonus number column(s): {bonus_cols}")

#     # Convert date column to datetime
#     if 'date' in data.columns:
#         data['date'] = pd.to_datetime(data['date'])

#     # Sort by date (ascending order)
#     if 'date' in data.columns:
#         data = data.sort_values('date').reset_index(drop=True)

#     # Add time-based features if requested
#     if add_features and 'date' in data.columns:
#         # Standard time features
#         data['day_of_week'] = data['date'].dt.dayofweek
#         data['day_of_year'] = data['date'].dt.dayofyear
#         data['month'] = data['date'].dt.month
#         data['quarter'] = data['date'].dt.quarter
#         data['year'] = data['date'].dt.year
#         data['week_of_year'] = data['date'].dt.isocalendar().week
        
#         # Add cyclical encoding for time features (better than one-hot for periodic data)
#         # Encode day of week (0-6) as cyclical feature
#         data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
#         data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
#         # Encode month (1-12) as cyclical feature
#         data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
#         data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
#         # Encode day of year (1-366) as cyclical feature
#         data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 366)
#         data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 366)
        
#         # Drop original categorical features that have been cyclically encoded
#         data = data.drop(columns=['day_of_week', 'month', 'day_of_year'])
        
#         # Keep quarter as one-hot encoded
#         data = pd.get_dummies(data, columns=['quarter'], drop_first=False)
    
#     # Add frequency analysis features
#     lottery_number_cols = [col for col in data.columns if col.startswith('num')]
#     if len(lottery_number_cols) > 0:
#         # Calculate historical frequency of each number
#         all_numbers = pd.Series([num for nums in data[lottery_number_cols].values for num in nums])
#         frequency = all_numbers.value_counts().to_dict()
        
#         # Calculate "hotness" and "coldness" of numbers (based on recent draws)
#         window_size = min(20, len(data))  # Use last 20 draws or all if less
#         recent_numbers = pd.Series([num for nums in data[lottery_number_cols].tail(window_size).values for num in nums])
#         recent_frequency = recent_numbers.value_counts().to_dict()
        
#         # Add these as features to each draw
#         for i, col in enumerate(lottery_number_cols):
#             data[f'{col}_freq'] = data[col].map(lambda x: frequency.get(x, 0))
#             data[f'{col}_recent_freq'] = data[col].map(lambda x: recent_frequency.get(x, 0))
    
#     # Select numeric columns for the model (exclude date column)
#     numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
#     if 'date' in numeric_cols:
#         numeric_cols.remove('date')

#     numeric_data = data[numeric_cols]

#     return data, numeric_data

# # Improved model with attention mechanism
# def build_enhanced_model_with_attention(input_shape, output_shape, learning_rate=0.001):
#     """
#     Build an enhanced LSTM model with attention mechanism for lottery prediction.
    
#     Args:
#         input_shape: Shape of input sequences (window_size, features)
#         output_shape: Number of output values (typically 6 for lottery)
#         learning_rate: Learning rate for Adam optimizer
    
#     Returns:
#         Compiled Keras model
#     """
#     # Import necessary layers for attention mechanism
#     from tensorflow.keras.layers import Layer, Permute, Multiply, Reshape, TimeDistributed, Lambda, Concatenate
#     import tensorflow.keras.backend as K
    
#     # Custom attention layer
#     class AttentionLayer(Layer):
#         def __init__(self, **kwargs):
#             super(AttentionLayer, self).__init__(**kwargs)
        
#         def build(self, input_shape):
#             self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
#                                      initializer="normal")
#             self.b = self.add_weight(name="attention_bias", shape=(input_shape[1], 1),
#                                      initializer="zeros")
#             super(AttentionLayer, self).build(input_shape)
        
#         def call(self, x):
#             # Alignment scores
#             e = K.tanh(K.dot(x, self.W) + self.b)
#             # Remove dimension of size 1
#             e = K.squeeze(e, axis=-1)
#             # Compute the weights
#             alpha = K.softmax(e)
#             # Weighted sum (context vector)
#             context = K.sum(x * K.expand_dims(alpha, axis=-1), axis=1)
#             return context
    
#     # Build model
#     inputs = keras.Input(shape=input_shape)
    
#     # First bidirectional LSTM layer
#     x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
    
#     # Second bidirectional LSTM layer
#     x = Bidirectional(LSTM(128, return_sequences=True))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
    
#     # Apply attention
#     x = AttentionLayer()(x)
    
#     # Dense layers
#     x = Dense(128, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)
    
#     x = Dense(64, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
    
#     # Output layer
#     outputs = Dense(output_shape)(x)
    
#     # Create model
#     model = keras.Model(inputs=inputs, outputs=outputs)
    
#     # Use Adam optimizer with custom learning rate
#     optimizer = Adam(learning_rate=learning_rate)
    
#     model.compile(
#         optimizer=optimizer,
#         loss='mse',
#         metrics=['mae']
#     )
    
#     return model

# # Improved prediction function with probability distribution
# def predict_next_draw_probabilistic(model, data, window_size, scaler=None, num_simulations=1000):
#     """
#     Predict the next lottery draw with probability estimation.
    
#     Args:
#         model: Trained model
#         data: Full preprocessed dataset
#         window_size: Window size used for model training
#         scaler: Scaler used for normalization (if any)
#         num_simulations: Number of Monte Carlo simulations to run
    
#     Returns:
#         Dictionary with predicted numbers and their probabilities
#     """
#     # Get the most recent window of data
#     last_window = data.iloc[-window_size:].values
    
#     # Reshape for model input (adding batch dimension)
#     x_pred = last_window.reshape(1, window_size, data.shape[1])
    
#     # Generate base prediction
#     prediction = model.predict(x_pred).flatten()
    
#     # Ensure we only take the first 6 predictions if there are more
#     if len(prediction) > 6:
#         prediction = prediction[:6]
    
#     # Inverse transform if scaler was used
#     if scaler is not None:
#         # Create dummy array with same shape as the original data
#         dummy = np.zeros((1, scaler.n_features_in_))
#         # Place the prediction in the correct columns
#         dummy[0, :len(prediction)] = prediction
#         # Inverse transform
#         prediction = scaler.inverse_transform(dummy)[0, :len(prediction)]
    
#     # Round to integers
#     rounded_prediction = np.round(prediction).astype(int)
    
#     # Define valid range for lottery numbers
#     min_val, max_val = 1, 49  # Adjust based on your lottery
    
#     # Run Monte Carlo simulations with small random perturbations
#     simulations = []
#     for _ in range(num_simulations):
#         # Add small random noise to the prediction
#         noisy_pred = prediction + np.random.normal(0, 0.5, len(prediction))
#         noisy_pred = np.clip(noisy_pred, min_val, max_val)
#         noisy_pred = np.round(noisy_pred).astype(int)
        
#         # Ensure uniqueness
#         while len(set(noisy_pred)) < len(noisy_pred):
#             for i in range(len(noisy_pred)):
#                 if list(noisy_pred).count(noisy_pred[i]) > 1:
#                     noisy_pred[i] = np.random.randint(min_val, max_val + 1)
        
#         simulations.append(tuple(sorted(noisy_pred)))
    
#     # Calculate probabilities based on simulation frequency
#     from collections import Counter
#     simulation_counts = Counter(simulations)
#     total_sims = len(simulations)
    
#     # Get the most common combinations and their probabilities
#     most_common_draws = simulation_counts.most_common(10)
#     result = {
#         "predicted_numbers": list(rounded_prediction),
#         "top_combinations": [
#             {"combination": list(combo), "probability": count/total_sims}
#             for combo, count in most_common_draws
#         ]
#     }
    
#     return result

# # Improved evaluation metrics
# def evaluate_predictions_advanced(model, x_test, y_test, scaler=None):
#     """
#     Evaluate model predictions with advanced metrics for lottery prediction.
    
#     Args:
#         model: Trained model
#         x_test: Test input sequences
#         y_test: Actual lottery numbers
#         scaler: Scaler used for normalization (if any)
    
#     Returns:
#         Dictionary with various evaluation metrics
#     """
#     # Make predictions
#     y_pred = model.predict(x_test)
    
#     # Inverse transform if scaler was used
#     if scaler is not None:
#         # Create dummy arrays with same shape as the original data
#         dummy_pred = np.zeros((y_pred.shape[0], scaler.n_features_in_))
#         dummy_test = np.zeros((y_test.shape[0], scaler.n_features_in_))
        
#         # Place the values in the correct columns
#         dummy_pred[:, :y_pred.shape[1]] = y_pred
#         dummy_test[:, :y_test.shape[1]] = y_test
        
#         # Inverse transform
#         y_pred = scaler.inverse_transform(dummy_pred)[:, :y_test.shape[1]]
#         y_test = scaler.inverse_transform(dummy_test)[:, :y_test.shape[1]]
    
#     # Round predictions for lottery numbers
#     y_pred_rounded = np.round(y_pred).astype(int)
    
#     # Clip to valid range
#     min_val, max_val = 1, 49  # Adjust based on your lottery
#     y_pred_rounded = np.clip(y_pred_rounded, min_val, max_val)
    
#     # Ensure uniqueness in each prediction (row)
#     for i in range(y_pred_rounded.shape[0]):
#         values = y_pred_rounded[i]
#         unique_values = set()
#         for j in range(len(values)):
#             if values[j] in unique_values:
#                 # Find a number not in the set
#                 for new_val in range(min_val, max_val+1):
#                     if new_val not in unique_values:
#                         values[j] = new_val
#                         break
#             unique_values.add(values[j])
    
#     # Calculate basic error metrics
#     mae = np.abs(y_test - y_pred_rounded).mean()
    
#     # Calculate hit rates (how many numbers match)
#     hit_rates = []
#     for i in range(len(y_test)):
#         actual_set = set(y_test[i])
#         pred_set = set(y_pred_rounded[i])
#         hits = len(actual_set.intersection(pred_set))
#         hit_rates.append(hits)
    
#     hit_distribution = {i: hit_rates.count(i) for i in range(7)}  # 0-6 hits
    
#     # Calculate average number of hits
#     avg_hits = sum(hit_rates) / len(hit_rates)
    
#     # Calculate expected value for random guessing
#     # For a 6/49 lottery, probability of k hits is:
#     # P(k hits) = C(6,k) * C(43,6-k) / C(49,6)
#     import math
#     def combination(n, k):
#         return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    
#     expected_hits_random = sum(
#         k * combination(6, k) * combination(43, 6-k) / combination(49, 6)
#         for k in range(7)
#     )
    
#     # Return comprehensive evaluation results
#     return {
#         "MAE": mae,
#         "Average_Hits": avg_hits,
#         "Hit_Distribution": hit_distribution,
#         "Random_Expected_Hits": expected_hits_random,
#         "Performance_vs_Random": (avg_hits / expected_hits_random) * 100  # As percentage
#     }

# # Main function with improvements
# def main_improved():
#     """
#     Main function to run the improved lottery prediction process.
#     """
#     # 1. Load and preprocess lottery data with enhanced features
#     try:
#         print("Loading and preprocessing data with enhanced features...")
#         full_data, numeric_data = load_and_preprocess_data(
#             r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_lottery\dataset.csv'
#         )
#         print(f"Loaded data with {len(full_data)} draws and {numeric_data.shape[1]} features")
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return
    
#     # 2. Scale the data
#     print("Scaling data...")
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(numeric_data)
#     scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
    
#     # 3. Create sequences
#     print("Creating sequences...")
#     window_size = 15  # Increased window size for more context
#     x, y = prepare_sequences(scaled_df, window_size)
#     print(f"Created {len(x)} sequences with shape {x.shape}")
    
#     # 4. Split data (time series - keep chronological order)
#     # Use earlier 70% for training, middle 15% for validation, last 15% for testing
#     train_size = int(0.7 * len(x))
#     val_size = int(0.15 * len(x))
    
#     x_train, y_train = x[:train_size], y[:train_size]
#     x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
#     x_test, y_test = x[train_size+val_size:], y[train_size+val_size:]
    
#     print(f"Training set: {x_train.shape}")
#     print(f"Validation set: {x_val.shape}")
#     print(f"Test set: {x_test.shape}")
    
#     # 5. Build enhanced model with attention
#     print("Building improved model with attention...")
#     input_shape = (x_train.shape[1], x_train.shape[2])
#     output_shape = y_train.shape[1]
#     model = build_enhanced_model_with_attention(input_shape, output_shape)
#     model.summary()
    
#     # 6. Train model with improved callbacks
#     print("Training model...")
#     callbacks = [
#         # Early stopping to prevent overfitting
#         EarlyStopping(
#             monitor='val_loss',
#             patience=30,
#             restore_best_weights=True,
#             verbose=1
#         ),
        
#         # Model checkpoint to save best model
#         ModelCheckpoint(
#             'best_lottery_model.h5',
#             monitor='val_loss',
#             save_best_only=True,
#             verbose=1
#         ),
        
#         # Reduce learning rate when plateau is reached
#         ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=10,
#             min_lr=0.00001,
#             verbose=1
#         ),
        
#         # Add TensorBoard for better visualization
#         tf.keras.callbacks.TensorBoard(
#             log_dir=f'./logs/lottery_model_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
#             histogram_freq=1
#         )
#     ]
    
#     history = model.fit(
#         x_train, y_train,
#         validation_data=(x_val, y_val),
#         epochs=300,
#         batch_size=32,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # 7. Visualize training
#     print("Visualizing training progress...")
#     visualize_training(history)
    
#     # 8. Evaluate on test set with advanced metrics
#     print("Evaluating model with advanced metrics...")
#     eval_results = evaluate_predictions_advanced(model, x_test, y_test, scaler)
#     print("\nAdvanced evaluation results:")
#     for key, value in eval_results.items():
#         print(f"{key}: {value}")
    
#     # 9. Predict next draw with probability distribution
#     print("\nPredicting next lottery draw with probability distribution...")
#     next_draw_prob = predict_next_draw_probabilistic(model, scaled_df, window_size, scaler)
#     print(f"Predicted numbers for next draw: {next_draw_prob['predicted_numbers']}")
#     print("\nTop predicted combinations and their probabilities:")
#     for i, combo in enumerate(next_draw_prob['top_combinations'][:5]):
#         print(f"{i+1}. {combo['combination']} - {combo['probability']*100:.2f}%")
    
#     # 10. Save the model for future use
#     model.save('improved_lottery_prediction_model.h5')
#     print("Model saved as 'improved_lottery_prediction_model.h5'")
    
#     return model, next_draw_prob









