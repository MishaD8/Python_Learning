import tensorflow as tf

# print(tf.__version__)

# print("Tensorflow version:", tf.__version__)

print(tf.__version__)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam  
print("Keras modules imported successfully")