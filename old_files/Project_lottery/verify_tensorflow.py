import tensorflow as tf

# print(tf.__version__)

# print("Tensorflow version:", tf.__version__)

print(tf.__version__)
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam   # type: ignore
print("Keras modules imported successfully")