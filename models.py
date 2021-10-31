
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

def SimpleModel(n_action):
    """
    Our embeddings have shape (1, 10), which affects our hidden layer
    dimension and output dimension
    Flattening resolves potential issues that would arise otherwise
    """
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))
    return model