from keras.models import Sequential
from keras.layers import Dense, Dropout , BatchNormalization, Flatten
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam


def neural_network_model(X_train):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1] , activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(196, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))
    model.compile(optimizer = 'adam',loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# def neural_network_model(X_train):
#     model = Sequential()
#     # model.add(Flatten())
#     model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dense(128,activation='relu'))
#     model.add(Dense(4,activation='softmax'))
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
