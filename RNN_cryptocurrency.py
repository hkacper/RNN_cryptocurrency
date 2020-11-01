import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Predict next 3 minutes of LTC based on last 60 minutes
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'
EPOCHS = 10
BATCH_SIZE = 64
NAME = f'{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")}'
 
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_minutes = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_minutes.append([n for n in i[:-1]])
        if len(prev_minutes) == SEQ_LEN:
            sequential_data.append([np.array(prev_minutes), i[-1]])
    
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
             sells.append([seq, target])
        elif target  == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X).astype('float32'), np.array(y) 



main_df = pd.DataFrame()

# Merge csv to one file 
ratios = ['BTC-USD', 'LTC-USD' , 'ETH-USD', 'BCH-USD']
for ratio in ratios:
    dataset = f'crypto_data/{ratio}.csv'
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={'close':f'{ratio}_close', 'volume':f'{ratio}_volume'}, inplace=True)

    df.set_index('time', inplace=True)
    df = df[[f'{ratio}_close', f'{ratio}_volume']]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)


main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

x_train, y_train = preprocess_df(main_df)
x_val, y_val = preprocess_df(validation_main_df)

# x_train = np.asarray(x_train)
# y_train = np.asarray(y_train)
# x_val = np.asarray(x_val)
# y_val = np.asarray(y_val)

print(f'train data: {len(x_train)}, validation: {len(x_val)}')
# print(f'Dont buys: {y_train.count(0)}, buys: {y_train.count(1)}')
# print(f'VALIDATION Dont buys: {y_val.count(0)}, buys: {y_val.count(1)}')

model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation='tanh', input_shape=(x_train.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = 'RNN_Final-{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint('models\{}.model'.format(filepath), monitor='val_acc', verbose=10, save_best_only=True, mode='max')

history = model.fit(x_train,
                    y_train, 
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_val, y_val),
                    callbacks=[tensorboard, checkpoint])