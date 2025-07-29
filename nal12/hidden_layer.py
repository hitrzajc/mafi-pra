import tensorflow as tf
import numpy as np
print("TensorFlow version ",tf.__version__)

from tensorflow import keras
import data_higgs as dh

from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping


BATCH_SIZE = 100

# Seed value
# Apparently you may use different seed values at each stage
SEED_VALUE= 10001
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(SEED_VALUE)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED_VALUE)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(SEED_VALUE)

#-------- routines

def split_xy(rawdata):

    #split features and labels from data
    #prepare the data => normalizations !

    # split
    data_y=rawdata['hlabel'] # labels only: 0.=bkg, 1.=sig
    data_x=rawdata.drop(['hlabel'], axis=1) # features only

    #now prepare the data
    mu = data_x.mean()
    s = data_x.std()
    dmax = data_x.max()
    dmin = data_x.min()

    # normal/standard rescaling
    data_x = (data_x - mu)/s

    # scaling to [-1,1] range
    #data_x = -1. + 2.*(data_x - dmin)/(dmax-dmin)

    # scaling to [0,1] range
    # data_x = (data_x - dmin)/(dmax-dmin)


    return data_x,data_y

import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0

path = "pdf/nn/"
dataset = dh.load_data("data","data1")
data_trn=dataset['train']
data_val=dataset['valid'] 
data_fnames=dataset['feature_names'].to_numpy()[1:]
n_dims=data_fnames.shape[0]

print("HIGGS dataset loaded successfully.")

from sklearn.model_selection import train_test_split

size = int(4.5* 1e5)
print(size)
x_tmp, y_tmp = split_xy(data_trn.iloc[:size])
x_train, x_test,y_train, y_test = train_test_split(x_tmp,y_tmp,test_size=0.1) # 10% split

x_val, y_val = split_xy(data_val.iloc[:len(data_val)])


ds_train = tf.data.Dataset.from_tensor_slices((x_train.to_numpy(),y_train.to_numpy().reshape(-1, 1)))
ds_train = ds_train.repeat()
ds_train = ds_train.batch(BATCH_SIZE,drop_remainder=True)

ds_test = tf.data.Dataset.from_tensor_slices((x_test.to_numpy(),y_test.to_numpy().reshape(-1, 1)))
ds_test = ds_test.repeat()
ds_test = ds_test.batch(BATCH_SIZE,drop_remainder=True)

train_steps=int(x_train.shape[0]/BATCH_SIZE)
test_steps=int(x_test.shape[0]/BATCH_SIZE)
print("Steps train:{} and test:{}".format(train_steps,test_steps))

dnn = Sequential()
dnn.add(Dense(50, input_dim=n_dims, activation='relu'))
dnn.add(Dense(50, input_dim=n_dims, activation='relu'))
dnn.add(Dense(1, activation='sigmoid')) # output layer/value
# plot_model(dnn, to_file='dnn_model.png', show_shapes=True)

dnn.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', 'AUC', 'binary_crossentropy'])
dnn.summary()

    #optional early stopping
eval_metric = 'AUC'
earlystop_callback = EarlyStopping(
        mode='max',
        monitor='val_' + eval_metric,
        patience=5,
        min_delta=0.00001,
        verbose=1
    )

#run the training
dnn_model_history = dnn.fit(ds_train,
        epochs=20,
        steps_per_epoch=train_steps,
        callbacks=[earlystop_callback],
        validation_data=ds_test,
        validation_steps=test_steps,
    )



# Extract weights between 1st and 2nd hidden layer
weights = dnn.layers[1].get_weights()[0]  # Shape: (50, 50)

# Plot weights as heatmap
fig, ax = plt.subplots()
cax = ax.imshow(weights, cmap='coolwarm', aspect='auto')

# Colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Weight Value')

ax.set_title('Weights: Hidden Layer 1 → Hidden Layer 2')
ax.set_xlabel('Neurons in Layer 2')
ax.set_ylabel('Neurons in Layer 1')

plt.savefig(path + 'weights_heatmap.pdf', dpi=200)
