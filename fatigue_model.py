# Fatigue model
# Paper: Neural networks for fatigue crack propagation predictions in real-time under uncertainty
# V. Giannella , F. Bardozzo , A. Postiglione , R. Tagliaferri , R. Sepe and R. Citarella
# Corresponding author: vgiannella@unisa.it
# Source code: fbardozzo@unisa.it


import numpy as np
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras.models import Model, Sequential, load_model
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)
import levenberg_marquardt as lm

tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)


def scaler(x, min_v, max_v):
    x0 = np.asarray(x)
    scale = (x - min_v)/(max_v -min_v)
    return scale, x0.mean(), x0.std(), min_v, max_v

def descaler(x, x_mean, x_std, x_min, x_max):
    x = np.asarray(x)
    x = (x * (x_max - x_min)) + x_min
    return  x

def preprocess_files(name_file = 'train_data.csv'):

    df = pd.read_csv(name_file)
    whole_dataset  = df.replace(",", ".")
    whole_dataset  = np.matrix(whole_dataset)
    x0  = np.asarray(whole_dataset[:,(0)] , dtype=np.float64 )
    x1  = np.asarray(whole_dataset[:,(1)] , dtype=np.float64 )
    x2  = np.asarray(whole_dataset[:,(2)] , dtype=np.float64 )
    x3  = np.asarray(whole_dataset[:,(3)] , dtype=np.float64 )
    x4  = np.asarray(whole_dataset[:,(4)] , dtype=np.float64 ) 
    y  = np.asarray(whole_dataset[:,(6)]  , dtype=np.float64) 

    y = pd.DataFrame(y)
    y = np.asarray(y)
    y_min = 4.25
    y_max = 5.5
    y, y_mean, y_std, _, _ = scaler(y, y_min, y_max)

    x_list  =  [x0,x1,x2,x3,x4]
    x_norm  =  []
    for c in range(0, len(x_list), 1):
        x = pd.DataFrame(x_list[c])

        if c == 0:
            min_v = -10.3
            max_v = -9.1
        elif c == 1:
            min_v = 2.3
            max_v = 3.5
        elif c == 2:
            min_v = 11000
            max_v = 17000
        elif c == 3:
            min_v = 2.1
            max_v = 2.9
        elif c == 4:
            min_v = 1.5
            max_v = 2.5

        x, _, _, _, _ = scaler(x, min_v, max_v)


        x = np.asarray(x)
        x_norm.append(x)

    x = np.squeeze(x_norm)

    x = pd.DataFrame(x).T

    x = np.asarray(x)


    return x, y, y_min, y_max, y_mean, y_std

def get_rows(x, y, start=0, to=500):
    train_on_x = []
    train_on_y = []
    count=0
    for i in range(start, to, 1):
        train_on_x.append(x[i,])
        train_on_y.append(y[i])
        count+=1
    train_on_x = np.vstack(train_on_x)
    train_on_x = train_on_x.reshape(1, count, -1)
    train_on_y = np.asarray(train_on_y)
    train_on_y = train_on_y.reshape(1, -1)

    return train_on_x, train_on_y


def fatigue_model0(input_x, n_feat=5, out_size=1, opt_mode="adam"):
    input_shape = input_x.shape
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=6386)
    bias_initializer   =  tf.keras.initializers.GlorotUniform(seed=6386)
    kernel_reg   = tf.keras.regularizers.l2(0.06)
    bias_reg     = tf.keras.regularizers.l2(0.06)

    in_l  = tf.keras.layers.Input(shape=( input_shape[2]))

    print(input_shape)

    f = tf.keras.layers.Dense(7,   activation='sigmoid', use_bias=True)(in_l)

    f = tf.keras.layers.Dense(out_size,  input_dim=n_feat,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_reg,
                              bias_regularizer=bias_reg,
                              activation="linear",

                              use_bias=True)(f)
    model = Model(in_l, f)

    model.summary()

    if opt_mode=="adam":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    elif opt_mode=="sgd":
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    else:

        model_wrapper = lm.ModelWrapper(
            tf.keras.models.clone_model(model))
        opt = tf.keras.optimizers.SGD(learning_rate=1.0)
        model_wrapper.compile(
            optimizer=opt, loss=lm.MeanSquaredError() , metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])

    return model_wrapper



def fatigue_model123(input_x, n_feat=5, out_size=1, opt_mode="adam"):
    input_shape = input_x.shape
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=6386)
    bias_initializer   =  tf.keras.initializers.GlorotUniform(seed=6386)
    kernel_reg   = tf.keras.regularizers.l2(0.06)
    bias_reg     = tf.keras.regularizers.l2(0.06)

    in_l  = tf.keras.layers.Input(shape=( input_shape[2]))

    print(input_shape)

    f = tf.keras.layers.Dense(7,   activation='sigmoid', use_bias=True)(in_l)
    f = tf.keras.layers.Dense(4,    activation='sigmoid', use_bias=True)(f)

    f = tf.keras.layers.Dense(out_size,  input_dim=n_feat,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_reg,
                              bias_regularizer=bias_reg,
                              activation="linear",

                              use_bias=True)(f)
    model = Model(in_l, f)

    model.summary()

    if opt_mode=="adam":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    elif opt_mode=="sgd":
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    else:

        model_wrapper = lm.ModelWrapper(
            tf.keras.models.clone_model(model))

        opt = tf.keras.optimizers.SGD(learning_rate=1.0)
        model_wrapper.compile(
            optimizer=opt, loss=lm.MeanSquaredError() , metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])

    return model_wrapper


def fatigue_model456(input_x, n_feat=5, out_size=1, opt_mode="adam"):
    input_shape = input_x.shape
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=6386)
    bias_initializer   =  tf.keras.initializers.GlorotUniform(seed=6386)
    kernel_reg   = tf.keras.regularizers.l2(0.06)
    bias_reg     = tf.keras.regularizers.l2(0.06)

    in_l  = tf.keras.layers.Input(shape=( input_shape[2]))

    print(input_shape)

    f = tf.keras.layers.Dense(64,   activation='sigmoid', use_bias=True)(in_l)
    f = tf.keras.layers.Dropout(rate=0.1, seed=9 )(f)
    f = tf.keras.layers.Dense(7,    activation='sigmoid', use_bias=True)(f)

    f = tf.keras.layers.Dense(out_size,  input_dim=n_feat,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_reg,
                              bias_regularizer=bias_reg,
                              activation="linear",

                              use_bias=True)(f)
    model = Model(in_l, f)

    model.summary()

    if opt_mode=="adam":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    elif opt_mode=="sgd":
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    else:

        model_wrapper = lm.ModelWrapper(
            tf.keras.models.clone_model(model))

        opt = tf.keras.optimizers.SGD(learning_rate=1.0)
        model_wrapper.compile(
            optimizer=opt, loss=lm.MeanSquaredError() , metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])

    return model_wrapper

def fatigue_model789(input_x, n_feat=5, out_size=1, opt_mode="adam"):
    input_shape = input_x.shape
    print(input_shape)
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=6386)
    bias_initializer   =  tf.keras.initializers.GlorotUniform(seed=6386)
    kernel_reg   = tf.keras.regularizers.l2(0.06)
    bias_reg     = tf.keras.regularizers.l2(0.06)

    in_l  = tf.keras.layers.Input(shape=(input_shape[2]))
    in_l = tf.expand_dims(in_l,2 )

    f = tf.keras.layers.Conv1D(filters=32,  kernel_size=2, activation='sigmoid', use_bias=True)(in_l)
    f = tf.keras.layers.Flatten()(f)

    f = tf.keras.layers.Dropout(rate=0.05, seed=9 )(f)
    f = tf.keras.layers.Dense(7,    activation='sigmoid', use_bias=True)(f)

    f = tf.keras.layers.Dense(out_size,  input_dim=n_feat,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_reg,
                              bias_regularizer=bias_reg,
                              activation="linear",

                              use_bias=True)(f)
    model = Model(in_l, f)

    model.summary()


    if opt_mode=="adam":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    elif opt_mode=="sgd":
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    else:
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())

        model_wrapper = lm.ModelWrapper(
            tf.keras.models.clone_model(model))

        opt = tf.keras.optimizers.SGD(learning_rate=1.0)
        model_wrapper.compile(
            optimizer=opt, loss=lm.MeanSquaredError() , metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])

    return model_wrapper



def fatigue_model131415(input_x, n_feat=5, out_size=1, opt_mode="adam"):
    input_shape = input_x.shape
    print(input_shape)
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=6386)
    bias_initializer   =  tf.keras.initializers.GlorotUniform(seed=6386)
    kernel_reg   = tf.keras.regularizers.l2(0.06)
    bias_reg     = tf.keras.regularizers.l2(0.06)

    in_l  = tf.keras.layers.Input(shape=(input_shape[2]))
    in_l = tf.expand_dims(in_l,2 )

    f = tf.keras.layers.Conv1D(filters=32,  kernel_size=2, activation='sigmoid', use_bias=True)(in_l)
    f = tf.keras.layers.Flatten()(f)

    f = tf.keras.layers.Dense(7,    activation='sigmoid', use_bias=True)(f)


    f = tf.keras.layers.Dense(out_size,  input_dim=n_feat,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_reg,
                              bias_regularizer=bias_reg,
                              activation="linear",

                              use_bias=True)(f)
    model = Model(in_l, f)

    model.summary()


    if opt_mode=="adam":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    elif opt_mode=="sgd":
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])
        return model

    else:
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())

        model_wrapper = lm.ModelWrapper(
            tf.keras.models.clone_model(model))

        opt = tf.keras.optimizers.SGD(learning_rate=1.0)
        model_wrapper.compile(
            optimizer=opt, loss=lm.MeanSquaredError() , metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae', 'mse'])

    return model_wrapper



def get_tensor_slice(x_val, y_val, n):
    x_val = np.squeeze(x_val)
    y_val = np.squeeze(y_val)
    x_val = tf.expand_dims(tf.cast(x_val, tf.float32), axis=-1)
    y_val = tf.expand_dims(tf.cast(y_val, tf.float32), axis=-1)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(n).cache()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return  val_dataset

def test_model(label, checkpoint_sm, model, x_val, y_val, n, n_s, ll):
    model.load_weights(checkpoint_sm)
    x_val = np.squeeze(x_val)
    y_val = np.squeeze(y_val)
    x_val = tf.expand_dims(tf.cast(x_val, tf.float32), axis=-1)
    y_val = tf.expand_dims(tf.cast(y_val, tf.float32), axis=-1)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(n).cache()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    t_eval = model.evaluate(val_dataset, batch_size=n)

    if label == "test1":
        obj    = open('.\report_test1_' + str(ll) + '.txt', "a")
        a =  "size, " + str(n) + ", shift, " + str(n_s)
        for e in t_eval:
            xss  = ", " + str(e)
            a += xss
        a += "\n"
        obj.write(str(a))
        obj.close()

    if label == "test2":
        obj    = open('.\report_test2_' + str(ll) + '.txt', "a")
        a =  "size, " + str(n) + ", shift, " + str(n_s)
        for e in t_eval:
            xss  = ", " + str(e)
            a += xss
        a += "\n"
        obj.write(str(a))
        obj.close()

    print(label + " MSE:", t_eval)
    return model

def model_fit_lm(x_train, y_train, x_val, y_val, x_test, y_test,x_test2, y_test2, label, n_e=100, n_s=0 ):

    if label=="mod1":
        opt_mode="adam"
        model = fatigue_model123(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod2":
        opt_mode="sgd"
        model = fatigue_model123(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod3":
        opt_mode="lm"
        model = fatigue_model123(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod4":
        opt_mode="adam"
        model = fatigue_model456(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod5":
        opt_mode="sgd"
        model = fatigue_model456(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod6":
        opt_mode="lm"
        model = fatigue_model456(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod7":
        opt_mode="adam"
        model = fatigue_model789(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod8":
        opt_mode="sgd"
        model = fatigue_model789(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod9":
        opt_mode="lm"
        model = fatigue_model789(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod10":
        opt_mode="adam"
        model = fatigue_model0(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod11":
        opt_mode="sgd"
        model = fatigue_model0(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod12":
        opt_mode="lm"
        model = fatigue_model0(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod13":
        opt_mode="adam"
        model = fatigue_model131415(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod14":
        opt_mode="sgd"
        model = fatigue_model131415(x_train, opt_mode=opt_mode)
        monitor = "val_mse"
    if label=="mod15":
        opt_mode="lm"
        model = fatigue_model131415(x_train, opt_mode=opt_mode)
        monitor = "val_mse"

    subd = label
    path = "./checkpoints/"
    save_path0 = path + subd + "_" + opt_mode + "_" + str(y_train.shape[1]) + "/"
    save_path  = save_path0 + "_fatigue_model.h5"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor= monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only = True,
        mode='min',
        save_freq='epoch')

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir= save_path0 + "logs")
    train_dataset = get_tensor_slice(x_train, y_train, y_train.shape[1])

    val_dataset   = get_tensor_slice(x_val,   y_val, 1000)
    model.fit(train_dataset, batch_size=1, validation_data=val_dataset, epochs=n_e, callbacks = [tb_callback, cp_callback ])

    model = test_model("validation", save_path, model,  x_val,      y_val,   y_train.shape[1], n_s, label)
    model = test_model("test1",       save_path, model, x_test,   y_test,   y_train.shape[1], n_s, label)
    model = test_model("test2",       save_path, model, x_test2,   y_test2,   y_train.shape[1], n_s, label)
    return model



def model_rebuild(model, x, y_min, y_max, y_mean, y_std):
    y_h = model.predict(x)
    d_y = []
    _, s = y_h.shape
    for i in range(0, s, 1):
        d_yd  = descaler(y_h[:,i], y_mean, y_std, y_min, y_max)
        d_y.append(d_yd)
    return y_h, np.asarray(d_y, dtype=np.float64).flatten()


import argparse

parser = argparse.ArgumentParser(description='Fatigue Neural Networks Benchmark')
parser.add_argument('-s','--size', help='Size of the training set', required=True)
parser.add_argument('-t','--model', help='Type of model and optimizator (mod1, mod2 ... mod9)', required=True)
parser.add_argument('-e','--epochs', help='Number of epochs', required=True)
parser.add_argument('-y','--shift', help='Training shift', required=True)
args = vars(parser.parse_args())

x,   y,  y_min,  y_max,  y_mean, y_std    = preprocess_files("./data/NN_Data_Train.csv")
x2, y2, y_min2, y_max2, y_mean2, y_std2   = preprocess_files("./data/NN_Data_Test.csv")


print("Create training, validation and test set")
x_train, y_train = get_rows(x,         y, int(args["shift"]),    int(args["shift"])+int(args["size"]))
x_val,     y_val = get_rows(x,         y, 8000, 10000)
x_test,   y_test = get_rows(x2,       y2, 0,    10000)

x22, y22, y_min22, y_max22, y_mean22, y_std22   = preprocess_files("./data/NN_Data_Test_2.csv")
x_test2,   y_test2 = get_rows(x22,       y22, 0,    10000)

model = model_fit_lm(x_train, y_train, x_val, y_val, x_test, y_test, x_test2,   y_test2, str(args["model"]), int(args["epochs"]), int(args["shift"]) )


y_h, d_yh = model_rebuild(model, x2, y_min, y_max, y_mean, y_std)
aa = pd.read_csv("./data/NN_Data_Test.csv")
aa["y_pred"]  = np.asarray(d_yh)
aa["error"]   = aa["LN"] - aa["y_pred"]
print(aa.head())
aa.to_excel('./data/test1_' + str(args["model"]) + " dim" + str(args["size"]) + " shift" + str(args["shift"]) +'.xlsx', index_label=False, index=False)


x2, y2, y_min2, y_max2, y_mean2, y_std2   = preprocess_files("./data/NN_Data_Test_2.csv")
x_test,   y_test = get_rows(x2,       y2, 0,    10000)
y_h, d_yh = model_rebuild(model, x2, y_min, y_max, y_mean, y_std)
aa = pd.read_csv("./data/NN_Data_Test_2.csv")
aa["y_pred"]  = np.asarray(d_yh)
aa["error"]   = aa["LN"] - aa["y_pred"]
print(aa.head())
aa.to_excel('./data/test2_' + str(args["model"]) + " dim" + str(args["size"]) + " shift" + str(args["shift"]) +'.xlsx', index_label=False, index=False)

