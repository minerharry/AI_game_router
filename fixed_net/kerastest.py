import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random

generate = True;
image_shape = (30,30);
data_size = 300000;
base_data_file = "fixed_net/data/grayscale_random"
avg_data_file = base_data_file+"_average"

if generate:
    data = np.random.rand(data_size,*image_shape)*0.5 - 0.25, np.random.rand(data_size) - 0.5;
    data = [x + y for (x,y) in zip(*data)]
    np.save(base_data_file,data);
    print()
    data2 = [[np.average(datum)] for datum in data];
    np.savez(avg_data_file,data,data2);

pixel = True; average = False; mnist = False
coordinate = False;

inputs = keras.Input(shape=(902 if coordinate else 1800,), name="image")
x = layers.Dense(64, activation="tanh", name="dense_1")(inputs)
x = layers.Dense(64, activation="tanh", name="dense_2")(x)
x = layers.Dense(64, activation="tanh", name="dense_3")(x)
outputs = layers.Dense(1, activation="tanh", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
keras.utils.plot_model(model, "my_first_model.png")

in_train = in_val = in_test = out_train = out_val = out_test = None;

if pixel:
    basedata = np.load(base_data_file + ".npy");
    basedata = basedata[:int(data_size/10)]

    #print(basedata[:10]);

    basedata = np.repeat(basedata,10,0);

    #print(basedata[:20])
    
    pixel_data = np.random.randint(0,image_shape[0],(len(basedata),2));

    out_dat = [base[pixl[0]][pixl[1]] for (base,pixl) in zip(basedata,pixel_data)];

    if not coordinate:
        pixel_positional = np.ndarray((data_size,30,30));
        for ((x,y),board) in zip(pixel_data,pixel_positional):
            board[x][y] = 1;
        in_dat = np.concatenate((basedata.reshape(len(basedata),900),pixel_positional.reshape(len(basedata),900)),axis=1).astype("float32");
    else:
        in_dat = np.concatenate((basedata.reshape(len(basedata),900),pixel_data),axis=1).astype("float32");

    

#    print(len(out_dat[0]))

    out_dat = np.array(out_dat).astype("float32")

    in_train = in_dat[:data_size - 10000]
    in_val = in_dat[data_size - 10000: data_size - 5000]
    in_test = in_dat[data_size - 5000:]
    out_train = out_dat[:data_size - 10000]
    out_val = out_dat[data_size - 10000: data_size - 5000]
    out_test = out_dat[data_size - 5000:]


    

elif average:
    zFile = np.load(avg_data_file + ".npz");
    (in_dat,out_dat) = (zFile['arr_0'],zFile['arr_1']);

    in_dat = in_dat.reshape(len(in_dat), image_shape[0]*image_shape[1]).astype("float32")
    out_dat = out_dat.astype("float32")

    in_train = in_dat[:data_size - 20000]
    in_val = in_dat[data_size - 20000: data_size - 10000]
    in_test = in_dat[data_size - 10000:]
    out_train = out_dat[:data_size - 10000]
    out_val = out_dat[data_size - 20000: data_size - 10000]
    out_test = out_dat[data_size - 10000:]

    print(in_train[:10])
    print(out_train[:10])
elif mnist:
    (in_train,out_train), (in_test,out_test) = keras.datasets.mnist.load_data()
    in_train = in_train.reshape(60000, 784).astype("float32") / 255
    in_test = in_test.reshape(10000, 784).astype("float32") / 255

    out_train = out_train.astype("float32")
    out_test = out_test.astype("float32")

    in_val = in_train[-10000:]
    out_val = out_train[-10000:]
    in_train = in_train[:-10000]
    out_train = out_train[:-10000]

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    in_train,
    out_train,
    batch_size=64,
    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(in_val, out_val),
)

print(history.history);


# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(in_test, out_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(in_test[:3])
print("predictions shape:", predictions.shape)
print("predictions:", predictions)
print("real values",out_test[:3])

