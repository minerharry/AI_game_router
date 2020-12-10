import numpy as np
import keras
from keras import layers

generate = True;
image_shape = (30,30);
data_size = 60000;
base_data_file = "grayscale_random"
avg_data_file = base_data_file+"_average"

if generate:
    data = np.random.rand(data_size,*image_shape);
    np.save(base_data_file,data);
    data2 = [[np.average(datum)] for datum in data];
    np.savez(avg_data_file,data,data2);


inputs = keras.Input(shape=(900,), name="image")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(1, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

zFile = np.load(avg_data_file);
(in_dat,out_dat) = (zFile['arr_0'],zFile['arr_1']);

in_train = in_dat[:50000]
in_val = in_dat[50000:55000]
in_test = in_dat[55000:]
out_train = out_dat[:50000]
out_val = out_dat[50000:55000]
out_test = out_dat[55000:]


model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    in_train,
    out_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(in_val, out_val),
)
