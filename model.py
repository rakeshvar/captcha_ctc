import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length=None, label_length=None):
        if input_length is None or label_length is None:
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


default_layer_specs = (
    ['conv1', 7, (3, 3)],
    ['pool1', (2, 1)],
    ['conv2', 21, (3, 3)],
    ['pool2', (2, 1)],
)


def build_model(num_chars, img_height, img_width=None, will_provide_lengths=False,
                num_features=101,
                num_out=201,
                layerspecs=default_layer_specs):
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float64")
    labels = layers.Input(name="label", shape=(None,), dtype="int64")
    input_length, label_length = (None, None) if not will_provide_lengths else \
                                (layers.Input(name='input_length', shape=[1], dtype='int64'),
                                 layers.Input(name='label_length', shape=[1], dtype='int64'))

    x = input_img
    width_down, height_down, num_kernels = 1, 1, 1
    for lyr in layerspecs:
        if lyr[0].startswith('conv'):
            x = layers.Conv2D(lyr[1], lyr[2], activation="relu",
                              kernel_initializer="he_normal", padding="same", name=lyr[0], )(x)
            num_kernels = lyr[1]
        if lyr[0].startswith('pool'):
            x = layers.MaxPooling2D(lyr[1], name=lyr[0])(x)
            width_down *= lyr[1][0]
            height_down *= lyr[1][1]

    new_shape = (img_width // width_down), (img_height // height_down) * num_kernels
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(num_features, activation="relu", name="dense1")(x)
    x = layers.Dropout(.5)(x)
    x = layers.Bidirectional(layers.LSTM(2 * num_out, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(num_out, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(num_chars + 1, activation="softmax", name="softmax")(x)
    x = layers.Dropout(.5)(x)
    output = CTCLayer(name="ctc_loss")(labels, x, input_length, label_length)

    model_inputs = [input_img, labels]
    if will_provide_lengths:
        model_inputs += [input_length, label_length]

    model = keras.models.Model(inputs=model_inputs, outputs=output, name="ocr_model")
    model.compile(optimizer=keras.optimizers.Adam())
    return model
