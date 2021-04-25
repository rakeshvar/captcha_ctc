from tensorflow import keras
from utils import Printer
from scribe.scribe import Scribe
from scribeargs import scribe_args
from model import build_model

"""Initialize """
scriber = Scribe(**scribe_args)
printer = Printer(scriber.alphabet.chars)
num_to_char = scriber.get_char_of_index
print(scriber)

"""Model"""
model = build_model(scriber.nClasses, scriber.nDims, scriber.avg_len)
prediction_model = keras.models.Model(model.get_layer(name="image").input,
                                      model.get_layer(name="softmax").output)
# model.summary()
# prediction_model.summary()


class MyCallBack(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        num_samples = 5
        for (i, batch) in zip(range(num_samples), scriber.data_generator()):
            image, labels = batch[0]
            probabilities = prediction_model.predict(image)[0]
            printer.show_all(labels[0], probabilities, image[0,:,:,0].T, show_images=(i==num_samples-1))

history = model.fit(scriber.data_generator(), steps_per_epoch=100, epochs=100, callbacks=[MyCallBack()])
