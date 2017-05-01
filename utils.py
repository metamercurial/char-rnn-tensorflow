import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences


class IMDBDatasetManager():
    def __init__(self, training_data, test_data, batch_size, seq_length, vocab_size):
        # training data
        self.train_x, self.train_y = training_data
        self.train_x = pad_sequences(self.train_x, maxlen=seq_length, value=0.)
        self.train_y = to_categorical(self.train_y, nb_classes=2)
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_batches = int(len(self.train_x) / batch_size) # len(train_y) should be equals len(train_x) too
        self._preprocess()
        self.batch_pointer = 0

        # test data
        self.test_x, self.test_y = test_data
        self.test_x = pad_sequences(self.test_x, maxlen=seq_length, value=0.)
        self.test_y = to_categorical(self.test_y, nb_classes=2)

    def _preprocess(self):
        # divide into batches
        self.x_batches = np.split(self.train_x.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(self.train_y.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def get_next_batch(self):
        x, y = self.x_batches[self.batch_pointer], self.y_batches[self.batch_pointer]
        self.batch_pointer += 1
        return x, y



class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)

        # 65 characters, 52 alphabets (caps, small) + misc characters incl. whitespace
        self.vocab_size = len(self.chars)
        # dictionary char->index
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        # likely text written in index (of vocabulary matrix)
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        # ydata is simply the text sequence shifted one word down for every word
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
