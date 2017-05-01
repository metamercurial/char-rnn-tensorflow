from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os

import utils
from model_sentiment import ModelSentiment
import safety

from tflearn.datasets import imdb

def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=1.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)


def train(args):
    # data_loader = utils.TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = 1000
    args.seq_length = 20
    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        ckpt = safety.safely_check_init_from(args)

    # safety.safely_create_files(args, data_loader)

    # IMDB Dataset loading
    train, test, _ = imdb.load_data(path='imdb.pkl', n_words=args.vocab_size,
                                    valid_portion=0.1)

    data_manager = utils.IMDBDatasetManager(train, test, args.batch_size, args.seq_length, args.vocab_size)

    model = ModelSentiment(args)

    info_after_steps = 50

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        loss_total = 0
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            data_manager.batch_pointer = 0
            state = sess.run(model.initial_state)
            for b in range(data_manager.num_batches):
                start = time.time()
                x, y = data_manager.get_next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)

                # instrument for tensorboard
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * data_manager.num_batches + b)
                end = time.time()

                loss_total += train_loss

                if b % info_after_steps == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                          .format(e * data_manager.num_batches + b,
                                  args.num_epochs * data_manager.num_batches,
                                  e, loss_total / info_after_steps, end - start))
                    loss_total = 0

                if (e * data_manager.num_batches + b) % args.save_every == 0\
                        or (e == args.num_epochs-1 and
                            b == data_manager.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_manager.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
