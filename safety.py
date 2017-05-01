import os
import tensorflow as tf

from six.moves import cPickle

def safely_check_init_from(args):
    # check if all necessary files exist
    assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from
    assert os.path.isfile(
        os.path.join(args.init_from, "config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
    assert os.path.isfile(os.path.join(args.init_from,
                                       "chars_vocab.pkl")), "chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
    ckpt = tf.train.get_checkpoint_state(args.init_from)
    assert ckpt, "No checkpoint found"
    assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

    # # open old config and check if models are compatible
    # with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
    #     saved_model_args = cPickle.load(f)
    # need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
    # for checkme in need_be_same:
    #     assert vars(saved_model_args)[checkme] == vars(args)[
    #         checkme], "Command line argument and saved model disagree on '%s' " % checkme
    # return checkout
    return ckpt

def safely_check_compatibilit(data_loader):
    # open saved vocab/dict and check if vocabs/dicts are compatible
    with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
        saved_chars, saved_vocab = cPickle.load(f)
    assert saved_chars == data_loader.chars, "Data and loaded model disagree on character set!"
    assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

def safely_create_files(args, data_loader):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)