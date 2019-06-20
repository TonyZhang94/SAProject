# -*- coding: utf-8 -*-

import _pickle as pickle

from CharEmbedding.samples import *
from CharEmbedding.utils import Char2Vec
from CharEmbedding.settings import *
from CharEmbedding.encoder import *

from SAProject.MFBM.sa import *
from SAProject.Classify.model import *


def run(char_timestamp="1559804726"):
    obj = Match(5)
    result = obj.match()
    train, test, dev = obj.make_inputs(result)

    with open("../CharEmbedding/char2info/char2vec.pkl", mode="rb") as fp:
        init_embedding = pickle.load(fp)

    char_timestamp = "1559804726"
    char_model = EncoderModel(holder_dim=holder_dim,
                         hidden_dim=hidden_dim,
                         batch_size=batch_size,
                         init_embedding=init_embedding,
                         checkpoints_path=checkpoints_path.format(char_timestamp)
                         )

    paths = {}
    timestamp = str(int(time.time()))
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join('.', "data_path_save", timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix

    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path

    batch_size_ = 64
    epoch_num_ = 40
    hidden_dim_ = 300
    dropout_keep_prob = 0.5
    lr = 0.001
    clip_grad = 5.0
    tag2label = {-1: -1, 0: 0, 1: 1}
    num_tags = 3
    shuffle = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

    model = BiLSTM_CRF(batch_size_, epoch_num_, hidden_dim_, dropout_keep_prob, lr, clip_grad,
                       tag2label, num_tags, shuffle, paths, char_model=char_model, config=config)
    model.build_graph()

    model.train(train=train, dev=dev)


if __name__ == '__main__':
    run()
