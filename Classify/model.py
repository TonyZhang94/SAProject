import numpy as np
import os, time, sys
import tensorflow as tf
import random

from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
# from char_data import pad_sequences, batch_yield
# from utils import get_logger
# from eval import conlleval
from CharEmbedding.utils import Char2Vec, CutChar


def sentence2embedding(sent, obj_char2vec, init_embedding):
    one_input = list()
    for ch in sent:
        try:
            vec = init_embedding[ch]
        except KeyError:
            vec = obj_char2vec.char2vec(ch)
            init_embedding[ch] = vec
        one_input.append(vec)
    return one_input


def batch_yield(data, batch_size, tag2label, obj_char2vec, obj_cut_char, init_embedding, shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    sequence_length = 0
    for (sent_, tag_) in data:
        sequence_length = max(sequence_length, len(sent_))
    for (sent_, tag_) in data:
        sent_ = obj_cut_char.cut(sent_)
        while len(sent_) < sequence_length:
            sent_.append("<BLANK>")
        sent_ = sentence2embedding(sent_, obj_char2vec, init_embedding)
        label_ = tag_
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


class BiLSTM_CRF(object):
    def __init__(self, batch_size, epoch_num, hidden_dim, dropout_keep_prob, lr, clip_grad,
                       tag2label, num_tags, shuffle, paths, char_model, config):
        self.batch_size = batch_size  # 64
        self.epoch_num = epoch_num  # 40
        self.hidden_dim = hidden_dim  # 300
        self.dropout_keep_prob = dropout_keep_prob  # 0.5
        self.lr = lr  # 0.001 learning rate
        self.clip_grad = clip_grad

        self.tag2label = tag2label
        self.num_tags = num_tags  # 3
        self.shuffle = shuffle  # True

        self.model_path = paths['model_path']  # data_path_save/{timestamp}/checkpoints/model-XXX
        self.summary_path = paths['summary_path']  # data_path_save/{timestamp}/summaries/
        # self.logger = get_logger(paths['log_path'])  # data_path_save/{timestamp}/results/log.txt
        self.result_path = paths['result_path']  # data_path_save/{timestamp}/results/
        self.config = config

        self.char_model = char_model
        self.obj_char2vec = Char2Vec()
        self.obj_cut_char = CutChar()

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        # self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.input_embedding = self.char_model.embedding
        self.labels = tf.placeholder(tf.int32, shape=[None, self.num_tags], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            # _word_embeddings = tf.Variable(self.embeddings,
            #                                dtype=tf.float32,
            #                                trainable=self.update_embedding,
            #                                name="_word_embeddings")
            # word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
            #                                          ids=self.word_ids,
            #                                          name="word_embeddings")
            word_embeddings = self.input_embedding
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            # cell_fw = LSTMCell(self.hidden_dim)
            # cell_bw = LSTMCell(self.hidden_dim)
            cell_fw = LSTMCell(300)
            cell_bw = LSTMCell(300)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            self.shapes = dict()
            self.shapes_output_fw_seq = tf.shape(output_fw_seq)
            self.shapes_output_bw_seq = tf.shape(output_bw_seq)

            # output_fw_seq [ 64, 288, 300 ] [batch_size, 一个batch中最长的评价，size of ht]
            # output_bw_seq [ 64, 288, 300 ] [batch_size, 一个batch中最长的评价，size of ht]
            shape_info = tf.shape(output_fw_seq)
            d1, d2, d3 = shape_info[0], shape_info[1], shape_info[2]

            output_fw_seq = tf.slice(output_fw_seq, [0, d2-1, 0], [d1, 1, d3])
            output_fw_seq = tf.reshape(output_fw_seq, [-1, d3])

            output_bw_seq = tf.slice(output_bw_seq, [0, 0, 0], [d1, 1, d3])
            output_bw_seq = tf.reshape(output_bw_seq, [-1, d3])

            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            self.shape_output1 = tf.shape(output)
            output = tf.nn.dropout(output, self.dropout_pl)
            self.shape_output2 = tf.shape(output)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            self.shape_W = tf.shape(W)  # W [ 600, 3 ]
            self.shape_b = tf.shape(b)  # b [ 3 ]

            s = tf.shape(output)  # [ 64, 288, 600 ]
            self.shape_s = s

            # [ 64, 600 ]
            # output = tf.reshape(output, [-1, 2*self.hidden_dim])  # [ 64*288, 600 ]
            # self.shape_output3 = tf.shape(output)

            pred = tf.matmul(output, W) + b  # [ 64, 3 ]
            self.shape_pred = tf.shape(pred)

            # self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])  # [ 64, 3 ]
            self.logits = pred
            self.shape_logits = tf.shape(self.logits)

    def loss_op(self):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # grads_and_vars = optim.compute_gradients(self.loss)
            # grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            # self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.tag2label,
                              self.obj_char2vec, self.obj_cut_char, self.char_model.init_embedding, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            # _, loss_train, summary, step_num_, \
            # shapes_output_fw_seq, shapes_output_bw_seq, \
            # shape_output1, shape_output2, shape_W, shape_b, \
            # shape_s, shape_pred, shape_logits = \
            #     sess.run([self.train_op, self.loss, self.merged, self.global_step,
            #               self.shapes_output_fw_seq, self.shapes_output_fw_seq,
            #               self.shape_output1, self.shape_output2, self.shape_W, self.shape_b,
            #               self.shape_s, self.shape_pred, self.shape_logits],
            #              feed_dict=feed_dict)
            #
            # print("shapes_output_fw_seq", shapes_output_fw_seq)
            # print("shapes_output_bw_seq", shapes_output_bw_seq)
            # print("shape_output1", shape_output1)
            # print("shape_output2", shape_output2)
            # print("shape_W", shape_W)
            # print("shape_b", shape_b)
            # print("shape_s", shape_s)
            # print("shape_pred", shape_pred)
            # print("shape_logits", shape_logits)
            #
            # print("record_loss", loss_train)
            # exit()

            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                print(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        print('===========validation / test===========')
        # label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        # self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        # labels_, _ = pad_sequences(labels, pad_mark=0)
        labels_ = labels

        feed_dict = {self.char_model.inputs: seqs,
                     self.labels: labels_,
                     self.sequence_lengths: seq_len_list}

        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list
