from .utils import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from math import pi
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class KGTrE():
    def __init__(self, data, embed_dim, metric_dim, batch_size, learning_rate, l2, args):

        tf.compat.v1.reset_default_graph()
        self.data = data['data']
        self.embed_dim = embed_dim
        self.metric_dim = metric_dim
        self.sample_num = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.node_num = 0
        self.features = {}  # initial embedding
        self.relation = {}
        self.dataset = args.dataset
        for nti in data['feature']:
            self.features[nti] = tf.convert_to_tensor(data['feature'][nti])
            self.node_num += len(data['feature'][nti])

        self.HTS = data['HTS']  # Hierarchical Tree Structures
        self.adjs = []
        for si in data['adjs']:
            adjs_si = []
            for adj in si:
                adjs_si.append(cal_matrix(adj))
            self.adjs.append(adjs_si)

        self._construct_network()
        self._optimize_line()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _construct_network(self):
        nti_emb = {}
        for nti in self.features:  # self aggregation
            nti_emb[nti] = []
            self_emb = tf.compat.v1.layers.dense(self.features[nti], self.embed_dim,
                                                 use_bias=False)
            nti_emb[nti].append(self_emb)

        for htsi, adjs in zip(self.HTS, self.adjs):  # hierarchical aggregation
            gruCell = tf.keras.layers.GRUCell(self.embed_dim, name="".join(htsi))
            h_hat = []
            h = []
            for a, nti in enumerate(htsi):
                if a == 0:
                    h_hat.append(self.features[nti])
                else:
                    h.append(
                        tf.matmul(adjs[a - 1], tf.compat.v1.layers.dense(h_hat[a - 1], self.embed_dim, use_bias=False)))# 行列式乘法（2*3）*（3*2）=（2*2）
                    output, state = gruCell(self.features[nti], h[a - 1])
                    nti_emb[nti].append(output)
                    h_hat.append(state)

        self.final_emb = {}
        for nti in nti_emb:  # intergrating
            embs = nti_emb[nti]
            a1 = tf.compat.v1.layers.dense(embs[0], 1, use_bias=False)
            embs = tf.convert_to_tensor(embs)

            a2 = tf.compat.v1.layers.dense(embs[1], 1, use_bias=False)
            a = [a1, a2]
            alpha = tf.nn.softmax(tf.nn.leaky_relu(a), axis=0)
            emb_i = tf.nn.relu(tf.reduce_sum(tf.multiply(alpha, embs), axis=0))
            self.final_emb[nti] = emb_i


        self.embs = tf.concat([tf.compat.v1.layers.dense(self.final_emb[nti], self.metric_dim, use_bias=False)
                               for nti in self.final_emb], axis=0)
        self.metrics = tf.Variable(xavier_init([7, self.metric_dim]))


    def _optimize_line(self):
        """
        Unsupervised traininig
        """
        tf.compat.v1.disable_eager_execution()
        self.u_s = tf.compat.v1.placeholder(name='u_id', dtype=tf.int32,
                                            shape=[self.sample_num, 3])  # node pair with same types
        self.u_d = tf.compat.v1.placeholder(name='u_id', dtype=tf.int32,
                                            shape=[self.sample_num, 4])  # node pair with distinct types
        self.u_i_d = self.u_d[:, 0]
        self.u_j_d = self.u_d[:, 1]
        self.label_d = tf.cast(self.u_d[:, 2], tf.float32)
        self.r = self.u_d[:, 3]

        self.u_i_s = self.u_s[:, 0]
        self.u_j_s = self.u_s[:, 1]
        self.label_s = tf.cast(self.u_s[:, 2], tf.float32)

        self.u_i_embedding_d = tf.matmul(tf.one_hot(self.u_i_d, depth=self.node_num,
                                                    dtype=tf.float32), self.embs)
        self.u_j_embedding_d = tf.matmul(tf.one_hot(self.u_j_d, depth=self.node_num,
                                                    dtype=tf.float32), self.embs)
        self.u_i_embedding_s = tf.matmul(tf.one_hot(self.u_i_s, depth=self.node_num,
                                                    dtype=tf.float32), self.embs)
        self.u_j_embedding_s = tf.matmul(tf.one_hot(self.u_j_s, depth=self.node_num,
                                                    dtype=tf.float32), self.embs)

        # Relational Metric learning
        M_r = tf.nn.embedding_lookup(self.metrics, self.r)
        self.inner_product_d = tf.reduce_sum(M_r * tf.nn.tanh(self.u_i_embedding_d + self.u_j_embedding_d), axis=1)
        self.inner_product_s = tf.reduce_sum(self.u_i_embedding_s * self.u_j_embedding_s,
                                             axis=1)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label_d * self.inner_product_d)
                                    ) - tf.reduce_mean(tf.log_sigmoid(self.label_s * self.inner_product_s))
        self.l2_loss = self.l2 * sum(tf.nn.l2_loss(var)  # l2 norm
                                     for var in tf.trainable_variables() if 'bias' not in var.name)
        self.loss = self.loss + self.l2_loss
        self.line_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss)

    def train_line(self, u_s, u_d):
        """
        Train one minibatch.
        """
        feed_dict = {self.u_s: u_s, self.u_d: u_d}
        _, loss = self.sess.run((self.line_optimizer, self.loss), feed_dict=feed_dict)
        return loss

    def cal_embed(self):
        return self.sess.run(self.final_emb)

    def cal_relation(self):
        final_emb = self.sess.run(self.final_emb)
        r_pv = []
        r_vp = []
        r_vc = []
        r_cv = []
        r_ap = []
        r_pa = []

        for i in range(len(self.data[0])):
            r_vp.append(final_emb['V'][self.data[2][i]] - final_emb['P'][self.data[1][i]])
            r_pv.append(final_emb['P'][self.data[1][i]] - final_emb['V'][self.data[2][i]])
            r_cv.append(final_emb['C'][self.data[3][i]] - final_emb['V'][self.data[2][i]])
            r_vc.append(final_emb['V'][self.data[2][i]] - final_emb['C'][self.data[3][i]])
            r_pa.append(final_emb['P'][self.data[1][i]] - final_emb['A'][self.data[0][i]])
            r_ap.append(final_emb['A'][self.data[0][i]] - final_emb['P'][self.data[1][i]])
        pv = np.average(np.array(r_pv), axis=0).reshape(1, self.embed_dim)
        vp = np.average(np.array(r_vp), axis=0).reshape(1, self.embed_dim)
        vc = np.average(np.array(r_vc), axis=0).reshape(1, self.embed_dim)
        cv = np.average(np.array(r_cv), axis=0).reshape(1, self.embed_dim)
        ap = np.average(np.array(r_ap), axis=0).reshape(1, self.embed_dim)
        pa = np.average(np.array(r_pa), axis=0).reshape(1, self.embed_dim)
        a = np.concatenate((pv, vp), axis=0)
        b = np.concatenate((vc, cv), axis=0)
        c = np.concatenate((ap, pa), axis=0)
        d = np.concatenate((a, b), axis=0)
        self.relation = np.concatenate((d, c), axis=0)
        self.relation = torch.tensor(self.relation)
        return self.relation