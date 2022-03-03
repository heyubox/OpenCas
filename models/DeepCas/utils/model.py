import sys
import numpy as np
import tensorflow as tf
# from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn

def batched_scalar_mul(w, x):
    x_t = tf.transpose(x, [2, 0, 1, 3])
    shape = x_t.get_shape()
    x_t = tf.reshape(x_t, [int(shape[0]), -1])
    wx_t = tf.multiply(w, x_t)
    res = tf.reshape(wx_t, [int(shape[0]), -1, int(shape[2]), int(shape[3])])
    res = tf.transpose(res, [1, 2, 0, 3])
    return res

def batched_scalar_mul3(w, x):
    x_t = tf.transpose(x, [1, 0, 2])
    shape = x_t.get_shape()
    x_t = tf.reshape(x_t, [int(shape[0]), -1])
    wx_t = tf.multiply(w, x_t)
    res = tf.reshape(wx_t, [int(shape[0]), -1, int(shape[2])])
    res = tf.transpose(res, [1, 0, 2])
    return res

class DeepCas(object):
    def __init__(self, config, sess, node_embed):
        
        self.n_sequences = config.n_sequences
        self.learning_rate = config.learning_rate
        self.emb_learning_rate = config.emb_learning_rate
        self.training_iters = config.training_iters
        self.sequence_batch_size = config.sequence_batch_size
        self.batch_size = config.batch_size
        self.display_step = config.display_step

        self.embedding_size = config.embedding_size
        self.n_input = config.n_input = config.embedding_size
        self.n_steps = config.n_steps
        self.n_hidden_gru = config.n_hidden_gru
        self.n_hidden_dense1 = config.n_hidden_dense1
        self.n_hidden_dense2 = config.n_hidden_dense2
        self.scale1 = config.l1
        self.scale2 = config.l2
        self.scale = config.l1l2
        if config.activation == "tanh":
            self.activation = tf.tanh
        else:
            self.activation = tf.nn.relu
        self.max_grad_norm = config.max_grad_norm
        self.initializer = tf.random_normal_initializer(stddev=config.stddev)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(self.scale1, self.scale2)
        self.dropout_prob = config.dropout_prob
        self.sess = sess
        self.node_vec = node_embed
        self.name = "deepcas"


        self.classification = config.classification
        self.n_class = config.n_class
        self.one_dense_layer = config.one_dense_layer
        
        self.build_input()
        self.build_var()
        self.pred = self.build_model()

        total_parameters = 0
        for variable in tf.trainable_variables():
            if variable is not None:
                # shape is an array of tf.Dimension
                shape = variable.get_shape()

                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
        print('Total_parameters:{}'.format(total_parameters))

        truth = self.y

        # Define loss and optimizer
        if self.classification:
            cost = tf.losses.sparse_softmax_cross_entropy(truth, self.pred)
            self.preds = tf.arg_max(self.pred, dimension=-1)
            pred_soft = tf.nn.softmax(self.pred, dim=-1)
            # cost = tf.reduce_mean(tf.pow(self.pred - truth, 2)) # + self.scale*tf.add_n([self.regularizer(var) for var in tf.trainable_variables()])
            correct_prediction = tf.equal(pred_soft, tf.cast(truth, tf.int64))
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("acc", acc)
            error = acc
        else:
            cost = tf.reduce_mean(tf.pow(self.pred - truth, 2)) # + self.scale*tf.add_n([self.regularizer(var) for var in tf.trainable_variables()])
            error = tf.reduce_mean(tf.pow(self.pred - truth, 2))
            tf.summary.scalar("error", error)

       # optim = self.optimizer
       # gvs = optim.compute_gradients(cost)
       # capped_gvs = [(tf.clip_by_norm(grad, self.max_grad_norm), var)
       #               if not 'embedding' in var.name
       #               else (tf.clip_by_norm(tf.mul(grad, 0.005), self.max_grad_norm), var)
       #               for grad, var in gvs]
       # train_op = optim.apply_gradients(capped_gvs)
        
        var_list1 = [var for var in tf.trainable_variables() if not 'embedding' in var.name]
        var_list2 = [var for var in tf.trainable_variables() if 'embedding' in var.name]
        opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate=self.emb_learning_rate)

        # opt1 = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        # opt2 = tf.train.RMSPropOptimizer(learning_rate=self.emb_learning_rate)
        grads = tf.gradients(cost, var_list1 + var_list2)
        grads1 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[:len(var_list1)]]
        grads2 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1):]]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        train_op = tf.group(train_op1, train_op2)
        # train_op = train_op1
        
        self.cost = cost
        self.error = error
        self.train_op = train_op
        
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        
    
    def build_input(self):
        self.x = tf.placeholder(tf.int32, [None, self.n_sequences, self.n_steps], name="x")
        self.y = tf.placeholder(tf.float32, [None, 1], name="y")
        if self.classification:
            self.y = tf.placeholder(tf.int32, [None, 1], name="y")
        self.sz = tf.placeholder(tf.float32, [None, 1], name="sz")
        
    def build_var(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('embedding'):
                # self.embedding = tf.get_variable('embedding', initializer=tf.constant(self.node_vec, dtype=tf.float32))
                initializer = tf.random_uniform_initializer(minval = 0,maxval = 1,dtype = tf.float32)
                self.embedding = tf.get_variable('embedding', initializer=initializer([self.node_vec.shape[0], self.node_vec.shape[1]]))
                # self.embedding = tf.constant(self.node_vec, dtype=tf.float32)
            with tf.variable_scope('BiGRU'):
                # self.gru_fw_cell = rnn_cell.GRUCell(self.n_hidden_gru)
                # self.gru_bw_cell = rnn_cell.GRUCell(self.n_hidden_gru)
                self.gru_fw_cell = rnn.GRUCell(self.n_hidden_gru)  # for tf1.x
                self.gru_bw_cell = rnn.GRUCell(self.n_hidden_gru)
            with tf.variable_scope('attention'):
                self.p_step = tf.get_variable('p_step', initializer=self.initializer([1, self.n_steps]), dtype=tf.float32)
                self.a_geo = tf.get_variable('a_geo', initializer=self.initializer([1]))
            with tf.variable_scope('dense'):
                self.weights = {
                    'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                             self.n_hidden_dense1]))
                }
                self.biases = {
                    'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([self.n_hidden_dense1]))
                }
                if not self.one_dense_layer:
                    self.weights['dense2'] = tf.get_variable('dense2_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                                            self.n_hidden_dense2]))
                    self.biases['dense2'] = tf.get_variable('dense2_bias', initializer=self.initializer([self.n_hidden_dense2]))
                    middle_hidden = self.n_hidden_dense2
                else:
                    middle_hidden = self.n_hidden_dense1

                if self.classification:
                    self.weights['out_class'] = tf.get_variable('out_weight_class', initializer=self.initializer([middle_hidden, self.n_class]))
                else:
                    self.weights['out']= tf.get_variable('out_weight', initializer=self.initializer([middle_hidden, 1]))
                    self.biases['out']= tf.get_variable('out_bias', initializer=self.initializer([1]))


    def build_model(self):
        with tf.device('/gpu:1'):
            with tf.variable_scope('deepcas') as scope:
                with tf.variable_scope('embedding'):
                    x_vector = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, self.x), 
                                             self.dropout_prob)
                    # (batch_size, n_sequences, n_steps, n_input)

                    # My add, add a Linear layer after embedding
                    # x_vector = tf.reshape(x_vector, [-1, self.n_sequences * self.n_steps, self.embedding_size])
                    # x_vector = tf.nn.leaky_relu(tf.add(tf.matmul(x_vector, self.emb_linear['dense1']), self.emb_bias['dense1']))
                    # x_vector = tf.reshape(x_vector, [-1, self.n_sequences, self.n_steps, self.embedding_size])
                    x_vector = tf.layers.dense(x_vector, self.embedding_size)
                    print(x_vector.shape)

                with tf.variable_scope('BiGRU'):
                    x_vector = tf.transpose(x_vector, [1, 0, 2, 3])
                    # (n_sequences, batch_size, n_steps, n_input)
                    x_vector = tf.reshape(x_vector, [-1, self.n_steps, self.n_input])
                    # (n_sequences*batch_size, n_steps, n_input)
            
                    x_vector = tf.transpose(x_vector, [1, 0, 2])
                    # (n_steps, n_sequences*batch_size, n_input)
                    x_vector = tf.reshape(x_vector, [-1, self.n_input])
                    # (n_steps*n_sequences*batch_size, n_input)
            
                    # Split to get a list of 'n_steps' tensors of shape (n_sequences*batch_size, n_input)
                    # x_vector = tf.split(0, self.n_steps, x_vector)
                    x_vector = tf.split(x_vector, self.n_steps, 0)  # for tf1.x

                    # outputs, _, _ = rnn.bidirectional_rnn(self.gru_fw_cell, self.gru_bw_cell, x_vector,
                    #                                       dtype=tf.float32)
                    outputs, _, _ = rnn.stack_bidirectional_rnn([self.gru_fw_cell], [self.gru_bw_cell], x_vector,
                                                          dtype=tf.float32)  # for tf1.x
                    # hidden_states = tf.transpose(tf.pack(outputs), [1, 0, 2])
                    hidden_states = tf.transpose(tf.stack(outputs), [1, 0, 2])
                    # (n_sequences*batch_size, n_steps, 2*n_hidden_gru)
                    hidden_states = tf.transpose(tf.reshape(hidden_states, [self.n_sequences, -1, self.n_steps, 2*self.n_hidden_gru]), [1, 0, 2, 3])
                    # (batch_size, n_sequences, n_steps, 2*n_hiddent_gru)
        
                with tf.variable_scope('attention'):
                    # attention over sequence steps
                    attention_step = tf.nn.softmax(self.p_step)
                    attention_step = tf.transpose(attention_step, [1,0])
                    attention_result = batched_scalar_mul(attention_step, hidden_states)
                    # (batch_size, n_sequences, n_steps, 2*n_hiddent_gru)
            
                    # attention over sequence batches
                    p_geo = tf.sigmoid(self.a_geo)
                    attention_batch = tf.pow(tf.multiply(p_geo, tf.ones_like(self.sz)), tf.div(1.0 + tf.log(self.sz), tf.log(2.0)))
            
                    attention_batch_seq = tf.tile(attention_batch, [1, self.sequence_batch_size])
                    # print(type(self.n_sequences), type(self.sequence_batch_size))
                    # for i in range(1, int(self.n_sequences/self.sequence_batch_size)):
                    #     attention_batch_seq = tf.concat(1,
                    #                                     [attention_batch_seq,
                    #                                      tf.tile(tf.pow(1-attention_batch, i)*attention_batch,
                    #                                              [1, self.sequence_batch_size])])
                    for i in range(1, int(self.n_sequences/self.sequence_batch_size)):   # for tf1.x
                        attention_batch_seq = tf.concat(
                                                        [attention_batch_seq,
                                                         tf.tile(tf.pow(1-attention_batch, i)*attention_batch,
                                                                 [1, self.sequence_batch_size])], 1)
                    attention_batch_lin = tf.reshape(attention_batch_seq, [-1, 1])
            
                    shape = attention_result.get_shape()
                    shape = [-1, int(shape[1]), int(shape[2]), int(shape[3])]
                    attention_result_t = tf.multiply(attention_batch_lin, tf.reshape(attention_result, [-1, shape[2]*shape[3]]))
                    attention_result = tf.reshape(attention_result_t, [-1, shape[1], shape[2], shape[3]])
                    hidden_graph = tf.reduce_sum(attention_result, reduction_indices=[1, 2])
        
                with tf.variable_scope('dense'):
                    dense1 = self.activation(tf.add(tf.matmul(hidden_graph, self.weights['dense1']), self.biases['dense1']))
                    if not self.one_dense_layer:
                        dense1 = self.activation(tf.add(tf.matmul(dense1, self.weights['dense2']), self.biases['dense2'])) # actually dense2
                    # pred = self.activation(tf.add(tf.matmul(dense1, self.weights['out']), self.biases['out']))
                    if self.classification:
                        pred = tf.matmul(dense1, self.weights['out_class'])
                    else:
                        pred = tf.add(tf.matmul(dense1, self.weights['out']), self.biases['out'])
                    
                return pred
        
    def train_batch(self, x, y, sz):
        self.sess.run(self.train_op, feed_dict={self.x: x, self.y: y, self.sz: sz})
        
    def get_error(self, x, y, sz):
        return self.sess.run(self.error, feed_dict={self.x: x, self.y: y, self.sz: sz})

    def get_pre_true(self,x,y,sz):
        return self.sess.run([self.pred,self.y], feed_dict={self.x: x, self.y: y, self.sz: sz})
