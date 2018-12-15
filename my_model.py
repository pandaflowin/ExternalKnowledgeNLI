import tensorflow as tf
# from util import blocks
from nltk.corpus import wordnet as wn
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d, linear, conv2d, cosine_similarity, variable_summaries, dense_logits, fuse_gate
from my.tensorflow import flatten, reconstruct, add_wd, exp_mask
from tqdm import tqdm
import numpy as np

class MyModel(object):
    def __init__(self, config, seq_length, emb_dim, hidden_dim, emb_train,  embeddings = None, pred_size = 3, context_seq_len = None, query_seq_len = None, word_indice = None):
        ## Define hyperparameters
        # tf.reset_default_graph()
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length
        self.pred_size = pred_size 
        self.context_seq_len = context_seq_len
        self.query_seq_len = query_seq_len

        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='p')
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='h')
        self.mask_p = tf.placeholder(tf.float32, [None, self.sequence_length], name = 'p_mask')
        self.mask_h = tf.placeholder(tf.float32, [None, self.sequence_length], name = 'h_mask')
        
        self.premise_def = tf.placeholder(tf.int32, [None, self.sequence_length, self.sequence_length], name='p_def')
        self.hypothesis_def = tf.placeholder(tf.int32, [None, self.sequence_length, self.sequence_length], name='h_def')
        self.mask_p_def = tf.placeholder(tf.float32, [None, self.sequence_length, self.sequence_length], name = 'p_def_mask')
        self.mask_h_def = tf.placeholder(tf.float32, [None, self.sequence_length, self.sequence_length], name = 'h_def_mask')
        
        self.k_1 = tf.Variable(1, dtype = tf.float32, name = 'k1')
        self.k_2 = tf.Variable(1, dtype = tf.float32, name = 'k2')
        
        self.mask_p_r = self.mask_p[:,::-1]
        self.mask_h_r = self.mask_h[:,::-1]

        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
    
        self.r_matrix = tf.placeholder(tf.float32, [None, self.sequence_length, self.sequence_length, 5])
        self.dropout_keep_rate = tf.train.exponential_decay(config.keep_rate, self.global_step, config.dropout_decay_step, config.dropout_decay_rate, staircase=False, name='dropout_keep_rate')
        config.keep_rate = self.dropout_keep_rate
        tf.summary.scalar('dropout_keep_rate', self.dropout_keep_rate)
        self.y = tf.placeholder(tf.int32, [None], name='label_y')
        self.keep_rate_ph = tf.placeholder(tf.float32, [], name='keep_prob')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        

    ## Fucntion for embedding lookup and dropout at embedding layer
        def emb_drop(E, x):
            emb = tf.nn.embedding_lookup(E, x)
            emb_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(emb, config.keep_rate), lambda: emb)
            return emb_drop

         ### Embedding layer ###

        with tf.variable_scope("emb"):
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                self.E = tf.Variable(embeddings, trainable=emb_train)
                premise_in = emb_drop(self.E, self.premise_x)   #P
                hypothesis_in = emb_drop(self.E, self.hypothesis_x)  #H
                premise_in_def = emb_drop(self.E, self.premise_def)
                hypothesis_in_def = emb_drop(self.E, self.hypothesis_def)
        
        def R_Function(self):
            return tf.reduce_sum(self.k_1 * (self.r_matrix[:,:,:,0] - self.r_matrix[:,:,:,1]) + self.k_2 * (self.r_matrix[:,:,:,2] + self.r_matrix[:,:,:,3] - self.r_matrix[:,:,:,4]), -1)

        def biLSTM(inputs, dim, seq_len, name):
            """
            A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

            Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
            Same shape for cell states.
            """
            with tf.name_scope(name):
                with tf.variable_scope('forward' + name):
                    lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
                with tf.variable_scope('backward' + name):
                    lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

                hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, dtype=tf.float32, scope=name)

            return hidden_states, cell_states
        premise_in_bi_out, _ = biLSTM(premise_in, 300, self.sequence_length, 'p_bilstm')
        hypothesis_in_bi_out, _ = biLSTM(hypothesis_in, 300, self.sequence_length, 'h_bilstm')
        premise_in = premise_in_bi_out[0] * self.mask_p[:,:,None] + premise_in_bi_out[1] * self.mask_p_r[:,:,None]
        hypothesis_in = hypothesis_in_bi_out[0] * self.mask_h[:,:,None] + hypothesis_in_bi_out[1] * self.mask_h_r[:,:,None]
        

        # p：None 48 300, h：None 300 48
        h_T = tf.transpose(hypothesis_in, perm = [0, 2, 1])
        
        # e：None 48 48
        r_score = R_Function(self)
        e = tf.add(tf.matmul(premise_in, h_T), r_score[:,:,None])

        # alpha_c, beta_c：None 48 48 1
        e_exp = tf.exp(tf.clip_by_value(e, 50, -50))
        alpha_c = tf.expand_dims(e_exp / ((tf.reduce_sum(e_exp,2)[:,:,None])), -1)
        beta_c = tf.expand_dims(e_exp / ((tf.reduce_sum(e_exp,1)[:,:,None])), -1)
        
        premise_in_def = tf.reduce_sum(premise_in_def, 2) / (tf.reduce_sum(self.mask_p_def, 2)[:,:,None] + 1e-5)
        hypothesis_in_def = tf.reduce_sum(hypothesis_in_def, 2) / (tf.reduce_sum(self.mask_h_def, 2)[:,:,None] + 1e-5)
        
        # premise_c = tf.add(tf.matmul(alpha_c[:,:,:,0], hypothesis_in_def), premise_in_def) / (1 + tf.reduce_sum(alpha_c[:,:,:,0], 1)[:,:,None])
        # hypothesis_c = tf.add(tf.matmul(beta_c[:,:,:,0], premise_in_def), hypothesis_in_def) / (1 + tf.reduce_sum(beta_c[:,:,:,0], 2)[:,:,None])
        ### previous
        premise_c = tf.reduce_sum(alpha_c * tf.expand_dims(hypothesis_in, 1), 2)
        hypothesis_c = tf.reduce_sum(beta_c * tf.expand_dims(premise_in, 2), 1)
        ###

        # premise_in None 48 300
        # premise_c  None 48 300
        # alpha_c None 48 48
        # r_matrix None 48 48 

        premise_kb = tf.reduce_sum(alpha_c * self.r_matrix, 2)
        hypothesis_kb = tf.reduce_sum(beta_c * self.r_matrix, 1)
    
        # None 48 (300 + 300 + 300 + 300 + 5) --> None 48 1205
        premise_in = tf.concat((premise_in, premise_c, premise_in - premise_c , premise_in * premise_c, premise_kb), 2)
        hypothesis_in = tf.concat((hypothesis_in, hypothesis_c, hypothesis_in - hypothesis_c, hypothesis_in * hypothesis_c, hypothesis_kb), 2)

        filters = 300
        kernel_size = 1
        padding = 'VALID'
        activation = tf.nn.relu
        # kernel_initializer = tf.truncated_normal_initializer()
        
        # premise in None 48 300 -> None 48 300
        # hypothesis_in None 48 300 -> None 48 300

        mean_p, var_p = tf.nn.moments(premise_in, axes=[2])
        mean_h, var_h = tf.nn.moments(hypothesis_in, axes=[2])
        premise_in = (premise_in - mean_p[:,:,None]) / (tf.sqrt(var_p + 1e-5)[:,:,None])
        hypthesis_in = (hypothesis_in - mean_h[:,:,None]) / (tf.sqrt(var_h + 1e-5)[:,:,None])
        tf.summary.scalar('premise_in_after_concat', premise_in[0][0][0])
        premise_in = tf.layers.conv1d(premise_in, filters = filters, kernel_size = kernel_size, padding = padding, activation = activation)
        hypothesis_in = tf.layers.conv1d(hypothesis_in, filters = filters, kernel_size = kernel_size, padding = padding, activation = activation)
        # None 48 305

        tf.summary.scalar('premise_in_after_concat_kb', premise_in[0][0][0])
        
        premise_in = tf.concat((premise_in, premise_kb), 2)
        hypothesis_in = tf.concat((hypothesis_in, hypothesis_kb), 2)

        premise_bi_out, premise_bi_out_states = biLSTM(premise_in, 300, self.sequence_length, "p_lstm_2")
        hypothesis_bi_out, hypothesis_bi_out_states = biLSTM(hypothesis_in, 300, self.sequence_length, "h_lstm_2")
        
        # None 48 300
        # premise_m = premise_in
        # hypothesis_m = hypothesis_in
        premise_m = premise_bi_out[0] * self.mask_p[:,:,None] + premise_bi_out[1] * self.mask_p_r[:,:,None]
        hypothesis_m = hypothesis_bi_out[0] * self.mask_h[:,:,None] + hypothesis_bi_out[1] * self.mask_h_r[:,:,None]
        print(premise_m, hypothesis_m)
        tf.summary.scalar('premise_m', premise_m[0][0][0])
        
        # premise_m = tf.concat(premise_bi_out, 2)
        # hypothesis_m = tf.concat(hypothesis_bi_out, 2)
        
        # None 48 5
        tmp_p_kb = tf.expand_dims(tf.reduce_sum(premise_kb, 2),2)
        tmp_h_kb = tf.expand_dims(tf.reduce_sum(hypothesis_kb, 2),2)
        kernel_size = 1
        filters = 1
        padding = "VALID"
        activation = tf.nn.relu
        # None 48 1
        tmp_p_kb = tf.exp(tf.clip_by_value(tf.layers.conv1d(tmp_p_kb, filters = filters, kernel_size = kernel_size, padding = padding, activation = activation), 50, -50))
        tmp_h_kb = tf.exp(tf.clip_by_value(tf.layers.conv1d(tmp_h_kb, filters = filters, kernel_size = kernel_size, padding = padding, activation = activation), 50, -50))
        
        # None 1 1
        tmp_p_kb_sum = tf.expand_dims(tf.reduce_sum(tmp_p_kb, 1),1)
        tmp_h_kb_sum = tf.expand_dims(tf.reduce_sum(tmp_h_kb, 1),1)
        # None 48 1 * None 48 305 -> None 48 305 -> None 305
        premise_w = tf.reduce_sum((tmp_p_kb / tmp_h_kb_sum) * premise_m, 1)
        hypothesis_w = tf.reduce_sum((tmp_h_kb / tmp_h_kb_sum) * hypothesis_m, 1)
        print(premise_w)
        
        logit1 = tf.reduce_sum(premise_m, 1) / 48
        logit2 = tf.reduce_max(premise_m, 1)

        logit3 = tf.reduce_sum(hypothesis_m, 1) / 48
        logit4 = tf.reduce_max(premise_m, 1)

        # None 305*6
        logits = tf.concat((logit1, logit2, logit3, logit4, premise_w, hypothesis_w), 1)
        tf.summary.scalar('logits', logits[0][0])
        
        activation = tf.nn.tanh
        self.logits = tf.layers.dense(logits, units = self.dim, activation = activation)
        tf.summary.scalar('logits_2', self.logits[0][0])
        
        self.logits = tf.layers.dense(self.logits, units = 3,activation = tf.nn.softmax)
        tf.summary.scalar('logits_3', self.logits[0][0])
        
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1),tf.cast(self.y,tf.int64)), tf.float32))
        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('loss', self.total_cost)

        self.summary = tf.summary.merge_all()

        total_parameters = 0
        for v in tf.global_variables():
            if not v.name.endswith("weights:0") and not v.name.endswith("biases:0") and not v.name.endswith('kernel:0') and not v.name.endswith('bias:0'):
                continue
            print(v.name)
            # print(type(v.name))
            shape = v.get_shape().as_list()
            param_num = 1
            for dim in shape:
                param_num *= dim 
            print(param_num)
            total_parameters += param_num
        print(total_parameters)