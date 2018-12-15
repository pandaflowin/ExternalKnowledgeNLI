import os  
import tensorflow as tf
import my_parameters as params
import logger as logger
from my_data_processing import *
from my_model import *
from evaluate import *
from tqdm import tqdm
from nltk.wsd import lesk
import gzip
import pickle

from tensorflow.python import debug as tf_debug

FIXED_PARAMETERS, config = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
if not os.path.exists(FIXED_PARAMETERS["log_path"]):
    os.makedirs(FIXED_PARAMETERS["log_path"])
if not os.path.exists(config.tbpath):
    os.makedirs(config.tbpath)
    config.tbpath = FIXED_PARAMETERS["log_path"]


if config.test:
    logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + "_test.log"
else:
    logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)
if False:#config.debug_model:
    # training_snli, dev_snli, test_snli, training_mnli, dev_matched, dev_mismatched, test_matched, test_mismatched = [],[],[],[],[],[], [], []
    test_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"], shuffle = False)[:499]
    training_snli, dev_snli, test_snli, training_mnli, dev_matched, dev_mismatched, test_mismatched = test_matched, test_matched,test_matched,test_matched,test_matched,test_matched,test_matched
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([test_matched])
    shared_content = load_mnli_shared_content()
else:#config.debug_model == False:
    logger.Log("Loading data SNLI")
    training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"], snli=True)
    dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
    test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)

    logger.Log("Loading data MNLI")
    training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
    dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
    dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])

    test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"], shuffle = False)
    test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"], shuffle = False)

    shared_content = load_mnli_shared_content()

    logger.Log("Loading embeddings")
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([training_mnli, training_snli, dev_matched, dev_mismatched, test_matched, test_mismatched, dev_snli, test_snli])

config.char_vocab_size = len(char_indices.keys())

embedding_dir = os.path.join(config.datapath, "embeddings")
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)


embedding_path = os.path.join(embedding_dir, "mnli_emb_snli_embedding.pkl.gz")

print("embedding path exist")

print(os.path.exists(embedding_path))
if os.path.exists(embedding_path):
    f = gzip.open(embedding_path, 'rb')
    loaded_embeddings = pickle.load(f)
    f.close()
else:
    loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)
    f = gzip.open(embedding_path, 'wb')
    pickle.dump(loaded_embeddings, f)
    f.close()

print('build up classifier')
class Classifier():
    def __init__(self):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step = config.display_step
        self.eval_step = config.eval_step
        self.save_step = config.eval_step
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 
        self.alpha = FIXED_PARAMETERS["alpha"]
        self.config = config
        self.word_indices = word_indices
        self.model = MyModel(self.config, seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train,word_indice =indices_to_words)
        
        logger.Log("Building model from %s.py" %("TEST"))
        self.global_step = self.model.global_step

        # Perform gradient descent
        if not config.test:
            tvars = tf.trainable_variables()
            # print(tf.gradients(self.model.total_cost, tvars))
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.model.total_cost, tvars), config.gradient_clip_value)
            
            # opt = tf.train.AdamOptimizerr(self.learning_rate)(self.learning_rate)
            opt = tf.train.AdadeltaOptimizer(self.learning_rate)
            self.optimizer = opt.apply_gradients(zip(grads, tvars),global_step = self.global_step)
        # tf things: initialize variables and create placeholder
        self.tb_writer = tf.summary.FileWriter(config.tbpath)
        logger.Log("Initializing variables")

        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    '''
        make batch here
        r : (None, max_seqlen, max_seqlen, 5) ralation matrix according to five ralation between words 
        premise : (None, max_seqlen) corresponding indices for words in premise
        hypothesis : (None, max_seqlen) corresponding indices for words in hypothesis
        premise_def : (None, max_seqlen, max_seqlen) corresponding indices for words definition in premise
        hypothesis_def : (None, max_seqlen, max_seqlen) corresponding indices for words definition in hypothesis
        labels : (None, 1) correct type of label (entailment, contradiction, neutral) 
        mask_p : keep sentence length info of premise
        hypothesis_p : keep sentence length info of hypothesis
        mask_p_def : keep sentence length info of premise mask 
        mask_h_def : keep sentence length into of hypothesis mask
        genres : data type, ex: 911, face to face talk ....
    '''
    def batch_matrix(self, dataset, begin_idx, end_idx):
        indice = range(begin_idx, end_idx)
        r = np.zeros((end_idx - begin_idx, self.sequence_length, self.sequence_length, 5))
        premise = np.array([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indice])
        hypothesis = np.array([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indice])
        premise_def, hypothesis_def = np.zeros((2, end_idx - begin_idx, self.sequence_length, self.sequence_length))
        mask_p, mask_h = np.ones((2,end_idx - begin_idx, self.sequence_length))
        mask_p_def, mask_h_def = np.ones((2,end_idx - begin_idx, self.sequence_length, self.sequence_length))
        labels = np.array([dataset[i]['label'] for i in indice])
        genres = np.array([dataset[i]['genre'] for i in indice])
        for i in range(begin_idx, end_idx):
            pre, hyp = dataset[i]['sentence1'].split()[:self.sequence_length], dataset[i]['sentence2'].split()[:self.sequence_length]
            mask_p[i - begin_idx, :len(pre)] = 1
            mask_h[i - begin_idx, :len(hyp)] = 1

            for i_p, p in enumerate(pre):
                # lesk: find the most possible sysnet of word p from word p and whole sentence pre
                tmp = lesk(pre, p)
                if tmp is not None:
                    # get p's definition
                    tmp = tmp.definition().strip('\'()').split()
                else:
                    continue
                    # turn definition words to corresponding indices
                premise_def[i - begin_idx][i_p][:len(tmp)] = ([word_indices[i] if i in word_indices else 0 for i in tmp])[:self.sequence_length]
                mask_p_def[i - begin_idx][i_p][:len(tmp)] = 1
            
            for i_h, h in enumerate(hyp):   
                tmp = lesk(hyp, h)
                if tmp is not None:
                    tmp = tmp.definition().strip('\'()').split()
                else:
                    continue
                hypothesis_def[i - begin_idx][i_h][:len(tmp)] = ([word_indices[i] if i in word_indices else 0 for i in tmp])[:self.sequence_length]
                mask_h_def[i - begin_idx][i_h][:len(tmp)] = 1
            
            for i_p, p in enumerate(pre):
                p_syn = lesk(pre, p)
                if p_syn is None:
                    continue
                for i_h, h in enumerate(hyp):
                    h_syn = lesk(hyp, h)
                    if h_syn is None:
                        continue
                    r[i - begin_idx][i_p][i_h][0] = is_synonym(p_syn, h_syn)
                    r[i - begin_idx][i_p][i_h][1] = is_antonym(p_syn, h_syn)
                    r[i - begin_idx][i_p][i_h][2] = is_hypernym([p_syn], h_syn, 1)
                    r[i - begin_idx][i_h][i_p][3] = is_hypernym([p_syn], h_syn, 1)
                    r[i - begin_idx][i_p][i_h][4] = is_co_hypernym(p_syn, h_syn)
        return r, premise, hypothesis, premise_def, hypothesis_def ,labels, mask_p, mask_h, mask_p_def, mask_h_def, genres

    def train(self, train_mnli, train_snli, dev_mat, dev_mismat, dev_snli):

        sess_config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = sess_config)
        # tfdebug for detect nan error
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess,dump_root='\\debug')
        # self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)   

        self.sess.run(self.init)

        self.step = 0
        self.epoch = 0
        self.best_dev_mat = 0.
        self.best_mtrain_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001]
        self.best_step = 0
        self.train_dev_set = False
        self.dont_print_unnecessary_info = False
        self.collect_failed_sample = False

        # Restore most recent checkpoint if it exists. 
        # Also restore values for best dev-set accuracy and best training-set accuracy
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.completed = False
                dev_acc_mat, dev_cost_mat, confmx = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                best_dev_mismat, dev_cost_mismat, _ = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                best_dev_snli, dev_cost_snli, _ = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                self.best_mtrain_acc, mtrain_cost, _ = evaluate_classifier(self.classify, train_mnli[0:5000], self.batch_size)
                logger.Log("Confusion Matrix on dev-matched\n{}".format(confmx))
                if self.alpha != 0.:
                    self.best_strain_acc, strain_cost, _  = evaluate_classifier(self.classify, train_snli[0:5000], self.batch_size)
                    logger.Log("Restored best matched-dev acc: %f\n Restored best mismatched-dev acc: %f\n Restored best SNLI-dev acc: %f\n Restored best MulitNLI train acc: %f\n Restored best SNLI train acc: %f" %(dev_acc_mat, best_dev_mismat, best_dev_snli,  self.best_mtrain_acc,  self.best_strain_acc))
                else:
                    logger.Log("Restored best matched-dev acc: %f\n Restored best mismatched-dev acc: %f\n Restored best SNLI-dev acc: %f\n Restored best MulitNLI train acc: %f" %(dev_acc_mat, best_dev_mismat, best_dev_snli, self.best_mtrain_acc))
                if config.training_completely_on_snli:
                    self.best_dev_mat = best_dev_snli
            else:
                self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)
        # # Combine MultiNLI and SNLI data. Alpha has a default value of 0, if we want to use SNLI data, it must be passed as an argument.
        beta = int(self.alpha * len(train_snli))

        ### Training cycle
        logger.Log("Training...")
        logger.Log("Model will use %s percent of SNLI data during training" %(self.alpha))
        while True:
            if config.training_completely_on_snli:
                training_data = train_snli
                beta = int(self.alpha * len(training_data))
                if config.snli_joint_train_with_mnli:
                    training_data = train_snli + random.sample(train_mnli, beta)

            else:
                training_data = train_mnli + random.sample(train_snli, beta)

            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)
            # Boolean starting that training has not been completed,
            self.completed = False
            samples = 0
            use_check = 0
            for i in range(total_batch + 1):
                print('batch :',i)
                # make mini batch here


                if i != (total_batch):
                    #make batch here
                    r, premise, hypothesis, premise_def, hypothesis_def ,labels, mask_p, mask_h, mask_p_def, mask_h_def, genres = self.batch_matrix(training_data, i * self.batch_size, (i+1) * self.batch_size)

                else:
                    #make batch here with boundary 
                    r, premise, hypothesis, premise_def, hypothesis_def ,labels, mask_p, mask_h, mask_p_def, mask_h_def, genres = self.batch_matrix(training_data, i * self.batch_size, len(training_data))

            
                feed_dict = {self.model.r_matrix : r,
                            self.model.premise_x : premise,
                            self.model.hypothesis_x : hypothesis,
                            self.model.mask_p : mask_p,
                            self.model.mask_h : mask_h,
                            self.model.premise_def : premise_def,
                            self.model.hypothesis_def : hypothesis_def,
                            self.model.mask_p_def : mask_p_def,
                            self.model.mask_h_def : mask_h_def,
                            self.model.y : labels,
                            self.model.keep_rate_ph: 1.0,
                            self.model.is_train: True
                            }
                
                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging

                if self.step % self.display_step == 0:
                    _, c, summary, logits, acc = self.sess.run([self.optimizer, self.model.total_cost, self.model.summary, self.model.logits, self.model.acc], feed_dict)
                    
                    self.tb_writer.add_summary(summary, self.step)
                    logger.Log("Step: {} completed".format(self.step))
                    # print(labels)
                    # print(logits)
                    print(c)
                    print(acc)
                else:
                    # _, c, logits, acc = self.sess.run([self.optimizer, self.model.total_cost, self.model.logits, self.model.acc], feed_dict)
                     _, c, summary, logits, acc = self.sess.run([self.optimizer, self.model.total_cost, self.model.summary, self.model.logits, self.model.acc], feed_dict)
                    self.tb_writer.add_summary(summary, self.step)
                    logger.Log("Step: {} completed".format(self.step))
                    # print(labels)
                    # print(logits)
                    print(c)
                    print(acc)

                if self.step % self.eval_step == 0:
                    if config.training_completely_on_snli and self.dont_print_unnecessary_info:
                        dev_acc_mat = dev_cost_mat = 1.0
                    else:
                        dev_acc_mat, dev_cost_mat, confmx = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                        logger.Log("Confusion Matrix on dev-matched\n{}".format(confmx))
                    
                    if config.training_completely_on_snli:
                        dev_acc_snli, dev_cost_snli, _ = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                        dev_acc_mismat, dev_cost_mismat = 0,0
                    elif not self.dont_print_unnecessary_info or 100 * (1 - self.best_dev_mat / dev_acc_mat) > 0.04:
                        dev_acc_mismat, dev_cost_mismat, _ = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                        dev_acc_snli, dev_cost_snli, _ = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    else:
                        dev_acc_mismat, dev_cost_mismat, dev_acc_snli, dev_cost_snli = 0,0,0,0

                    if self.dont_print_unnecessary_info and config.training_completely_on_snli:
                        mtrain_acc, mtrain_cost, = 0, 0
                    else:
                        mtrain_acc, mtrain_cost, _ = evaluate_classifier(self.classify, train_mnli[0:5000], self.batch_size)
                    
                    if self.alpha != 0.:
                        if not self.dont_print_unnecessary_info or 100 * (1 - self.best_dev_mat / dev_acc_mat) > 0.04:
                            strain_acc, strain_cost,_ = evaluate_classifier(self.classify, train_snli[0:5000], self.batch_size)
                        elif config.training_completely_on_snli:
                            strain_acc, strain_cost,_ = evaluate_classifier(self.classify, train_snli[0:5000], self.batch_size)
                        else:
                            strain_acc, strain_cost = 0, 0
                        logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t Dev-SNLI acc: %f\t MultiNLI train acc: %f\t SNLI train acc: %f" %(self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc, strain_acc))
                        logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t Dev-SNLI cost: %f\t MultiNLI train cost: %f\t SNLI train cost: %f" %(self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost, strain_cost))
                    else:
                        logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t Dev-SNLI acc: %f\t MultiNLI train acc: %f" %(self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc))
                        logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t Dev-SNLI cost: %f\t MultiNLI train cost: %f" %(self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost))

                if self.step % self.save_step == 0:
                    self.saver.save(self.sess, ckpt_file)
                if config.training_completely_on_snli:
                    dev_acc_mat = dev_acc_snli
                    mtrain_acc = strain_acc
                best_test = 100 * (1 - self.best_dev_mat / dev_acc_mat)
                if best_test > 0:
                    self.saver.save(self.sess, ckpt_file + "_best")
                    self.best_dev_mat = dev_acc_mat
                    self.best_mtrain_acc = mtrain_acc
                    if self.alpha != 0.:
                        self.best_strain_acc = strain_acc
                    self.best_step = self.step
                    logger.Log("Checkpointing with new best matched-dev accuracy: %f" %(self.best_dev_mat))

                if self.best_dev_mat > 0.777 and not config.training_completely_on_snli:
                    self.eval_step = 500
                    self.save_step = 500
                    



                if self.best_dev_mat > 0.780 and not config.training_completely_on_snli:
                    self.eval_step = 100
                    self.save_step = 100
                    self.dont_print_unnecessary_info = True 
                    # if config.use_sgd_at_the_end:
                    #     self.optimizer =  tf.train.GradientDescentOptimizer(0.00001).minimize(self.model.total_cost, global_step = self.global_step)


                if self.best_dev_mat > 0.872 and config.training_completely_on_snli:
                    self.eval_step = 500
                    self.save_step = 500
                
                if self.best_dev_mat > 0.878 and config.training_completely_on_snli:
                    self.eval_step = 100
                    self.save_step = 100
                    self.dont_print_unnecessary_info = True 

                self.step += 1

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)

            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, avg_cost))
            
            self.epoch += 1 
            self.last_train_acc[(self.epoch % 5) - 1] = mtrain_acc

            # Early stopping
            self.early_stopping_step = 35000
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

            if (progress < 0.1) or (self.step > self.best_step + self.early_stopping_step):
                logger.Log("Best matched-dev accuracy: %s" %(self.best_dev_mat))
                logger.Log("MultiNLI Train accuracy: %s" %(self.best_mtrain_acc))
                if config.training_completely_on_snli:
                    self.train_dev_set = True

                    # if dev_cost_snli < strain_cost:
                    self.completed = True
                    break
                else:
                    self.completed = True
                    break

    def classify(self, examples):
        # This classifies a list of examples
        if (test == True) or (self.completed == True):
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            sess_config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = sess_config)
            
            self.sess.run(self.init)
            self.saver.restore(self.sess, best_path)
            logger.Log("Model restored from file: %s" % best_path)

        total_batch = int(len(examples) / self.batch_size)
        pred_size = 3
        logits = np.empty(pred_size)
        # logits = np.zeros((self.batch_size, pred_size))
        genres = []
        costs = 0
        for i in tqdm(range(total_batch + 1), ascii = True):
            if i != total_batch:
                #make batch here
                r, premise, hypothesis, premise_def, hypothesis_def ,labels, mask_p, mask_h, mask_p_def, mask_h_def, minibatch_genres = self.batch_matrix(examples, i * self.batch_size, (i+1) * self.batch_size)
            else:
                #make batch here with boundary 
                r, premise, hypothesis, premise_def, hypothesis_def ,labels, mask_p, mask_h, mask_p_def, mask_h_def, minibatch_genres = self.batch_matrix(examples, i * self.batch_size, len(examples))
            feed_dict = {self.model.r_matrix : r,
                            self.model.premise_x : premise,
                            self.model.hypothesis_x : hypothesis,
                            self.model.mask_p : mask_p,
                            self.model.mask_h : mask_h,
                            self.model.premise_def : premise_def,
                            self.model.hypothesis_def : hypothesis_def,
                            self.model.mask_p_def : mask_p_def,
                            self.model.mask_h_def : mask_h_def,
                            self.model.y : labels,
                            self.model.keep_rate_ph: 1.0,
                            self.model.is_train: True
                            }
            genres += list(minibatch_genres)
            logit, cost = self.sess.run([self.model.logits, self.model.total_cost], feed_dict)
            costs += cost
            logits = np.vstack([logits, logit])


        # if test == True:

        return genres, np.argmax(logits[1:], axis = 1), costs

    def generate_predictions_with_id(self, path, examples):
        if (self.completed == True):
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.saver.restore(self.sess, best_path)
            logger.Log("Model restored from file: %s" % best_path)

        total_batch = int(len(examples) / self.batch_size)
        pred_size = 3
        logits = np.empty(pred_size)
        costs = 0
        IDs = np.empty(1)
        for i in tqdm(range(total_batch + 1), ascii = True):
            if i != total_batch:
                #make batch here
                r, premise, hypothesis, premise_def, hypothesis_def ,labels, mask_p, mask_h, mask_p_def, mask_h_def = self.batch_matrix(examples, i * self.batch_size, (i+1) * self.batch_size)
            else:
                #make batch here with boundary 
                r, premise, hypothesis, premise_def, hypothesis_def ,labels, mask_p, mask_h, mask_p_def, mask_h_def = self.batch_matrix(examples, i * self.batch_size, len(examples))

            feed_dict = {self.model.r_matrix : r,
                            self.model.premise_x : premise,
                            self.model.hypothesis_x : hypothesis,
                            self.model.mask_p : mask_p,
                            self.model.mask_h : mask_h,
                            self.model.premise_def : premise_def,
                            self.model.hypothesis_def : hypothesis_def,
                            self.model.mask_p_def : mask_p_def,
                            self.model.mask_h_def : mask_h_def,
                            self.model.y : labels,
                            self.model.keep_rate_ph: 1.0,
                            self.model.is_train: True
                            }
            logit = self.sess.run(self.model.logits, feed_dict)
            IDs = np.concatenate([IDs, pairIDs])
            logits = np.vstack([logits, logit])
        IDs = IDs[1:]
        logits = np.argmax(logits[1:], axis=1)
        save_submission(path, IDs, logits[1:])
classifier = Classifier()

test = params.train_or_test()


if False:#config.preprocess_data_only:
    pass
elif True:#test == False: 
    classifier.train(training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli)
    logger.Log("Acc on matched multiNLI dev-set: %s" %(evaluate_classifier(classifier.classify, dev_matched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Acc on mismatched multiNLI dev-set: %s" %(evaluate_classifier(classifier.classify, dev_mismatched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Acc on SNLI test-set: %s" %(evaluate_classifier(classifier.classify, test_snli, FIXED_PARAMETERS["batch_size"]))[0])

    sys.exit()

    if config.training_completely_on_snli:
        logger.Log("Generating SNLI dev pred")
        dev_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_dev_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_snli_path, dev_snli)

        logger.Log("Generating SNLI test pred")
        test_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_test_{}.csv".format(modname))
        classifier.generate_predictions_with_id(test_snli_path, test_snli)
        
    else:
        logger.Log("Generating dev matched answers.")
        dev_matched_path = os.path.join(FIXED_PARAMETERS["log_path"], "dev_matched_submission_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_matched_path, dev_matched)
        logger.Log("Generating dev mismatched answers.")
        dev_mismatched_path = os.path.join(FIXED_PARAMETERS["log_path"], "dev_mismatched_submission_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_mismatched_path, dev_mismatched)

else:
    if config.training_completely_on_snli:
        logger.Log("Generating SNLI dev pred")
        dev_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_dev_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_snli_path, dev_snli)

        logger.Log("Generating SNLI test pred")
        test_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_test_{}.csv".format(modname))
        classifier.generate_predictions_with_id(test_snli_path, test_snli)
        
    else:
        logger.Log("Evaluating on multiNLI matched dev-set")
        matched_multinli_dev_set_eval = evaluate_classifier(classifier.classify, dev_matched, FIXED_PARAMETERS["batch_size"])
        logger.Log("Acc on matched multiNLI dev-set: %s" %(matched_multinli_dev_set_eval[0]))
        logger.Log("Confusion Matrix \n{}".format(matched_multinli_dev_set_eval[2]))

        logger.Log("Generating dev matched answers.")
        dev_matched_path = os.path.join(FIXED_PARAMETERS["log_path"], "dev_matched_submission_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_matched_path, dev_matched)
        logger.Log("Generating dev mismatched answers.")
        dev_mismatched_path = os.path.join(FIXED_PARAMETERS["log_path"], "dev_mismatched_submission_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_mismatched_path, dev_mismatched)
