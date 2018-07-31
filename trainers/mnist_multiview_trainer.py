from trainers.base_trainer import BaseTrainer
from tqdm import tqdm
import numpy as np

import tensorflow as tf

from utils.metrics import AverageMeter
from utils.logger import DefinedSummarizer
from utils.plots import plot_images

class MnistMultiviewTrainer(BaseTrainer):
    def __init__(self, sess, model, config, logger, data_loader):
        """
        Constructing the Mnist trainer based on the Base Train..
        Here is the pipeline of constructing
        - Assign sess, model, config, logger, data_loader(if_specified)
        - Initialize all variables
        - Load the latest checkpoint
        - Create the summarizer
        - Get the nodes we will need to run it from the graph
        :param sess:
        :param model:
        :param config:
        :param logger:
        :param data_loader:
        """
        super(MnistMultiviewTrainer, self).__init__(sess, model, config, logger, data_loader)

        # load the model from the latest checkpoint
        self.model.load(self.sess)

        # Summarizer
        self.summarizer = logger

        self.x, self.y, self.training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node = tf.get_collection('train')
    
    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        #self.test(self.model.cur_epoch_tensor.eval(self.sess))
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)

    def train_epoch(self, epoch=None):
        """
        Train one epoch
        :param epoch: cur epoch number
        :return:
        """
        # initialize dataset
        self.data_loader.initialize(self.sess, state='train')

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="Train-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, acc = self.train_step()
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        self.sess.run(self.model.global_epoch_inc)

        # summarize
        summaries_dict = {'train/loss_per_epoch': loss_per_epoch.val,
                          'train/acc_per_epoch': acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        self.model.save(self.sess)
        
        print("""
Train-{}  loss:{:.4f} -- acc:{:.4f}
        """.format(epoch, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()

    def train_step(self):
        """
        Run the session of train_step in tensorflow
        also get the loss of that minibatch.
        :return: loss of some metrics to be used in summaries
        """
        _, loss, acc = self.sess.run([self.train_op, self.loss_node, self.acc_node],
                                     feed_dict={self.training: True})
        return loss, acc
    
    
    def test(self, epoch, state='test'):
        # initialize dataset
        self.data_loader.initialize(self.sess, state=state)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Test-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        ensemble_cross_entropy_per_epoch = AverageMeter()
        ensemble_acc_per_epoch = AverageMeter()
        # Iterate over batches
        for _ in tt:
            # One Train step on the current batch
            loss, acc = self.sess.run([self.loss_node, self.acc_node],
                                     feed_dict={self.training: False})
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

            y_sample, sampled_softmax = self.sess.run([self.y, self.model.sampled_softmaxs])
            ensemble_softmax = np.mean(sampled_softmax, axis=1)
            #for _ in range(self.config.num_test_sample):
            #   y_sample, softmax_sample = self.sess.run([self.y, self.model.out_softmax], feed_dict={self.training: False})
            #  ensemble_softmax += np.array(softmax_sample)/self.config.num_test_sample
            
            onehot_y_sample = np.eye(self.config.num_classes)[y_sample]
            ensemble_cross_entropy = np.mean(np.sum(onehot_y_sample*ensemble_softmax, axis=1), axis=0)
            ensemble_acc = np.mean(np.equal(np.argmax(ensemble_softmax, axis=1),y_sample))

            ensemble_cross_entropy_per_epoch.update(ensemble_cross_entropy)
            ensemble_acc_per_epoch.update(ensemble_acc)

        # summarize
        summaries_dict = {'test/loss_per_epoch': loss_per_epoch.val,
                          'test/acc_per_epoch': acc_per_epoch.val, 
                          'test/ensemble_cross_entropy_per_epoch': ensemble_cross_entropy_per_epoch.val,
                          'test/ensemble_acc_per_epoch': ensemble_acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)
        
        print("""
Test-{}  loss:{:.4f} -- acc:{:.4f} -- ensemble cross entropy:{:.4f} -- ensemble acc:{:.4f}
        """.format(epoch, loss_per_epoch.val, acc_per_epoch.val, ensemble_cross_entropy_per_epoch.val, ensemble_acc_per_epoch.val))

        tt.close()

        # generate images
        filename = 'epoch-'+str(self.model.global_step_tensor.eval(self.sess))
        plot_images(self.sample_x(), filename, self.config.etc_dir, shape=(32, 32), n_rows=8)
        tt.close()

    def sample_x(self):
        z_samples = np.random.normal(size=[self.config.batch_size, self.config.num_hidden_neurons])
        x_samples = self.sess.run([self.model.x_decoded], feed_dict={self.model.z: z_samples})[0]
        x_samples = np.array(x_samples).reshape([-1, 32, 32])

        return x_samples

