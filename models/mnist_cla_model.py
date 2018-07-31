from models.base_model import BaseModel
import tensorflow as tf

class MnistClaModel(BaseModel):
    def __init__(self, data_loader, config):
        super(MnistClaModel, self).__init__(config)
        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x = None
        self.y = None
        self.training = None
        self.out_argmax = None
        self.loss_node = None
        self.acc_node = None
        self.optimizer = None
        self.train_op = None

        self.build_model()
        self.init_saver()

    def build_model(self):
        """

        :return:
        """
        
        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor+1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor+1)
        
        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_input()
            self.training = tf.placeholder(tf.bool, name='training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.training)

        """
        Network Architecture
        """
        print('Building network...')
        with tf.variable_scope('network'):
            conv1 = MnistClaModel.conv_bn_relu(self.x, 32, (5, 5), self.training, name='conv1_block')
            
            with tf.variable_scope('max_pool1'):
                max_pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name='max_pool')

            conv2 = MnistClaModel.conv_bn_relu(max_pool1, 64, (5, 5), self.training, name='conv2_block')

            with tf.variable_scope('max_pool2'):
                max_pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='max_pool')

            with tf.variable_scope('flatten'):
                flattened = tf.layers.flatten(max_pool2, name='flatten')

            dense1 = MnistClaModel.dense_bn_relu_dropout(flattened, 1024, 0.5, self.training, name='dense1_block')
            
            with tf.variable_scope('out'):
                self.out = tf.layers.dense(dense1, self.config.num_classes,
                                           kernel_initializer=tf.initializers.truncated_normal, name='out')
                tf.add_to_collection('out', self.out)

        print('Network:')
        print(conv1, max_pool1, conv2, max_pool2, flattened, dense1, self.out, sep='\n')


        """
        Some operators for the training process
        """
        with tf.variable_scope('out_argmax'):
            self.out_argmax = tf.argmax(self.out, axis=-1, output_type=tf.int64, name='out_argmax')

        with tf.variable_scope('loss_acc'):
            self.loss_node = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.out)
            self.acc_node = tf.reduce_mean(tf.cast(tf.equal(self.y, self.out_argmax), tf.float32))

        with tf.variable_scope('train_op'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss_node, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_op)
        tf.add_to_collection('train', self.loss_node)
        tf.add_to_collection('train', self.acc_node)

    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

    @staticmethod
    def conv_bn_relu(x, filters, kernel_size, training, name):
        with tf.variable_scope(name):
            out = tf.layers.conv2d(x, filters, kernel_size, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv')
            out = tf.layers.batch_normalization(out, training=training, name='bn')
            out = tf.nn.relu(out)
            return out

    @staticmethod
    def dense_bn_relu_dropout(x, num_neurons, dropout_rate, training, name):
        with tf.variable_scope(name):
            out = tf.layers.dense(x, num_neurons, kernel_initializer=tf.initializers.truncated_normal, name='dense')
            out = tf.layers.batch_normalization(out, training=training, name='bn')
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, dropout_rate, training=training, name='dropout')
            return out
