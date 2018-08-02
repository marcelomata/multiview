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
        self.loss = None
        self.acc = None
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
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv2d(self.x, 64, (5, 5), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')

            with tf.variable_scope('max_pool1'):
                max_pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name='max_pool')

            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv2d(max_pool1, 64, (5, 5), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2')

            with tf.variable_scope('max_pool2'):
                max_pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='max_pool')

            with tf.variable_scope('conv3'):
                conv3 = tf.layers.conv2d(max_pool2, 64, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2')

            with tf.variable_scope('max_pool3'):
                max_pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), name='max_pool')

            with tf.variable_scope('flatten'):
                flattened = tf.layers.flatten(max_pool3, name='flatten')

            with tf.variable_scope('dense1'):
                dense1 = tf.layers.dense(flattened, 500, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense1')

            with tf.variable_scope('dense2'):
                #dense2 = tf.layers.dense(concat, 500, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')
                dense2 = tf.layers.dense(dense1, 500, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')

            with tf.variable_scope('dense3'):
                dense3 = tf.layers.dense(dense2, 2*64, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense3')

            with tf.variable_scope('dense4'):
                dense4 = tf.layers.dense(dense3, 500, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense4')

            with tf.variable_scope('dense5'):
                dense5 = tf.layers.dense(dense4, 500, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense5')

            with tf.variable_scope('out'):
                self.out = tf.layers.dense(dense5, self.config.num_classes,
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
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.out)
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.out))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.out_argmax), tf.float32))

        with tf.variable_scope('train_op'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_op)
        tf.add_to_collection('train', self.cross_entropy)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)

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
