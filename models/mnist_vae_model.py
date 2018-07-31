from models.base_model import BaseModel
import tensorflow as tf

class MnistVaeModel(BaseModel):
    def __init__(self, data_loader, config):
        super(MnistVaeModel, self).__init__(config)
        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x = None
        self.y = None
        self.training = None
        self.loss_node = None
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
        print('Padding:')
        padded_x = tf.pad(self.x, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), "CONSTANT")
        print(padded_x)
        z_mean, z_stddev = MnistVaeModel.encoder(padded_x, self.config.num_hidden_neurons, name='encoder1')

        self.z = MnistVaeModel.sampler(z_mean, z_stddev, self.config.batch_size, self.config.num_hidden_neurons, name='sampler1')
        self.x_decoded = MnistVaeModel.decoder(self.z, name='decoder1')

        """
        Some operators for the training process
        """
        with tf.variable_scope('loss'):
            # calculate KL divergence between approximate posterior q and prior p
            with tf.variable_scope('KL'):
                kl = MnistVaeModel.normal_kl(z_mean, z_stddev)

            # calculate reconstruction error between decoded sample and original input batch
            with tf.variable_scope('log_lik'):
                log_lik = MnistVaeModel.bern_log_lik(padded_x, self.x_decoded)

            self.loss_node = (kl+log_lik)/self.config.batch_size

        with tf.variable_scope('train_op'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss_node, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_op)
        tf.add_to_collection('train', self.loss_node)

    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

    @staticmethod
    def encoder(x, num_hidden_neurons, name='encoder'):
        """
        Encoder Architecture
        """
        with tf.variable_scope(name):
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv2d(x, 64, (5, 5), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')

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
                dense1 = tf.layers.dense(flattened, 500, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal, name='dense1')

            with tf.variable_scope('dense2'):
                dense2 = tf.layers.dense(dense1, 2*num_hidden_neurons, kernel_initializer=tf.initializers.truncated_normal, name='dense2')

            with tf.variable_scope('z_params'):
                z_mean, z_logvar = dense2[:, :num_hidden_neurons], dense2[:, num_hidden_neurons:]
                z_stddev = tf.sqrt(tf.exp(z_logvar))

        print('Encoder:')
        print(conv1, max_pool1, conv2, max_pool2, conv3, max_pool3, flattened, dense1, dense2, [z_mean, z_stddev], sep='\n')
        return z_mean, z_stddev
    
    @staticmethod 
    def sampler(z_mean, z_stddev, batch_size, num_hidden_neurons, name='sampler'):
        with tf.variable_scope(name):
            epsilon = tf.random_normal(shape=[batch_size, num_hidden_neurons])
            z = z_mean+z_stddev*epsilon
        
        print('Sampler:')
        print(z)
        return z


    @staticmethod
    def decoder(z, name='decoder'):
        with tf.variable_scope(name):
            with tf.variable_scope('dense1'):
                dense1 = tf.layers.dense(z, 500, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal, name='dense1')

            with tf.variable_scope('dense2'):
                dense2 = tf.layers.dense(dense1, 1024, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal, name='dense2')

            with tf.variable_scope('unflatten'):
                unflattened = tf.reshape(dense2, [-1, 4, 4, 64])

            with tf.variable_scope('deconv1'):
                deconv1 = tf.layers.conv2d_transpose(unflattened, 64, (5, 5), padding='same', strides=(2, 2), activation=tf.nn.relu, name='deconv1')

            with tf.variable_scope('deconv2'):
                deconv2 = tf.layers.conv2d_transpose(deconv1, 64, (5, 5), padding='same', strides=(2, 2), activation=tf.nn.relu, name='deconv2')
            
            with tf.variable_scope('deconv3'):
                deconv3 = tf.layers.conv2d_transpose(deconv2, 1, (5, 5), padding='same', strides=(2, 2), activation=tf.nn.sigmoid, name='deconv3')

        print('Decoder:')
        print(dense1, dense2, unflattened, deconv1, deconv2, deconv3, sep='\n')
        return deconv3

    @staticmethod
    def normal_kl(mean, stddev, eps=1e-8):
        """
        Calculates KL Divergence between q~N(mu, sigma^T * I) and p~N(0, I).
        q(z|x) is the approximate posterior over the latent variable z,
        and p(z) is the prior on z.
        :param mu: Mean of z under approximate posterior.
        :param sigma: Standard deviation of z
            under approximate posterior.
        :param eps: Small value to prevent log(0).
        :return: kl: KL Divergence between q(z|x) and p(z).
        """
        
        var = tf.square(stddev)
        kl = 0.5*tf.reduce_sum(tf.square(mean)+var-1.-tf.log(var+eps))
        return kl
    
    @staticmethod
    def bern_log_lik(x, x_recon, eps=1e-8):
        """
        Calculates negative log likelihood -log(p(x|z)) of outputs,
        assuming a Bernoulli distribution.
        :param targets: MNIST images.
        :param outputs: Probability distribution over outputs.
        :return: log_like: -log(p(x|z)) (negative log likelihood)
        """
        log_lik = -tf.reduce_sum(x*tf.log(x_recon+eps)+(1.-x)*tf.log((1.-x_recon)+eps))
        return log_lik
            