# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("MNIST_tw/", one_hot=True, validation_size=1000)

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 150
batch_size = 100

#28x28 -> 24x24
WH = 24
SEC_WH = 12
THIRD_WH = 6
# FOURTH_WH = 4 : origin value
FOURTH_WH = 3
INPUT_SIZE = WH*WH
LABELS = 10

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def sameWHD(self, laststep, currentFilter, nCycle, lv):
        for i in range(nCycle):
            # layer2
            with tf.variable_scope('Level'+ str(lv) +'conv_blocks_%d' % (i + 1), reuse=False):
                # with tf.variable_scope('block1', reuse=False):
                bn1 = tf.layers.batch_normalization(laststep, name="bn1")
                conv1 = tf.layers.conv2d(inputs=bn1, filters=currentFilter, kernel_size=[3, 3],
                                         padding="SAME", activation=tf.nn.relu, name="conv_block1")
                # with tf.variable_scope('block1', reuse=False):
                bn2 = tf.layers.batch_normalization(conv1, name="bn2")
                conv2 = tf.layers.conv2d(inputs=bn2, filters=currentFilter, kernel_size=[3, 3],
                                         padding="SAME", activation=tf.nn.relu, name="conv_block2")
                laststep = laststep + conv2

        return laststep

    def modifyWHD(self, laststep, currentfilter, lv):
        with tf.variable_scope('Mid'+ str(lv) +'decrease_conv_wh_increase_depth'):
            # decrease wh, wh by stride 2
            bn1 = tf.layers.batch_normalization(laststep, name="bn1")
            conv1 = tf.layers.conv2d(inputs=bn1, filters=currentfilter*2, kernel_size=[3, 3], strides=2,
                                     padding="SAME", activation=tf.nn.relu, name="conv_block1")

            # with tf.variable_scope('block1', reuse=False):
            bn2 = tf.layers.batch_normalization(conv1, name="bn2")
            conv2 = tf.layers.conv2d(inputs=bn2, filters=currentfilter*2, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu, name="conv_block2")

            # laststep should be changed to (128,half,half,double)
            pooled_input = tf.nn.avg_pool(laststep, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [currentfilter // 2, currentfilter // 2]])

            laststep = padded_input + conv2
        return laststep

    def lastLevel(self, laststep, currentFilter):
        with tf.variable_scope("Level_Last"):
            bn_com = tf.layers.batch_normalization(laststep)

            # Convolutional Layer #2 and Pooling Layer #2
            conv_com = tf.layers.conv2d(inputs=bn_com, filters=currentFilter, kernel_size=[3, 3], strides=[4, 4],
                                        padding="same", activation=tf.nn.relu)
            pool_com = tf.layers.max_pooling2d(inputs=conv_com, pool_size=[2, 2],
                                               padding="same", strides=2)

            # Dense Layer with Relu
            flat = tf.reshape(pool_com, [-1, 128 * 1 * 1])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, INPUT_SIZE])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, WH, WH, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            conv0 = tf.layers.conv2d(inputs=X_img, filters=4, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            laststep = conv0

            laststep = self.sameWHD(laststep, currentFilter=4, nCycle=5, lv=0)
            laststep = self.modifyWHD(laststep, currentfilter=4, lv=0)
            laststep = self.sameWHD(laststep, currentFilter=8, nCycle=5, lv=1)
            laststep = self.modifyWHD(laststep, currentfilter=8, lv=1)
            laststep = self.sameWHD(laststep, currentFilter=16, nCycle=3, lv=2)
            laststep = self.modifyWHD(laststep, currentfilter=16, lv=2)
            laststep = self.sameWHD(laststep, currentFilter=32, nCycle=3, lv=3)
            laststep = self.modifyWHD(laststep, currentfilter=32, lv=3)
            laststep = self.sameWHD(laststep, currentFilter=64, nCycle=3, lv=4)
            laststep = self.modifyWHD(laststep, currentfilter=64, lv=4)
            laststep = self.sameWHD(laststep, currentFilter=128, nCycle=3, lv=5)
            self.lastLevel(laststep, currentFilter=128)

            print('1 cycle complete')

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('1 _build_net complete')

    def predict(self, x_test, training=False):
        print('predict')
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        print('get_accuracy')
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        # print('train')

        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/resnet")
writer.add_graph(sess.graph)  # Show the graph

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    # summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
    # writer.add_summary(summary, global_step=step)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')



# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
