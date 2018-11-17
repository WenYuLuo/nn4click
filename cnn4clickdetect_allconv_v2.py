import tensorflow as tf
import GetData
import numpy as np
import random
import math
import os
import matplotlib.pylab as pl
from scipy import signal
import scipy.io as sio
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

# # batch normalization 函数内变量定义
# MOVING_AVERAGE_DECAY = 0.9997  # 滑动平均系数
# BN_DECAY = MOVING_AVERAGE_DECAY  # BN滑动平均系数
# BN_EPSILON = 0.001  # 避免计算的方差为零，使得归一化除方差无意义
# UPDATE_OPS_COLLECTION = 'update_ops' # must be grouped with training op
# CNN_VARIABLES = 'cnn_variables'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, padding="SAME"):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def maxpooling_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")


# def batchnorm(x, is_training):
#     # TODO: 理解并尝试使用tf.layers.batch_normalization
#     x_shape = x.get_shape()
#     params_shape = x_shape[-1:]
#
#     axis = list(range(len(x_shape) - 1))
#
#     beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
#     gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())
#
#     moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
#     moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)
#
#     # These ops will only be preformed when training.
#     mean, variance = tf.nn.moments(x, axis)
#     update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
#     update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
#
#     tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
#     tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
#
#     mean, variance = control_flow_ops.cond(
#         is_training, lambda: (mean, variance),
#         lambda: (moving_mean, moving_variance))
#
#     return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
#
#
# def _get_variable(name,
#                   shape,
#                   initializer,
#                   weight_decay=0.0,
#                   dtype='float',
#                   trainable=True):
#     "A little wrapper around tf.get_variable to do weight decay and add to"
#     "resnet collection"
#     with tf.variable_scope('get_variabel', reuse=None):
#         if weight_decay > 0:
#             regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
#         else:
#             regularizer = None
#         collections = [tf.GraphKeys.VARIABLES, CNN_VARIABLES]
#         return tf.get_variable(name,
#                                shape=shape,
#                                initializer=initializer,
#                                dtype=dtype,
#                                regularizer=regularizer,
#                                collections=collections,
#                                trainable=trainable)

def batchnorm(Ylogits, is_test, out_size, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(decay=0.998)
    bnepsilon = 1e-5
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, shift, scale, bnepsilon)
    return Ybn


def generate_impulse(data_template_path, add_noise_path, num, noise_propotion=0.4):
    if os.path.exists('./data_txt/manual_impulse.txt'):
        print('loading manual impluse...')
        impulse = np.loadtxt('./data_txt/manual_impulse.txt')
        label = np.loadtxt('./data_txt/manual_impulse_label.txt')
        print('done')
        data_length = impulse.shape[0]
        if num < data_length:
            return impulse[:num], label[:num]
    print('lack enough sample, prepared to generate manual impulse...')
    data_list = GetData.getFileName(data_template_path)
    noise_list = GetData.getFileName(add_noise_path)
    count = 0
    while count < num:
        rand_click = random.randint(0, len(data_list)-1)
        click = GetData.readWav(data_list[rand_click])
        # click = amplitude_norm(click)
        click = click.reshape((1, -1))
# generate impulse noise
        lamda = random.random() + 0.5
        impulse_width = random.randint(2, 8)
        impluse_max_value = np.amax(click)
        # reverse_max_value = -impluse_max_value / 50
        reverse_max_value = 0
        half_impulse = lamda * np.linspace(reverse_max_value, impluse_max_value, impulse_width, endpoint=False)
        another_half = half_impulse[::-1]
        half_impulse = half_impulse.reshape((1, -1))
        another_half = another_half.reshape((1, -1))
        tmp = another_half[0][1:impulse_width]
        single_impulse = np.hstack((half_impulse, tmp.reshape((1, -1))))

        # pl.plot(single_impulse[0])
        # pl.show()

        interval = random.randint(0, 256)
        interval_blank = np.zeros((1, interval))
        if interval % 2 == 0:
            impulse_interval = random.randint(5, 12)
            double_impulse = np.hstack((single_impulse, np.zeros((1, impulse_interval))))
            double_impulse = np.hstack((interval_blank, double_impulse, -1 * single_impulse))
            generate_impulse = double_impulse
        else:
            generate_impulse = np.hstack((interval_blank, single_impulse))
        # generate_impulse = generate_impulse.reshape((1, -1))
        size_impulse = generate_impulse.shape[1]
        if size_impulse < 256:
            supplement = np.zeros((1, 256-size_impulse))
            generate_impulse = np.hstack((generate_impulse, supplement))
            rand_noise = random.randint(0, len(noise_list)-1)
            noise = GetData.readWav(noise_list[rand_noise])
            rand_seg = random.randint(0, len(noise)-256)
            picked_noise = noise[rand_seg:rand_seg+256].reshape(1, -1)
            generate_impulse = (1-noise_propotion)*generate_impulse + noise_propotion*picked_noise  # add random noise
            # pl.plot(np.arange(0, 256), generate_impulse[0])
            # pl.show()
            generate_impulse = GetData.energyNormalize(generate_impulse)
            # generate_impulse = generate_impulse.reshape((1, -1))
            # pl.close()
            generate_label = GetData.label2Vector(1, 2)
        else:
            continue
        if count == 0:
            impulse = generate_impulse
            impulse_label = generate_label
        else:
            impulse = np.vstack((impulse, generate_impulse))
            impulse_label = np.vstack((impulse_label, generate_label))
        count = count + 1
    np.savetxt('./data_txt/manual_impulse.txt', impulse)
    np.savetxt('./data_txt/manual_impulse_label.txt', impulse_label)
    print('done')
    return impulse, impulse_label


def generate_noise(num, type='flat', num_class=2):
    # 读取noise音频
    print('generating %s noise samples...' % type)
    # noise = GetData.GetNoiseList('./noise')
    noise = GetNoiseList('./noise')
    # # noise = GetData.GetNoiseList('./fordebug/noise')

    # 随机截取num数量的噪声
    noise_single_label = GetData.label2Vector(1, num_class)
    # num = int(data.shape[0] / 2)
    noise_data = np.zeros((1, 256))
    noise_label = np.zeros((1, 2))
    for i in range(num):
        rand_noise = random.randint(0, len(noise) - 1)
        selected_noise = noise[rand_noise]
        if type == 'flat':
            start = random.randint(0, selected_noise.shape[0] - 300)
            noise_clipped = selected_noise[start:start + 256]
            # noise_flat = GetData.energyNormalize(noise_clipped)

            # 留给外部做归一化
            # noise_clipped = GetData.energyNormalize(noise_clipped)

            noise_clipped = noise_clipped.reshape((1, -1))
            if noise_clipped.shape[1] < 256:
                continue
            noise_data = np.vstack((noise_data, noise_clipped))
            noise_label = np.vstack((noise_label, noise_single_label))
        elif type == 'impulse':
            start = random.randint(256, selected_noise.shape[0] - 256)
            noise_clipped = selected_noise[start:start + 30]
            # 生成随机冲击
            idx = np.argmax(noise_clipped)
            idx_in_selected = start + idx
            rand_width = random.randint(0, 1)
            selected_noise[idx_in_selected] = selected_noise[idx_in_selected] * 20
            # print(selected_noise[idx_in_selected])
            if rand_width == 1:
                selected_noise[idx_in_selected - rand_width] = selected_noise[idx_in_selected - rand_width] * 20
                selected_noise[idx_in_selected + rand_width] = selected_noise[idx_in_selected + rand_width] * 20
                # selected_noise[idx_in_selected - j] = selected_noise[idx_in_selected] * j / 4
                # lambda_factor = random.random() - 0.5
                # selected_noise[idx_in_selected + j] = selected_noise[idx_in_selected] * (j + lambda_factor) / 4
                # print('%g %g'% (selected_noise[idx_in_selected - j], selected_noise[idx_in_selected + j]))
            if idx_in_selected % 2 == 0:
                minus_impulse = -1 * selected_noise[idx_in_selected - rand_width:idx_in_selected + rand_width + 1]
                impulse_inter = random.randint(1, 10)
                minus_index = idx_in_selected + rand_width + 1 + impulse_inter
                width = 2 * rand_width + 1
                selected_noise[minus_index:minus_index + width] = minus_impulse
            picked_start_pos = idx_in_selected - random.randint(0, 210)
            noise_impulse = selected_noise[picked_start_pos:picked_start_pos + 256]
            # pl.plot(noise_impulse)
            # pl.show()

            # 留给外部做归一化
            # noise_impulse = GetData.energyNormalize(noise_impulse)

            noise_impulse = noise_impulse.reshape((1, -1))
            if noise_impulse.shape[1] < 256:
                continue
            noise_data = np.vstack((noise_data, noise_impulse))
            noise_label = np.vstack((noise_label, noise_single_label))
    print('done')
    noise_actual = noise_data[1:num + 1]
    noise_actual_label = noise_label[1:num + 1]
    return noise_actual, noise_actual_label


def generate_flat_noise(num, number_class=2):
    if os.path.exists('./data_txt/noise_flat_data_norm.mat'):
        print('loading flat noise...')
        # flat_noise_actual = np.loadtxt('./data_txt/noise_flat_data.txt')
        # flat_actual_label = np.loadtxt('./data_txt/noise_flat_label.txt')
        # sio.savemat('./data_txt/noise_flat_data.mat', {'data': flat_noise_actual, 'label': flat_actual_label})
        noise = sio.loadmat('./data_txt/noise_flat_data_norm.mat')
        flat_noise_actual = noise['data']
        flat_actual_label = noise['label']
        print('done')
        length = len(flat_noise_actual)
        if length < num:
            print('lack enough flat noise sample, preparing to generate.')
            supple, supple_label = generate_noise(num-length, type='flat')
            flat_noise_actual = np.vstack((flat_noise_actual, supple))
            flat_actual_label = np.vstack((flat_actual_label, supple_label))
            # np.savetxt('./data_txt/noise_flat_data.txt', flat_noise_actual)
            # np.savetxt('./data_txt/noise_flat_label.txt', flat_actual_label)
            sio.savemat('./data_txt/noise_flat_data_norm.mat', {'data': flat_noise_actual, 'label': flat_actual_label})
            return flat_noise_actual, flat_actual_label
        else:
            return flat_noise_actual[0:num], flat_actual_label[0:num]
    else:
        flat_noise_actual, flat_actual_label = generate_noise(num, type='flat', num_class=number_class)
        # np.savetxt('./data_txt/noise_flat_data.txt', flat_noise_actual)
        # np.savetxt('./data_txt/noise_flat_label.txt', flat_actual_label)
        sio.savemat('./data_txt/noise_flat_data_norm.mat', {'data': flat_noise_actual, 'label': flat_actual_label})
        return flat_noise_actual, flat_actual_label
    # # 读取noise音频
    # print('generating noise samples...')
    # noise = GetData.GetNoiseList('./noise')
    # # # noise = GetData.GetNoiseList('./fordebug/noise')
    #
    # # 随机截取num数量的噪声
    # noise_single_label = GetData.label2Vector(1, 2)
    # # num = int(data.shape[0] / 2)
    # noise_data = np.zeros((1, 256))
    # noise_label = np.zeros((1, 2))
    # for i in range(num):
    #     rand_noise = random.randint(0, len(noise) - 1)
    #     selected_noise = noise[rand_noise]
    #     start = random.randint(0, selected_noise.shape[0] - 256)
    #     noise_clipped = selected_noise[start:start + 256]
    #     # noise_flat = GetData.energyNormalize(noise_clipped)
    #     noise_clipped = GetData.energyNormalize(noise_clipped)
    #     noise_data = np.vstack((noise_data, noise_clipped))
    #     noise_label = np.vstack((noise_label, noise_single_label))
    # print('done')
    # noise_actual = noise_data[1:num + 1]
    # noise_actual_label = noise_label[1:num + 1]
    # np.savetxt('./data_txt/noise_data.txt', noise_actual)
    # np.savetxt('./data_txt/noise_label.txt', noise_actual_label)
    # return noise_actual, noise_actual_label


def generate_impulse_noise(num, number_class=2):
    if os.path.exists('./data_txt/noise_dirimpulse_data_norm.mat'):
        print('loading impulse noise...')
        # impulse_noise_actual = np.loadtxt('./data_txt/noise_dirimpulse_data.txt')
        # impulse_actual_label = np.loadtxt('./data_txt/noise_dirimpulse_label.txt')
        # sio.savemat('./data_txt/noise_dirimpulse_data.mat', {'data': impulse_noise_actual, 'label': impulse_actual_label})
        noise = sio.loadmat('./data_txt/noise_dirimpulse_data_norm.mat')
        impulse_noise_actual = noise['data']
        impulse_actual_label = noise['label']
        print('done')
        length = len(impulse_noise_actual)
        if length < num:
            print('lack enough impulse noise sample, preparing to generate.')
            supple, supple_label = generate_noise(num-length, type='impulse', num_class=number_class)
            impulse_noise_actual = np.vstack((impulse_noise_actual, supple))
            impulse_actual_label = np.vstack((impulse_actual_label, supple_label))
            # np.savetxt('./data_txt/noise_dirimpulse_data.txt', impulse_noise_actual)
            # np.savetxt('./data_txt/noise_dirimpulse_label.txt', impulse_actual_label)
            sio.savemat('./data_txt/noise_dirimpulse_data_norm.mat',
                        {'data': impulse_noise_actual, 'label': impulse_actual_label})
            return impulse_noise_actual, impulse_actual_label
        else:
            return impulse_noise_actual[0:num], impulse_actual_label[0:num]
    else:
        impulse_noise_actual, impulse_actual_label = generate_noise(num, type='impulse')
        # np.savetxt('./data_txt/noise_dirimpulse_data.txt', impulse_noise_actual)
        # np.savetxt('./data_txt/noise_dirimpulse_label.txt', impulse_actual_label)
        sio.savemat('./data_txt/noise_dirimpulse_data_norm.mat',
                    {'data': impulse_noise_actual, 'label': impulse_actual_label})
        return impulse_noise_actual, impulse_actual_label


def GetNoiseList(path):
    # 读取噪音
    file_list = GetData.getFileName(path)
    noise = []
    for i in file_list:
        new_read = GetData.readWav(i)
        new_read = energy_normlize(new_read.reshape(1, -1))
        noise.append(new_read[0])
    return noise


def energy_normlize(data):
    norm_array = np.linalg.norm(data, ord=2, axis=1, keepdims=True)
    average_norm = norm_array**2 / data.shape[1]
    average_norm = np.sqrt(average_norm)
    return np.true_divide(data, average_norm)


def high_pass_filter(audio, cutoff, fs):
    # pl.plot(audio[0])

    wn = 2 * cutoff / fs
    b, a = signal.butter(8, wn, 'high')
    audio_filted = signal.filtfilt(b, a, audio, axis=1)

    # pl.plot(audio_filted[0])
    # pl.show()
    return audio_filted


def input_reshape(input_x, is_batch):
    x_reshaped = tf.cond(is_batch, lambda: input_batch_reshape(input_x), lambda: input_all_conv_reshape(input_x))
    return x_reshaped


def input_all_conv_reshape(input_x):
    x_reshaped = tf.reshape(input_x, [1, 1, -1, 1])
    return x_reshaped


def input_batch_reshape(input_x):
    x_reshaped = tf.reshape(input_x, [-1, 1, 256, 1])
    return x_reshaped


is_batch = tf.placeholder(dtype=bool, name='is_batch')

# 暂定输入数据维度为128
x = tf.placeholder(tf.float32, [None, None], name='x')
y = tf.placeholder(tf.float32, [None, 2], name='y')
# y = tf.placeholder(tf.float32, [None, 3], name='y')  # 3 class: add impulse

# 对于卷积网络，tensorflow的输入为4维[batch, row, col, channels]
x_signal = input_reshape(x, is_batch)

# 1——卷积层 卷积长度：1*8, 卷积核个数：16 32 40
W_conv1 = weight_variable([1, 9, 1, 16])
b_conv1 = bias_variable([16])

# TODO: 加入BN层
h_conv1 = conv2d(x_signal, W_conv1, padding='VALID')+b_conv1 # 248*1*16
# h_bn1 = tf.nn.relu(batchnorm(h_conv1, is_testing, out_size=16, convolutional=True)) # batch normalization
h_bn1 = tf.nn.relu(h_conv1)  # no batch normalization
h_pool1 = maxpooling_2x1(h_bn1)  # 输出128*1*16 # 124*1*16

# 2——卷积层 卷积长度：1*5， 卷积核个数：8
W_conv2 = weight_variable([1, 5, 16, 8])
b_conv2 = bias_variable([8])

# TODO:加入BN层
h_conv2 = conv2d(h_pool1, W_conv2, padding='VALID')+b_conv2 # 120*1*8
# h_bn2 = tf.nn.relu(batchnorm(h_conv2, is_testing, out_size=8,  convolutional=True))  # batch normalization
h_bn2 = tf.nn.relu(h_conv2)  # no batch normalization
h_pool2 = maxpooling_2x1(h_bn2)  # 输出64*1*8 # 60*1*8

# 3——卷积层 与第2层卷积一致
W_conv3 = weight_variable([1, 5, 8, 8])
b_conv3 = bias_variable([8])
#
# TODO：加入BN层
h_conv3 = conv2d(h_pool2, W_conv3, padding='VALID')+b_conv3 # 56*1*8
# h_bn3 = tf.nn.relu(batchnorm(h_conv3, is_testing, out_size=8, convolutional=True))  # batch normalization
h_bn3 = tf.nn.relu(h_conv3)  # no batch normalization
h_pool3 = maxpooling_2x1(h_bn3) # 输出为32*1*8 # 28*1*8

# 4——全连接层 神经元个数：512
# W_fc4 = weight_variable([28*1*8, 512])
# for 2 conv layer
W_fc4 = weight_variable([60*1*8, 512])
b_fc4 = bias_variable([512])
# h_pool3_flat = tf.reshape(h_pool3, [-1, 32*1*8])
# h_fc4 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc4)+b_fc4)
# TODO:dropout
keep_pro_l4_l5 = tf.placeholder(tf.float32, name='keep_pro_l4_l5')


def fc4(input_x, weight4, bias4, keep_pro):
    # x_flat = tf.reshape(input_x, [-1, 28 * 1 * 8])
    # for 2 conv layer
    x_flat = tf.reshape(input_x, [-1, 60 * 1 * 8])
    h_fc4 = tf.nn.relu(tf.matmul(x_flat, weight4) + bias4)
    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob=keep_pro)
    return h_fc4_drop


def conv4(input_x, weight4, bias4):
    # conv_weight4 = tf.reshape(weight4, [1, 28, 8, 512])
    # for 2 conv layer
    conv_weight4 = tf.reshape(weight4, [1, 60, 8, 512])
    h_conv4 = tf.nn.relu(conv2d(input_x, conv_weight4, padding="VALID") + bias4)
    return h_conv4

# for conv2
h_output4 = tf.cond(is_batch, lambda: fc4(h_pool2, W_fc4, b_fc4, keep_pro_l4_l5), lambda: conv4(h_pool2, W_fc4, b_fc4))
# h_output4 = tf.cond(is_batch, lambda: fc4(h_pool3, W_fc4, b_fc4, keep_pro_l4_l5), lambda: conv4(h_pool3, W_fc4, b_fc4))

# 5——全连接层 神经元个数：512
W_fc5 = weight_variable([512, 512])
b_fc5 = bias_variable([512])
# h_fc5 = tf.nn.relu(tf.matmul(h_fc4_drop, W_fc5)+b_fc5)


def fc5(input_x, weight5, bias5):
    # x_flat = tf.reshape(input_x, [-1, 32 * 1 * 8])
    h_fc5 = tf.nn.relu(tf.matmul(input_x, weight5) + bias5)
    return h_fc5


def conv5(input_x, weight5, bias5):
    conv_weight5 = tf.reshape(weight5, [1, 1, 512, 512])
    h_conv5 = tf.nn.relu(conv2d(input_x, conv_weight5, padding="VALID") + bias5)
    return h_conv5


h_output5 = tf.cond(is_batch, lambda: fc5(h_output4, W_fc5, b_fc5), lambda: conv5(h_output4, W_fc5, b_fc5))

# 6——输出层
W_fc6 = weight_variable([512, 2])
b_fc6 = bias_variable([2])
# W_fc6 = weight_variable([512, 3])  # 3 class
# b_fc6 = bias_variable([3])  # 3 class
# y_net_out = tf.matmul(h_fc5, W_fc6)+b_fc6


def fc6(input_x, weight6, bias6):
    # x_flat = tf.reshape(input_x, [-1, 32 * 1 * 8])
    h_fc6 = tf.nn.relu(tf.matmul(input_x, weight6) + bias6)
    return h_fc6


def conv6(input_x, weight6, bias6):
    conv_weight6 = tf.reshape(weight6, [1, 1, 512, 2])
    h_conv6 = tf.nn.relu(conv2d(input_x, conv_weight6, padding="VALID") + bias6)
    return h_conv6


h_output6 = tf.cond(is_batch, lambda: fc6(h_output5, W_fc6, b_fc6), lambda: conv6(h_output5, W_fc6, b_fc6))

tf.add_to_collection('saved_module', h_output6)
# 定义 softmax regression
# 目标函数
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_output6)
)
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

tf.add_to_collection('saved_module', train_step)

correct_prediction = tf.equal(tf.argmax(h_output6, 1), tf.argmax(y, 1))
# TODO： reduce_mean查看文档
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.add_to_collection('saved_module', accuracy)
saver = tf.train.Saver(max_to_keep=1)

if __name__ == '__main__':

    print('loading data...')
    # data = sio.loadmat('little_shift/smallset_quater_manual.mat')['data']
    data = sio.loadmat('little_shift/supplement_quater_manual.mat')['data']

    data = data[0:75000]
    data = energy_normlize(data)
    click_label = np.loadtxt('shuffle_data_label.txt')
    label = np.tile(click_label[0:7500], (10, 1))
    # label = click_label[0:22000]
    # bottlenose = np.loadtxt('./from_mat/bottlenose.txt')
    # common = np.loadtxt('./from_mat/common.txt')
    # spinner = np.loadtxt('./from_mat/spinner.txt')
    # melon = np.loadtxt('./from_mat/melon.txt')
    # data = np.vstack((bottlenose, common, spinner, melon))
    # data = high_pass_filter(data, cutoff=20000, fs=192000)
    # data = energy_normlize(data)
    # np.savetxt('./from_mat/concatenate_data.txt', data)
    # data = np.loadtxt('./from_mat/concatenate_data.txt')
    print('done')

    print('loading xiamen impulse...')
    xiamen_impulse = sio.loadmat('clean_click/xiamen_impulse_quater1.mat')['data']
    print(xiamen_impulse.shape)
    xiamen_impulse = energy_normlize(xiamen_impulse[0:15000])
    # xiamen_impulse = xiamen_impulse[0:5000]
    print('done!')

    # #######################带信噪比训练样本######################
    # db_list = list(range(9, 17, 2))
    # for i in db_list:
    #     bottlenose_db = np.loadtxt('./from_mat/bottlenose_%ddB.txt' % i)
    #     common_db = np.loadtxt('./from_mat/common_%ddB.txt' % i)
    #     spinner_db = np.loadtxt('./from_mat/spinner_%ddB.txt' % i)
    #     melon_db = np.loadtxt('./from_mat/melon_%ddB.txt' % i)
    #     db_data = np.vstack((bottlenose_db[0:5000], common_db[0:5000], spinner_db[0:5000], melon_db[0:5000]))
    #     db_data = high_pass_filter(db_data, cutoff=20000, fs=192000)
    #     db_data = energy_normlize(db_data)
    #     # pl.plot(db_data[0])
    #     # pl.show()
    #     data = np.vstack((data, db_data))
    # ###############################################################

    ######################## 载入难分类样本 ################
    # print('load hard recog sample from pre-trained model test...')
    # hard_neg_recog = np.loadtxt('./data_txt/hard_neg_recog.txt')
    # hard_neg_label = np.loadtxt('./data_txt/hard_neg_label.txt')
    # hard_pos_recog = np.loadtxt('./data_txt/hard_pos_recog.txt')
    # hard_pos_label = np.loadtxt('./data_txt/hard_pos_label.txt')
    # print('done')
    #
    # # data = np.vstack((data, hard_pos_recog))
    # # label = np.vstack((label, hard_pos_label))
    # manual_hard = np.loadtxt('./data_txt/hard_recog.txt')
    # manual_hard_label = np.loadtxt('./data_txt/hard_label.txt')
    #
    # num_pos_sample = data.shape[0]
    # delta_pos_neg = num_pos_sample - hard_neg_recog.shape[0] - manual_hard.shape[0]
    # print('the number of hard recog positive sample %g' % hard_pos_recog.shape[0])
    # print('the number of hard recog negetive sample %g' % hard_neg_recog.shape[0])
    # print('the number of hard recog manual sample %g' % manual_hard.shape[0])
    # data = np.vstack((data, hard_neg_recog, manual_hard))
    # label = np.vstack((label, hard_neg_label, manual_hard_label))


    ########################### 放大突刺样本 ############################
    # num = int(data.shape[0] / 3)
    # num = data.shape[0]
    # num = 100
    # num = int(delta_pos_neg/3)
    # 生成或导入噪声样本
    # manual_impulse, manual_impulse_label = generate_impulse(data_template_path='./fordebug', add_noise_path='./noise',
    #                                                         num=num)
    flat_num = 25000
    flat_noise, flat_noise_label = generate_flat_noise(flat_num)
    # flat_noise = flat_noise / 2 ** 16
    flat_noise = high_pass_filter(flat_noise, cutoff=4800, fs=192000)
    flat_noise = energy_normlize(flat_noise)

    print('loading flat noise...')
    flat_noise_from_mat = sio.loadmat('clean_click/flat_noise_quater.mat')['data']
    flat_noise_from_mat = energy_normlize(flat_noise_from_mat[0:20000])
    # flat_noise_from_mat = flat_noise_from_mat[0:6000]
    flat_noise_from_mat_label = flat_noise_label[0:20000]
    print('done!')
    # print(flat_noise[0])
    # pl.plot(flat_noise[0])
    # pl.show()

    impulse_num = 15000
    impluse_noise, impulse_noise_label = generate_impulse_noise(impulse_num)
    # impluse_noise = impluse_noise / 2**16
    xiamen_impulse_label = impulse_noise_label
    impluse_noise = high_pass_filter(impluse_noise, cutoff=4800, fs=192000)
    impluse_noise = energy_normlize(impluse_noise)

    # print(impluse_noise[0])
    # pl.plot(impluse_noise[0])
    # pl.show()

    # data = np.vstack((data, flat_noise))
    # label = np.vstack((label, flat_noise_label))

    data = np.vstack((data, flat_noise, flat_noise_from_mat, impluse_noise, xiamen_impulse))
    label = np.vstack((label, flat_noise_label, flat_noise_from_mat_label, impulse_noise_label, xiamen_impulse_label))
    # data = np.vstack((data, flat_noise, impluse_noise, manual_impulse))
    # label = np.vstack((label, flat_noise_label, impulse_noise_label, manual_impulse_label))
    ####################################################################

    (shuffled_data, shuffled_label) = GetData.ShuffleData(data, label)
    # 划分训练集与测试集
    length = shuffled_data.shape[0]
    point_partition = int(length/5)
    # 训练集：测试集：1-test_ratio : test_ratio
    test_ratio = 1
    seg_point = point_partition*test_ratio
    # test_data = tf.convert_to_tensor(shuffled_data[0:point_partition], dtype=tf.float32)
    # test_label = tf.convert_to_tensor(shuffled_label[0:point_partition], dtype=tf.float32)
    # train_data = tf.convert_to_tensor(shuffled_data[point_partition:length], dtype=tf.float32)
    # train_label = tf.convert_to_tensor(shuffled_label[point_partition:length], dtype=tf.float32)

    test_data = shuffled_data[0:point_partition]
    test_label = shuffled_label[0:point_partition]
    train_data = shuffled_data[point_partition:length]
    train_label = shuffled_label[point_partition:length]
    # train_data = shuffled_data
    # train_label = shuffled_label

    print('training begin.')
    with tf.Session() as sess:
        # x1 = tf.Variable(tf.truncated_normal([50, 128], mean=0, stddev=1.0, dtype=tf.float32))
        # ref = tf.Variable(tf.zeros([50, 2]))
        # indices = [0]
        # updates = tf.Variable(tf.ones([50, 1]))
        # y1 = tf.scatter_nd_update(ref, indices, updates)
        # x2 = tf.Variable(tf.truncated_normal([50, 128], mean=1, stddev=1.0, dtype=tf.float32))
        # ref = tf.Variable(tf.zeros([50, 2]))
        # indices = [1]
        # updates = tf.Variable(tf.ones([50, 1]))
        # y2 = tf.scatter_nd_update(ref, indices, updates)
        # x = tf.concat(1, [x1, x2])
        # y = tf.concat(1, [y1, y2])
        sess.run(tf.global_variables_initializer())
        num_train = train_data.shape[0]
        batch_size = 128
        num_batch = int(num_train/batch_size)
        for i in range(301):
            acc = 0
            for index in range(num_batch):
                batch_start = index*batch_size
                batch_end = (index+1)*batch_size
                x_batch = train_data[batch_start:batch_end]
                y_batch = train_label[batch_start:batch_end]
                ts, train_accuracy = sess.run((train_step, accuracy),
                                              feed_dict={x: x_batch, y: y_batch, keep_pro_l4_l5: 0.5, is_batch: True})
                acc = acc + train_accuracy
            # if i % 5 == 0:
            #     train_accuracy = accuracy.eval(feed_dict={x: train_data, y: train_label,
            #                                               keep_pro_l4_l5: 1.0, is_testing: True})
            acc = acc/num_batch
            print('epoch %d, average training accuracy %g' % (i, acc))
            # saver.save(sess, 'ckpt/allconv_cnn4click_small_clean_3conv_little_shift_v2.ckpt', global_step=i)
            saver.save(sess, 'ckpt/allconv_cnn4click_norm_quater_manual_conv2_supplement.ckpt', global_step=i)
            (train_data, train_label) = GetData.ShuffleData(train_data, train_label)
        print('training over.')

        # # 获取第一个卷积层权重
        # weight_conv1 = W_conv1.eval()
        # reshaped_weight = weight_conv1.reshape((9, 16))
        # np.savetxt('layer1_weight_manual_16.txt', np.transpose(reshaped_weight))

        # train_accuracy = accuracy.eval(feed_dict={x: train_data, y: train_label,
        #                                           keep_pro_l4_l5: 1.0, is_testing: True})
        # print('training accuracy %g' % train_accuracy)
        batch_size = 10000
        num_batch = math.ceil(num_train/batch_size)
        for index in range(num_batch):
            batch_start = index * batch_size
            batch_end = (index + 1) * batch_size
            if batch_end > num_train:
                batch_end = num_train
            x_batch = train_data[batch_start:batch_end]
            y_batch = train_label[batch_start:batch_end]
            train_accuracy = accuracy.eval(feed_dict={x: x_batch, y: y_batch,
                                                          keep_pro_l4_l5: 1.0, is_batch: True})
            print('train batch: %g accuracy: %g' % (index, train_accuracy))

        ############### 单样本测试 ####################
        # correct = 0
        # for i in range(num_train):
        #     # iscorrect = correct_prediction.eval(feed_dict={x: train_data[i].reshape(1, 256),
        #     #                                                y: train_label[i].reshape(1, 2),
        #     #                                                keep_pro_l4_l5: 1.0, is_testing: True})
        #     # if iscorrect:
        #     #     correct = correct+1
        #     y_out = sess.run(y_net_out, feed_dict={x: train_data[i].reshape(1, 256), keep_pro_l4_l5: 1.0, is_testing: True})
        #     predict = np.argmax(y_out, axis=1)
        #     if predict == np.argmax(train_label[i].reshape(1, 2), axis=1):
        #         correct = correct + 1
        # acc = correct / num_train
        # print('one by one test: %g' % acc)
        ################################################

        print('test:')
        # print('test accuracy %g' % accuracy.eval(
        #     feed_dict={x: test_data, y: test_label, keep_pro_l4_l5: 1.0, is_testing: True}))
        num_test = test_data.shape[0]
        batch_size = 128
        num_batch = int(num_test / batch_size)
        for index in range(num_batch):
            batch_start = index * batch_size
            batch_end = (index + 1) * batch_size
            if batch_end > num_test:
                batch_end = num_test
            x_batch = train_data[batch_start:batch_end]
            y_batch = train_label[batch_start:batch_end]
            train_accuracy = accuracy.eval(feed_dict={x: x_batch, y: y_batch,
                                                          keep_pro_l4_l5: 1.0, is_batch: True})
            print('test batch: %g accuracy: %g' % (index, train_accuracy))
        end = 0




        # print('impulse test %g' % accuracy.eval(
        #     feed_dict={x: impulse_noise1[num:num*2], y: noise_actual_label, keep_pro_l4_l5: 1.0, is_testing: True}))
        # whole_data = np.vstack((x1_left, x2_left, x3_left, x4_left,
        #                         x1_center, x2_center, x3_center, x4_center,
        #                         x1_right, x2_right, x3_right, x4_right))
        # num_test = int(len(whole_data)/28000)
        # sum_acc = 0
        # for j in range(num_test):
        #     batch_start = j*28000
        #     batch_end = (j + 1) * 28000
        #     test_batch = whole_data[batch_start:batch_end]
        #     test_batch_accuracy = accuracy.eval(
        #         feed_dict={x: test_batch, y: click_label[0:28000], keep_pro_l4_l5: 1.0, is_testing: True})
        #     print('test batch accuracy %g' % test_batch_accuracy)
        #     sum_acc = sum_acc + test_batch_accuracy
        # print('average accuracy %g' % (sum_acc/num_test))