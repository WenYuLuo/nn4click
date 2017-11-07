import tensorflow as tf
import numpy as np
import pylab as pl
from scipy import signal
import GetData
import wave
import random
import math
import os


def readWav(file_name, verbose=False):
    f = wave.open(file_name)
    nchannels = f.getnchannels()
    sample_width = f.getsampwidth()
    fs = f.getframerate()
    nsample = f.getnframes()

    str_data = f.readframes(nsample)
    f.close()

    wava_data = np.fromstring(str_data, dtype=np.short)
    wava_data.shape = -1, nchannels
    wava_data = wava_data.T

    x = wava_data[0]

    # wava_data = np.zeros((nchannels, int(nsample/nchannels)))
    # for i in range(nsample):
    #     frame = f.readframes(1)
    #     a = frame[0]
    #     b = frame[1]
    #     for j in range(nchannels):
    #         val = frame[j*sample_width:(j+1)*sample_width]
    #         hex2dec = struct.unpack('<H', val)
    #         wava_data[j][i] = hex2dec[0]

    time = np.arange(0, nsample) / fs
    len_time = int(len(time)/nchannels)
    time = time[0:len_time]

    # x = energyNormalize(x)

    if verbose:
        print(len(time))
        print(len(x))
        pl.plot(time, x)
        pl.xlabel('time')
        # pl.show()

    return x, fs


def amplitude_norm(x):
    max_value = np.max(np.absolute(x))
    normalized = x/max_value
    return normalized


def softmax(vector):
    # print(vector)
    vector = np.exp(vector)
    # print(vector)
    vector[np.isinf(vector)] = 1
    exp_sum = np.sum(vector, 1)
    return vector/exp_sum


# def generate_sim_audio(click_num, max_interval, noise_propotion=0.05, verbose=False):
def generate_sim_audio(max_interval, noise_propotion=0.1, verbose=False):
    # x1 = np.loadtxt('small_set.txt')
    # # start_point = 0
    # audio = np.zeros((1, 1))
    # for i in range(click_num):
    #     interval = random.randint(256, max_interval)
    #     interval_blank = np.zeros((1, interval))
    #     audio = np.hstack((audio, interval_blank))
    #     # picked_click = amplitude_norm(x1[i])
    #     picked_click = x1[i]
    #     picked_click = picked_click.reshape((1, 256))
    #     audio = np.hstack((audio, picked_click))
    file_list = GetData.getFileName('./fordebug')
    audio = np.zeros((1, 1))
    position = []
    for i in file_list:
        interval = random.randint(1000, max_interval)
        interval_blank = np.zeros((1, interval))
        audio = np.hstack((audio, interval_blank))
        pos_start = audio.shape[1]
        pos_end = pos_start+128
        position.append((pos_start, pos_end))
        click, fs = readWav(i)
        # click = amplitude_norm(click)
        click = click.reshape((1, -1))
        audio = np.hstack((audio, click))

        # generate impulse noise
        lamda = random.random()+0.5
        impulse_width = random.randint(2, 5)
        impluse_max_value = np.amax(audio)
        # reverse_max_value = -impluse_max_value / 20
        reverse_max_value = 0
        half_impulse = lamda * np.linspace(reverse_max_value, impluse_max_value, impulse_width, endpoint=False)
        another_half = half_impulse[::-1]
        single_impulse = np.hstack((half_impulse, another_half[1:impulse_width]))
        if interval % 2 == 0:
            impulse_interval = random.randint(5, 12)
            double_impulse = np.hstack((single_impulse, np.zeros((1, impulse_interval))[0]))
            double_impulse = np.hstack((double_impulse, -1*single_impulse))
            generate_impulse = double_impulse
        else:
            generate_impulse = single_impulse
        generate_impulse = generate_impulse.reshape((1, -1))
        audio = np.hstack((audio, interval_blank, generate_impulse))

    length = audio.shape[1]
    noise, fs = readWav('./wav/rewrited_palmyra072006-060803-231815 (5).wav')
    # noise = amplitude_norm(noise)
    len_rate = length/len(noise)
    if len_rate < 1:
        picked_noise = noise[0:length]
    else:
        tile_num = math.ceil(len_rate)
        tiled_noise = np.tile(noise, tile_num)
        picked_noise = tiled_noise[0:length]
    generated_audio = (1-noise_propotion)*audio+picked_noise*noise_propotion
    if verbose:
        time = np.arange(0, length)
        pl.plot(time, generated_audio[0])
        # pl.show()
    return generated_audio[0], position


if __name__ == '__main__':
    # # file_list = GetData.getFileName('./wav')
    # # verbose = True
    # # for i in file_list:
    # #     audio1, fs = readWav(i)
    # #     audio = audio1[0:5000]
    # #     fl = 5000
    # #     wn = 2*fl/fs
    # #     b, a = signal.butter(8, wn, 'high')
    # #     audio_filted = signal.filtfilt(b, a, audio)
    # #     time = np.arange(0, audio.shape[0]) / fs
    # #     if verbose:
    # #         pl.plot(time, audio_filted)
    # #         pl.title('high pass filter')
    # #         pl.xlabel('time')
    # m = np.loadtxt('./data_txt/hard_recog.txt')

    # audio1, fs = readWav('./wav/rewrited_AXW_40.wav')
    # audio = audio1
    # fl = 5000
    # wn = 2*fl/fs
    # b, a = signal.butter(8, wn, 'high')
    # audio_filted = signal.filtfilt(b, a, audio)
    # time = np.arange(0, audio.shape[0])
    # verbose = True
    # if verbose:
    #     pl.plot(time, audio_filted)
    #     pl.title('high pass filter')
    #     pl.xlabel('time')
    audio_filted, position = generate_sim_audio(max_interval=10000, noise_propotion=0.1, verbose=True)
    time = np.arange(0, len(audio_filted))

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        # 导入训练模型
        # saver = tf.train.import_meta_graph('ckpt/cnn4click_shifted.ckpt-199.meta')
        # saver.restore(sess, 'ckpt/cnn4click_shifted.ckpt-199')
        saver = tf.train.import_meta_graph('ckpt/cnn4click_impluse_from_noise.ckpt-99.meta')
        saver.restore(sess, 'ckpt/cnn4click_impluse_from_noise.ckpt-99')

        graph = tf.get_default_graph()

        # 获取模型参数
        x = graph.get_operation_by_name('x').outputs[0]
        y = graph.get_operation_by_name('y').outputs[0]
        is_testing = graph.get_operation_by_name('is_testing').outputs[0]
        keep_pro_l4_l5 = graph.get_operation_by_name('keep_pro_l4_l5').outputs[0]

        collection = graph.get_collection('saved_module')

        y_net_out = collection[0]
        train_step = collection[1]
        accuracy = collection[2]

        # noise_actual = np.loadtxt('small_noise_data.txt')
        # noise_actual_label = np.loadtxt('small_noise_label.txt')

        # 对生成信号测试
        index = 0
        input_length = 256
        detected = []
        audio_clipped = audio_filted
        num_detected = 0
        detected_visual = np.zeros_like(audio_filted)
        while True:
            start_point = index
            end_point = index + 256
            input_signal = audio_clipped[start_point:end_point]
            input_signal = GetData.energyNormalize(input_signal)
            input_signal = np.reshape(input_signal, (1, 256))
            y_out = sess.run(y_net_out, feed_dict={x: input_signal, keep_pro_l4_l5: 1.0, is_testing: True})
            # test_accuracy, y_out = sess.run([accuracy, y_net_out], feed_dict={x: noise_actual, y: noise_actual_label, keep_pro_l4_l5: 1.0, is_testing: True})
            y_out = softmax(y_out)
            print(y_out)
            predict = np.argmax(y_out, axis=1)
            pro = y_out[0][predict]
            if predict == 0 and pro > 0.9:
                detected_visual[start_point:end_point] += 10
                num_detected = num_detected+1
            # elif predict == 1:
            #     detected_visual[start_point:end_point] -= 10
            index = index + int(input_length/5)
            if index > len(audio_clipped)-256:
                break

        detected_visual = detected_visual*100
        print('the number of detected click: %g' % num_detected)

        pl.plot(time, detected_visual)
        pl.show()

