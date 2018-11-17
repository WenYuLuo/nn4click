import tensorflow as tf
import numpy as np
import pylab as pl
from scipy import signal
import GetData
import wave
import random
import math
import scipy.io as sio
import struct
import os
import detection_rate


def read_wav_file(file_path):
    wave_file = wave.open(file_path, 'rb')
    params = wave_file.getparams()
    channels, sampleWidth, frameRate, frames = params[:4]
    data_bytes = wave_file.readframes(frames)  # 读取音频，字符串格式
    wave_file.close()

    wave_data = np.zeros(channels * frames)
    if sampleWidth == 2:
        wave_data = np.fromstring(data_bytes, dtype=np.int16)  # 将字符串转化为int
    elif sampleWidth == 3:
        samples = np.zeros(channels * frames)
        for i in np.arange(samples.size):
            sub_bytes = data_bytes[i * 3:(i * 3 + 3)]
            sub_bytes = bit24_2_32(sub_bytes)
            samples[i] = struct.unpack('i', sub_bytes)[0]
        wave_data = samples
    # wave_data = np.fromstring(data_bytes, dtype=np.short)

    wave_data = wave_data.astype(np.float)
    wave_data = np.reshape(wave_data, [frames, channels])

    return wave_data, frameRate


def bit24_2_32(sub_bytes):
    if sub_bytes[2] < 128:
        return sub_bytes + b'\x00'
    else:
        return sub_bytes + b'\xff'


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
    vector[vector < 10e-10] = 0
    exp_sum = np.sum(vector, axis=1).reshape((-1, 1))
    return np.true_divide(vector, exp_sum)


# def generate_sim_audio(click_num, max_interval, noise_propotion=0.05, verbose=False):
def generate_sim_audio(max_interval, SNR=15, verbose=False):
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

    # file_list = GetData.getFileName('./fordebug')
    # audio = np.zeros((1, 1))
    # click_num = len(file_list)
    # position = []
    # click_energy = 0
    # for i in file_list:
    #     interval = random.randint(1000, max_interval)
    #     interval_blank = np.zeros((1, interval))
    #     audio = np.hstack((audio, interval_blank))
    #     pos_start = audio.shape[1]
    #     pos_end = pos_start + 128
    #     position.append((pos_start, pos_end))
    #     click, fs = readWav(i)
    #     # click = amplitude_norm(click)
    #     # pl.plot(click)
    #     # pl.show()
    #     click = click.reshape((1, -1))
    #     # energy_start = int(input('input the energy start position: '))
    #     # energy_end = int(input('input the energy end position: '))
    #     click_high_pass = high_pass_filter(click, cutoff=20000, fs=192000)
    #     click_energy = click_energy + calcu_energy(click_high_pass[0][10:150])
    #     audio = np.hstack((audio, click))

    # click_energy = 0
    # clicks = sio.loadmat('/media/fish/Elements/click_data/5th_workshop/bottlenose.mat')['click']
    # audio = np.zeros((1, 1))
    # position = []
    # click_num = 7
    # for i in range(click_num):
    #     interval = random.randint(1000, max_interval)
    #     interval_blank = np.zeros((1, interval))
    #     audio = np.hstack((audio, interval_blank))
    #     pos_start = audio.shape[1]
    #     pos_end = pos_start + 128
    #     position.append((pos_start, pos_end))
    #     click = clicks[i]
    #     click = click.reshape((1, -1))
    #     click_high_pass = high_pass_filter(click, cutoff=20000, fs=192000)
    #     click_energy = click_energy + calcu_click_energy(click_high_pass)
    #     audio = np.hstack((audio, click))

        # # generate impulse noise
        # lamda = random.random()+0.5
        # impulse_width = random.randint(2, 5)
        # impluse_max_value = np.amax(audio)
        # # reverse_max_value = -impluse_max_value / 20
        # reverse_max_value = 0
        # half_impulse = lamda * np.linspace(reverse_max_value, impluse_max_value, impulse_width, endpoint=False)
        # another_half = half_impulse[::-1]
        # single_impulse = np.hstack((half_impulse, another_half[1:impulse_width]))
        # if interval % 2 == 0:
        #     impulse_interval = random.randint(5, 12)
        #     double_impulse = np.hstack((single_impulse, np.zeros((1, impulse_interval))[0]))
        #     double_impulse = np.hstack((double_impulse, -1*single_impulse))
        #     generate_impulse = double_impulse
        # else:
        #     generate_impulse = single_impulse
        # generate_impulse = generate_impulse.reshape((1, -1))
        # audio = np.hstack((audio, interval_blank, generate_impulse))

    # length = audio.shape[1]

    noise, fs = readWav('./wav/rewrited_palmyra072006-060803-231815 (5).wav')
    # noise = amplitude_norm(noise)
    length = 80000
    len_rate = length / len(noise)
    if len_rate < 1:
        picked_noise = noise[0:length]
    else:
        tile_num = math.ceil(len_rate)
        tiled_noise = np.tile(noise, tile_num)
        picked_noise = tiled_noise[0:length]

    noise_high_pass = high_pass_filter(picked_noise.reshape((1, -1)), cutoff=5000, fs=192000)
    noise_energy = calcu_energy(noise_high_pass[0])
    # temp = calcu_energy(picked_noise)

    # 随机生成脉冲噪声
    impulse_num = 7
    count = 0
    while True:
        start = random.randint(256, picked_noise.shape[0] - 256)
        noise_clipped = picked_noise[start:start + 30]
        # 生成随机冲击
        idx = np.argmax(noise_clipped)
        idx_in_selected = start + idx
        rand_width = random.randint(0, 1)
        picked_noise[idx_in_selected] = picked_noise[idx_in_selected] * 10
        # print(selected_noise[idx_in_selected])
        if rand_width == 1:
            # 确保放大值均大于0
            picked_noise[idx_in_selected - rand_width] = picked_noise[idx_in_selected - rand_width] * 10
            picked_noise[idx_in_selected + rand_width] = picked_noise[idx_in_selected + rand_width] * 10
            # picked_noise[idx_in_selected - j] = picked_noise[idx_in_selected] * (4 - j) / 4
            # lambda_factor = random.random() - 0.5
            # picked_noise[idx_in_selected + j] = picked_noise[idx_in_selected] * (4 - j + lambda_factor) / 4
            # print('%g %g'% (selected_noise[idx_in_selected - j], selected_noise[idx_in_selected + j]))
        if idx_in_selected % 2 == 0:
            minus_impulse = -1 * picked_noise[idx_in_selected - rand_width:idx_in_selected + rand_width + 1]
            impulse_inter = random.randint(1, 10)
            minus_index = idx_in_selected + rand_width + 1 + impulse_inter
            width = 2 * rand_width + 1
            picked_noise[minus_index:minus_index + width] = minus_impulse
        count = count + 1
        if count > impulse_num:
            break

    # file_list = GetData.getFileName('./fordebug')
    # audio = np.zeros((1, 50000))
    # click_num = len(file_list)
    # position = []
    # # click_energy = 0
    # for i in range(click_num):
    #     pos_start = random.randint(i*7000, (i+1)*7000)
    #     pos_end = pos_start + 128
    #     position.append((pos_start, pos_end))
    #     click, fs = readWav(file_list[i])
    #     # click = amplitude_norm(click)
    #     # pl.plot(click)
    #     # pl.show()
    #     click_size = len(click)
    #     click = click.reshape((1, -1))
    #     # energy_start = int(input('input the energy start position: '))
    #     # energy_end = int(input('input the energy end position: '))
    #     click_high_pass = high_pass_filter(click, cutoff=20000, fs=192000)
    #     click_energy = calcu_energy(click_high_pass[0])
    #     k = math.sqrt((noise_energy * 10 ** (SNR / 10)) / click_energy)
    #     audio[0][pos_start:pos_start+click_size] = k * click[0][0:click_size]
    #     print('SNR: %g, weight: %g' % (SNR, k))

    # clicks = sio.loadmat('/media/fish/Elements/click_data/5th_workshop/bottlenose.mat')['click']
    clicks = sio.loadmat('/media/fish/Elements/click_data/clicks/Mesoplodon_CanaryIsles_Annotated_fine.mat')['clicks']
    clicks = clicks[1140:1150]
    audio = np.zeros((1, 80000))
    click_num = 10
    position = []
    for i in range(click_num):
        pos_start = random.randint(i * 7000, (i + 1) * 7000)
        pos_end = pos_start + 128
        # position.append((pos_start, pos_end))
        click = clicks[i]
        click_size = len(click)
        click = click.reshape((1, -1))
        click_high_pass = high_pass_filter(click, cutoff=5000, fs=192000)
        click_energy = calcu_click_energy(click_high_pass)
        # noise_energy_estimate = calcu_energy(np.hstack((click_high_pass[0][0:200], click_high_pass[0][310:512])))
        noise_energy_estimate = 0
        k = math.sqrt((noise_energy * 10 ** (SNR / 10)) / (click_energy-noise_energy_estimate))
        audio[0][pos_start:pos_start + click_size] = k * click[0][0:click_size]
        position.append((pos_start, pos_start + click_size))
        print('SNR: %g, weight: %g' % (SNR, k))

    # # 按信噪比计算click加权系数
    # average_click_energy = click_energy/click_num
    # noise_high_pass = high_pass_filter(picked_noise.reshape((1, -1)), cutoff=20000, fs=192000)
    # noise_energy = calcu_energy(noise_high_pass[0])
    # k = math.sqrt((noise_energy * 10 ** (SNR / 10)) / average_click_energy)
    # SNR = 10 * math.log(k**2 * (average_click_energy/noise_energy), 10)
    # pl.plot(audio[0])
    # pl.show()
    generated_audio = audio+picked_noise
    # print('SNR: %g, weight: %g' % (SNR, k))
    if verbose:
        pl.subplot(211)
        spec = generated_audio.tolist()
        pl.specgram(spec[0], Fs=96000, scale_by_freq=True, sides='default')
        show_audio = high_pass_filter(generated_audio, cutoff=5000, fs=192000)
        time = np.arange(0, length)
        pl.subplot(212)
        pl.plot(time, show_audio[0])
        # pl.show()
    return generated_audio[0], position


# def calcu_click_energy(x):
#     pow_x = x**2
#     x_norm = np.linalg.norm(x, ord=2)**2
#     start_idx = int(x.shape[1]/2)
#     energy_impulse = 0
#     size = 0
#     for i in range(1, start_idx):
#         energy_impulse = np.sum(pow_x[0][start_idx-i:start_idx+i])
#         if energy_impulse/x_norm > 0.95:
#             size = 2 * i + 1
#             break
#     print('size %g' % size)
#     return energy_impulse/size


def calcu_click_energy(x):
    # x = high_pass_filter(x, )
    pow_x = x**2
    # x_norm = np.linalg.norm(x, ord=2)**2
    start_idx = int(x.shape[1]/2)
    # energy_impulse = 0
    half_size = 50
    size = 2 * half_size
    energy_impulse = np.sum(pow_x[0][start_idx-half_size:start_idx+half_size])
    # print('size %g' % size)
    return energy_impulse/size


def calcu_energy(x):
    # x_norm = np.linalg.norm(x, ord=2)

    energy = np.sum(x**2)
    energy = energy / len(x)
    return energy


def energy_normlize(data):
    norm_array = np.linalg.norm(data, ord=2, axis=1, keepdims=True)
    average_norm = norm_array**2 / data.shape[1]
    average_norm = np.sqrt(average_norm)
    return np.true_divide(data, average_norm)


def high_pass_filter(audio, cutoff, fs):
    # pl.plot(audio[0])

    wn = 2 * cutoff / fs
    b, a = signal.butter(8, wn, 'high')
    audio_filted = signal.filtfilt(b, a, audio)

    # pl.plot(audio_filted[0])
    # pl.show()
    return audio_filted


def local_normalize(audio):
    sum_conv_kernel = np.ones((1, 256))
    audio_pow = audio**2
    sum_energy = signal.convolve(audio_pow.reshape(1, -1), sum_conv_kernel, mode='same')
    sqrt_mean_energy = np.sqrt(sum_energy / 256)
    local_norm = np.true_divide(audio, sqrt_mean_energy)
    return local_norm



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

    # audio, fs = readWav('/media/fish/Elements/click_data/5th_workshop/rewrited_DFR.wav')
    # audio, fs = readWav('./wav/rewrited_12-45A32(2)all_noise.wav')
    audio, fs = read_wav_file('/media/fish/Elements/clickdata/Pilot whale, Globicephala macrorhynchus/Pilot Whales_Bahamas(AUTEC)-Unannotated-NUWC/Set3-A2-092605-H23-0630-0645-1505-1520loc.wav')
    # audio, fs = readWav('/media/fish/Elements/click_data/rewrited_9-13NEWA11(2).wav')
    # audio, fs = readWav('rewrited_real_audio/poor/rewrited_Set3-A7-H69-081606-0400-0430-1017-1047loc_0-2min(2).wav')
    # audio, fs = readWav('192k/未命名文件夹/rewrited_Qx-Dd-SCI0608-Ziph-060819-150528(2.1).wav')
    # audio, fs = readWav('./noise/rewrited_noise_Qx-Dc-SC03-TAT09-060516-171606.wav')
    # audio, fs = readWav('/media/fish/Elements/click_data/annot_xiamen/rewrited_for_annotation_xiamen.wav')
    # audio, fs = read_wav_file('/media/fish/Elements/clickdata/ForCNNLSTM/3rdTraining_Data/Blainvilles_beaked_whale_'
    #                   '(Mesoplodon_densirostris)/Set4-A6-092705-H76-0155-0214-1030-1049loc_0300-0600min.wav')
    # audio = audio[0:400000]
    # audio, fs = readWav('./wav/rewrited_for_annot_9-13NEWA11.wav')
    # audio2 = audio2 / 32768
    # audio1 = sio.loadmat('/media/fish/Elements/click_data/5th_workshop/DFR.mat')
    # audio_filted = audio1['audio'][0]
    if audio.ndim > 1:
        audio = audio[:, 1]
    len_audio = len(audio)
    # audio = audio[int(len_audio/16):int(len_audio/8)]
    # audio = audio[0:350000]
    fl = fs/40
    # fs = 192000
    wn = 2*fl/fs
    b, a = signal.butter(8, wn, 'high')
    # audio = energy_normlize(audio.reshape(1, -1))
    audio_filted = signal.filtfilt(b, a, audio)

    audio_norm = local_normalize(audio_filted)

    audio_norm = audio_norm[0]


    # wn_for_show = [2*110000/fs, 2*150000/fs]
    # B, A = signal.butter(8, wn_for_show, 'bandpass')
    # audio_for_show = signal.filtfilt(B, A, audio)
    time = np.arange(0, audio_filted.shape[0])/fs
    verbose = True
    if verbose:
        pl.subplot(211)
        spec = audio_filted.tolist()
        pl.specgram(spec, Fs=fs, scale_by_freq=True, sides='default')
        pl.subplot(212)
        pl.plot(time, audio_filted)
        # pl.show()
        # pl.plot(time, audio_filted)
        pl.title('high pass filter')
        pl.xlabel('time')
        # pl.show()

    # audio, position = generate_sim_audio(max_interval=10000, SNR=15, verbose=True)
    # sio.savemat('./wav/simulation_v2.mat', {'audio': audio})
    # fl = 4800
    # fs = 192000
    # wn = 2*fl/fs
    # b, a = signal.butter(8, wn, 'high')
    # # audio = energy_normlize(audio.reshape(1, -1))
    # audio_filted = signal.filtfilt(b, a, audio)
    # time = np.arange(0, len(audio_filted))

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        # 导入训练模型
        # saver = tf.train.import_meta_graph('ckpt/cnn4click_shifted.ckpt-199.meta')
        # saver.restore(sess, 'ckpt/cnn4click_shifted.ckpt-199')
        # saver = tf.train.import_meta_graph('ckpt/cnn4click_manual_impulse_flat.ckpt-199.meta')
        # saver.restore(sess, 'ckpt/cnn4click_manual_impulse_flat.ckpt-199')
        # saver = tf.train.import_meta_graph('ckpt/cnn4click_mat_impulse_flat.ckpt-99.meta')
        # saver.restore(sess, 'ckpt/cnn4click_mat_impulse_flat.ckpt-99')
        # saver = tf.train.import_meta_graph('ckpt/cnn4click_mat_flat_impulse.ckpt-99.meta')
        # saver.restore(sess, 'ckpt/cnn4click_mat_flat_impulse.ckpt-99')

        # saver = tf.train.import_meta_graph('ckpt/allconv_cnn4click_small_clean_3conv_little_shift.ckpt-199.meta')
        # saver.restore(sess, 'ckpt/allconv_cnn4click_small_clean_3conv_little_shift.ckpt-199')

        saver = tf.train.import_meta_graph('ckpt/allconv_cnn4click_norm_quater_manual.ckpt-300.meta')
        saver.restore(sess, 'ckpt/allconv_cnn4click_norm_quater_manual.ckpt-300')
        # saver = tf.train.import_meta_graph('ckpt/allconv_cnn4click_norm_quater_manual_conv2_16kernal.ckpt-150.meta')
        # saver.restore(sess, 'ckpt/allconv_cnn4click_norm_quater_manual_conv2_16kernal.ckpt-150')

        graph = tf.get_default_graph()

        # 获取模型参数
        x = graph.get_operation_by_name('x').outputs[0]
        y = graph.get_operation_by_name('y').outputs[0]
        is_batch = graph.get_operation_by_name('is_batch').outputs[0]
        keep_pro_l4_l5 = graph.get_operation_by_name('keep_pro_l4_l5').outputs[0]

        collection = graph.get_collection('saved_module')

        y_net_out6 = collection[0]
        train_step = collection[1]
        accuracy = collection[2]

        # noise_actual = np.loadtxt('small_noise_data.txt')
        # noise_actual_label = np.loadtxt('small_noise_label.txt')

        # # 对生成信号测试
        # print('滑动生成测试样本')
        # index = 0
        # input_length = 256
        # detected = []
        # audio_clipped = audio_filted
        # num_detected = 0
        # detected_visual = np.zeros_like(audio_filted)
        # pos_idx = []
        # while True:
        #     start_point = index
        #     end_point = index + 256
        #     input_signal = audio_clipped[start_point:end_point]
        #     input_signal = GetData.energyNormalize(input_signal)
        #     input_signal = np.reshape(input_signal, (1, 256))
        #     if start_point == 0:
        #         test = input_signal
        #     else:
        #         test = np.vstack((test, input_signal))
        #     pos_idx.append((start_point, end_point))
        #     index = index + int(input_length / 4)
        #     if index > len(audio_clipped)-256:
        #         break
        # print('生成完毕！')
        # test = high_pass_filter(test, cutoff=20000, fs=192000)
        # test = energy_normlize(test)
        # # pl.plot(test[0])
        # # print(test[0])
        # # data = np.loadtxt('./from_mat/concatenate_data.txt')
        # # pl.plot(data[0])
        # # print(data[0])
        # # # pl.show()
        # #
        # # test = np.loadtxt('./data_txt/bottlenose_center.txt')
        # # test_label = np.loadtxt('./data_txt/bottlenose_label.txt')
        # # test = energy_normlize(test)
        # # print(test[0])
        # # pl.plot(test[0])
        # # pl.show()
        # #
        # # test_accuracy = accuracy.eval(feed_dict={x: test, y: test_label, keep_pro_l4_l5: 1.0, is_testing: True})
        # # print('test acc: %g' % test_accuracy)
        #
        # # pl.plot(test[0])
        # # pl.plot(test[1])
        # # pl.show()
        #
        limit_length = 10 * 400000
        data_seg = []
        if len_audio > limit_length:
            seg_num = math.floor(len_audio/limit_length);
            for i in range(seg_num):
                start_seg = limit_length*i
                if limit_length*(i+1) > len_audio:
                    end_seg = len_audio
                else:
                    end_seg = limit_length*(i+1)
                data = audio_norm[start_seg:end_seg]
                data_seg.append(data)
        else:
            data_seg.append(audio_norm)

        print('calculating...')
        # audio_filted = audio[0]
        detected_visual = np.zeros_like(audio_norm)
        # reshape_len = math.floor(len_audio/256) * 256
        # audio_filted = audio_filted[0:256]
        # audio_filted = energy_normlize(audio_filted.reshape(1, -1))
        for i in range(len(data_seg)):
            audio_norm = data_seg[i]
            y_out = sess.run(y_net_out6, feed_dict={x: audio_norm.reshape(1, -1), keep_pro_l4_l5: 1.0, is_batch: False})
            # y_out1 = sess.run(y_net_out6, feed_dict={x: audio_filted.reshape(1, -1), keep_pro_l4_l5: 1.0, is_batch: True})
            col_num = y_out.shape[2]
            y_out = y_out.reshape(col_num, 2)
            # print('temp')
            # test_accuracy, y_out = sess.run([accuracy, y_net_out], feed_dict={x: noise_actual, y: noise_actual_label, keep_pro_l4_l5: 1.0, is_testing: True})
            y_out = softmax(y_out)
            # print(y_out)
            predict = np.argmax(y_out, axis=1)
            start_point = 0
            for j in range(len(predict)):
                pro = y_out[j][predict[j]]
                if predict[j] == 0 and pro > 0.9:  # and pro > 0.9:
                    start_point = limit_length*i + 8*j
                    end_point = start_point + 256
                    detected_visual[start_point:end_point] += 1
                # num_detected = num_detected+1
            # elif predict == 1:
            #     detected_visual[start_point:end_point] -= 10

        # detected_visual = detected_visual*1000
        # # print('the number of detected click: %g' % num_detected)
        #
        # pl.plot(time, detected_visual)
        # # # pl.figure(2)
        # # # pl.stem(predict, '-.')
        # pl.show()


        # # 对生成信号测试
        # # print('滑动生成测试样本')
        # index = 0
        # input_length = 256
        # detected = []
        # audio_clipped = audio_filted
        # num_detected = 0
        # detected_visual = np.zeros_like(audio_filted)
        # pos_idx = []
        # while True:
        #     start_point = index
        #     end_point = index + 256
        #     input_signal = audio_clipped[start_point:end_point]
        #     # input_signal = GetData.energyNormalize(input_signal)
        #     input_signal = np.reshape(input_signal, (1, 256))
        #     # input_signal = high_pass_filter(input_signal, cutoff=5000, fs=fs)
        #     input_signal = energy_normlize(input_signal)
        #     # y_out = sess.run(y_net_out6, feed_dict={x: input_signal, keep_pro_l4_l5: 1.0, is_batch: False})
        #     y_out = sess.run(y_net_out6, feed_dict={x: input_signal, keep_pro_l4_l5: 1.0, is_batch: True})
        #     # y_out = y_out.reshape(1, 2)
        #     # print(y_out)
        #     # print(y_out1)
        #     # test_accuracy, y_out = sess.run([accuracy, y_net_out], feed_dict={x: noise_actual, y: noise_actual_label, keep_pro_l4_l5: 1.0, is_testing: True})
        #     y_out = softmax(y_out)
        #     print(y_out)
        #     predict = np.argmax(y_out, axis=1)
        #     pro = y_out[0][predict]
        #     # for j in range(len(pos_idx)):
        #     if predict == 0 and pro > 0.95:  # and pro > 0.9:
        #         detected_visual[start_point:end_point] += 1
        #         # num_detected = num_detected + 1
        #     index = index + int(input_length / 5)
        #     if index > len(audio_clipped)-256:
        #         break
        # # #
        # # #
        # detected click 定位
        index_detected = np.where(detected_visual >= 1)[0]
        detected_list = []
        is_begin = False
        # if not index_detected:
        #     pass
        # else:
        pos_start = index_detected[0]
        for i in range(len(index_detected)):
            if not is_begin:
                pos_start = index_detected[i]
                is_begin = True
            # 考虑到达list终点时的情况
            if i+1 >= len(index_detected):
                pos_end = index_detected[i]
                detected_list.append((pos_start, pos_end+1))
                break
            if index_detected[i+1] - index_detected[i] > 1:
                pos_end = index_detected[i]
                detected_list.append((pos_start, pos_end+1))
                is_begin = False
            else:
                continue
        # #
        # n = 16
        # top = (256-16*(n-1))*n
        # thresh = 0
        # for i in range(n):
        #     thresh = thresh + i*16
        # thresh = top + thresh
        # print(thresh)
    #
        index_to_remove = []
        for i in range(len(detected_list)):
            detected_pos = detected_list[i]
            detected_length = detected_pos[1]-detected_pos[0]
            if detected_length < 256+8*8:
                detected_visual[detected_pos[0]:detected_pos[1] + 1] = 0
                index_to_remove.append(i)
                continue
            ## snr estimate
            signal = audio_filted[detected_pos[0]:detected_pos[1] + 1]
            detected_clicks_energy = calcu_click_energy(signal.reshape(1, -1))
            noise_estimate1 = audio_filted[detected_pos[0] - 256:detected_pos[0]]
            noise_estimate2 = audio_filted[detected_pos[1] + 1:detected_pos[1] + 257]
            noise_estimate = np.hstack((noise_estimate1, noise_estimate2))
            noise_energy = calcu_energy(noise_estimate)
            snr = 10 * math.log10(detected_clicks_energy / noise_energy)
            if snr < 5:
                detected_visual[detected_pos[0]:detected_pos[1] + 1] = 0
                index_to_remove.append(i)
            # ##
            # length = detected_pos[1]-detected_pos[0]+1
            # confident = np.sum(detected_visual[detected_pos[0]:detected_pos[1]+1])
            # if confident <= length:
            #     detected_visual[detected_pos[0]:detected_pos[1]+1] = 0
            #     index_to_remove.append(i)
        has_removed = 0
        for i in index_to_remove:
            detected_list.pop(i-has_removed)
            has_removed = has_removed+1
        for i in detected_list:
            detected_visual[i[0]:i[1]] = 1



        # # # position = np.loadtxt('/media/fish/Elements/click_data/annot_xiamen/clicks.txt')
        # # # position = position * 400000
        # # # position = position.astype(int)
        # # position = sio.loadmat('clean_click/for_annot_9-13NEWA11_click.mat')['click']
        # # position = sio.loadmat('/media/fish/18959204307/新建文件夹/for_annot_9-13NEWA11_click.mat')['click']
        # position = sio.loadmat('/media/fish/Elements/click_data/annot_xiamen/for_annotation_xiamen_click.mat')['click']
        # position = position.tolist()
        # # position = position.tolist()
        # #
        # # maybe_pos = sio.loadmat('clean_click/for_annot_9-13NEWA11_maybe.mat')['maybe']
        # # maybe_pos = sio.loadmat('/media/fish/18959204307/新建文件夹/for_annot_9-13NEWA11_maybe.mat')['maybe']
        # # maybe_pos = sio.loadmat('/media/fish/Elements/click_data/annot_xiamen/for_annotation_xiamen_maybe.mat')['maybe']
        # # # maybe_pos = maybe_pos.tolist()
        # # maybe_position = np.zeros_like(audio_filted)
        # # for i in maybe_pos:
        # #     maybe_position[i[0]:i[1]] = 6500
        # # position = np.vstack((position, maybe_pos))
        # # position = position.tolist()
        # # pl.plot(time, maybe_position)
        #
        # # detected_list = sio.loadmat('from_mat/clickpos0.7.mat')['ClickPos']
        # # detected_visual = np.zeros_like(audio_filted)
        # # detected_list = detected_list.tolist()
        # # index_to_remove = []
        # # audio_filted = high_pass_filter(audio_filted, cutoff=15000, fs=fs)
        # # for i in range(len(detected_list)):
        # #     detected_pos = detected_list[i]
        # #     ## snr estimate
        # #     signal = audio_filted[detected_pos[0]:detected_pos[1] + 1]
        # #     detected_clicks_energy = calcu_click_energy(signal.reshape(1, -1))
        # #     noise_estimate1 = audio_filted[detected_pos[0] - 256:detected_pos[0]]
        # #     noise_estimate2 = audio_filted[detected_pos[1] + 1:detected_pos[1] + 257]
        # #     noise_estimate = np.hstack((noise_estimate1, noise_estimate2))
        # #     noise_energy = calcu_energy(noise_estimate)
        # #     snr = 10 * math.log10(detected_clicks_energy / noise_energy)
        # #     if snr < 15:
        # #         detected_visual[detected_pos[0]:detected_pos[1] + 1] = 0
        # #         index_to_remove.append(i)
        # #     # ##
        # #     # length = detected_pos[1]-detected_pos[0]+1
        # #     # confident = np.sum(detected_visual[detected_pos[0]:detected_pos[1]+1])
        # #     # if confident <= length:
        # #     #     detected_visual[detected_pos[0]:detected_pos[1]+1] = 0
        # #     #     index_to_remove.append(i)
        # # has_removed = 0
        # # for i in index_to_remove:
        # #     detected_list.pop(i-has_removed)
        # #     has_removed = has_removed+1
        # # for i in detected_list:
        # #     detected_visual[i[0]:i[1]] = 1
        # # #
        # #
        # clicks_position = np.zeros_like(audio_filted)
        # for i in position:
        #     clicks_position[i[0]:i[1]] = 6000
        # pl.plot(time, clicks_position)
        #
        print('the number of detected clicks: %g' % len(detected_list))
        # detected_matcher = detection_rate.marked_pool(position)
        # recall, precision = detected_matcher.calcu_detection_rate(detected_list)
        # print('the recall: %g \nthe precision: %g' % (recall, precision))
        # #
        # #
        detected_visual = detected_visual*20000
        # print('the number of detected click: %g' % num_detected)

        pl.plot(time, detected_visual)
        pl.show()