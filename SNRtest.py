import tensorflow as tf
import numpy as np
import pylab as pl
from scipy import signal
import GetData
import wave
import random
import math
import scipy.io as sio
import os
import detection_rate


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
def generate_sim_audio(click_name, fs, num_click=10, max_interval=6000, SNR=15, verbose=False):
    clicks_mat = sio.loadmat(click_name)
    if click_name == '/home/fish/ROBB/ClickProject/extracted_click/PacWhite.mat':
        clicks = clicks_mat['clicks']
    else:
        clicks = clicks_mat['clicks_15db']
    # fs = clicks_mat['fs']
    if fs == 96000:
        noise_path = '96k'
    elif fs == 192000:
        noise_path = '192k'
    noise_list = GetData.getFileName(noise_path)
    piced_noise_idx = random.randint(0, len(noise_list)-1)
    noise, n = readWav(noise_list[piced_noise_idx])

    # 计算噪声功率
    noise = noise/(2**16)
    noise_high_pass = high_pass_filter(noise.reshape((1, -1)), cutoff=fs/40, fs=fs)
    noise_energy = calcu_energy(noise_high_pass[0])
    if noise_energy<0:
        raise ValueError

    # 确定合成click数量
    len_clicks = len(clicks)
    clicks = GetData.ShuffleData(clicks)[0]
    if num_click < len_clicks:
        clicks_required = num_click
    else:
        clicks_required = len_clicks

    # TODO：混入impulse噪声
    xiamen_impulse = sio.loadmat('/home/fish/ROBB/ClickProject/extracted_click/12-16A43_not.mat')['not_data']
    xiamen_impulse = GetData.ShuffleData(xiamen_impulse)[0]

    # 按信噪比加入click
    impulse_num = 300
    for_shuffle_list = [i for i in range(clicks_required + impulse_num)]
    random.shuffle(for_shuffle_list)
    impulse_noise_index = for_shuffle_list[0:impulse_num]
    audio = np.zeros((1, 1))
    start_point = 1
    position = []
    has_added_impulse = 0
    for i in range(clicks_required+len(impulse_noise_index)):
        interval = random.randint(5000, max_interval)
        interval_blank = np.zeros((1, interval))
        start_point = start_point + interval
        audio = np.hstack((audio, interval_blank))
        if i not in impulse_noise_index:
            # picked_click = amplitude_norm(x1[i])
            click = clicks[i-has_added_impulse]
            click_size = len(click)
            click = click.reshape((1, -1))
            # click主要集中在中点附近
            click_high_pass = high_pass_filter(click, cutoff=fs/40, fs=fs)
            click_energy = calcu_click_energy(click_high_pass)
            noise_energy_estimate = calcu_energy(np.hstack((click_high_pass[0][0:200], click_high_pass[0][310:512])))
            # noise_energy_estimate = 0
            # k = math.sqrt((noise_energy * 10 ** (SNR / 10)) / (click_energy - noise_energy_estimate))
            k = math.sqrt((noise_energy * 10 ** (SNR / 10)) / click_energy )
            picked_click = k * click[0][128:384]
            picked_click = picked_click.reshape((1, -1))
            start_point = audio.shape[1]
            audio = np.hstack((audio, picked_click))
            end_point = audio.shape[1]-1
            position.append((start_point, end_point))
        else:
            has_added_impulse = has_added_impulse + 1
            picked = random.randint(0, len(xiamen_impulse)-1)
            impulse = xiamen_impulse[picked]
            impulse = impulse.reshape((1, -1))
            # click主要集中在中点附近
            impulse_high_pass = high_pass_filter(impulse, cutoff=fs / 40, fs=fs)
            impulse_energy = calcu_click_energy(impulse_high_pass)
            noise_energy_estimate = calcu_energy(np.hstack((impulse_high_pass[0][0:200], impulse_high_pass[0][310:512])))
            # noise_energy_estimate = 0
            k = math.sqrt((noise_energy * 10 ** (SNR / 10)) / (impulse_energy - noise_energy_estimate))
            picked_impulse = k * impulse[0][128:384]
            picked_impulse = picked_impulse.reshape((1, -1))
            audio = np.hstack((audio, picked_impulse))

    audio = np.hstack((audio, np.zeros((1, 200))))
    # 计算噪声所需长度
    len_audio = audio.shape[1]
    len_rate = len_audio / len(noise)
    if len_rate < 1:
        picked_noise = noise[0:len_audio]
    else:
        tile_num = math.ceil(len_rate)
        average = (noise[0]+noise[-1])/2
        # average = average.reshape(1, -1)
        for count in range(tile_num):
            noise = np.hstack((noise, average, noise))
        tiled_noise = noise
        # tiled_noise = np.tile(noise, tile_num)
        picked_noise = tiled_noise[0:len_audio]

    # 添加噪声
    # pl.plot(audio[0])
    # pl.show()
    generated_audio = audio + picked_noise.reshape(1, -1)
    # print("add %d impulse" %has_added_impulse)
    if verbose:
        pl.subplot(211)
        spec = generated_audio.tolist()
        pl.specgram(spec[0], Fs=fs, scale_by_freq=True, sides='default')
        show_audio = high_pass_filter(generated_audio, cutoff=fs/40, fs=fs)
        time = np.arange(0, len_audio)
        pl.subplot(212)
        pl.plot(time, show_audio[0])
        # pl.show()
    return generated_audio, position

    # # 随机生成脉冲噪声
    # impulse_num = 20
    # count = 0
    # while True:
    #     start = random.randint(256, picked_noise.shape[0] - 256)
    #     noise_clipped = picked_noise[start:start + 30]
    #     # 生成随机冲击
    #     idx = np.argmax(noise_clipped)
    #     idx_in_selected = start + idx
    #     rand_width = random.randint(0, 1)
    #     picked_noise[idx_in_selected] = picked_noise[idx_in_selected] * 20
    #     # print(selected_noise[idx_in_selected])
    #     if rand_width == 1:
    #         # 确保放大值均大于0
    #         picked_noise[idx_in_selected - rand_width] = picked_noise[idx_in_selected - rand_width] * 20
    #         picked_noise[idx_in_selected + rand_width] = picked_noise[idx_in_selected + rand_width] * 20
    #         # picked_noise[idx_in_selected - j] = picked_noise[idx_in_selected] * (4 - j) / 4
    #         # lambda_factor = random.random() - 0.5
    #         # picked_noise[idx_in_selected + j] = picked_noise[idx_in_selected] * (4 - j + lambda_factor) / 4
    #         # print('%g %g'% (selected_noise[idx_in_selected - j], selected_noise[idx_in_selected + j]))
    #     if idx_in_selected % 2 == 0:
    #         minus_impulse = -1 * picked_noise[idx_in_selected - rand_width:idx_in_selected + rand_width + 1]
    #         impulse_inter = random.randint(1, 10)
    #         minus_index = idx_in_selected + rand_width + 1 + impulse_inter
    #         width = 2 * rand_width + 1
    #         picked_noise[minus_index:minus_index + width] = minus_impulse
    #     count = count + 1
    #     if count > impulse_num:
    #         break


def calcu_click_energy(x):
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
    audio_filted = signal.filtfilt(b, a, audio, axis=1)

    # pl.plot(audio_filted[0])
    # pl.show()
    return audio_filted


def local_normalize(audio):
    sum_conv_kernel = np.ones((1, 256))
    audio_pow = audio**2
    sum_energy = signal.convolve(audio_pow.reshape(1,-1), sum_conv_kernel, mode='same')
    sqrt_mean_energy = np.sqrt(sum_energy / 256)
    local_norm = np.true_divide(audio, sqrt_mean_energy)
    return local_norm


if __name__ == '__main__':
    # 合成数据实验
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        # 导入训练模型

        saver = tf.train.import_meta_graph('ckpt/allconv_cnn4click_norm_quater_manual_conv2_supplement.ckpt-300.meta')
        saver.restore(sess, 'ckpt/allconv_cnn4click_norm_quater_manual_conv2_supplement.ckpt-300')


        graph = tf.get_default_graph()

        # 获取模型参数
        x = graph.get_operation_by_name('x').outputs[0]
        y = graph.get_operation_by_name('y').outputs[0]
        is_batch = graph.get_operation_by_name('is_batch').outputs[0]
        keep_pro_l4_l5 = graph.get_operation_by_name('keep_pro_l4_l5').outputs[0]

        collection = graph.get_collection('saved_module')

        # click_path = '/media/fish/Elements/click_data/clicks/Mesoplodon_CanaryIsles_Annotated_fine.mat'
        class_list = ['PacWhite', 'Rissos_(Grampus_grisieus)_15db',  'Blainvilles_beaked_whale_(Mesoplodon_densirostris)_15db', 'Pilot_whale_(Globicephala_macrorhynchus)_15db']
        snr_list = [3, 5, 7, 9, 11, 13, 15]
        if not os.path.exists('/home/fish/ROBB/ClickProject/updated_synthetic_data_supplement'):
            os.mkdir('/home/fish/ROBB/ClickProject/updated_synthetic_data_supplement')

        for class_name in class_list:
            print("%s:" %class_name)
            for snr_selection in snr_list:
                print("-----%d db-----" %snr_selection)
                # class_name = 'Rissos_(Grampus_grisieus)_15db'
                # class_name = 'PacWhite'
                # click_path = '/media/fish/18959204307/extracted_click/' + class_name +'.mat'
                click_path = '/home/fish/ROBB/ClickProject/extracted_click/' + class_name + '.mat'
                # snr_selection = 15
                # fs = sio.loadmat('/media/fish/Elements/click_data/clicks/PacWhitesidedDolphin_fs.mat')['fs']
                fs = 96000
                if not os.path.exists('/home/fish/ROBB/ClickProject/updated_synthetic_data_supplement/'+class_name):
                    os.mkdir('/home/fish/ROBB/ClickProject/updated_synthetic_data_supplement/'+class_name)
                recall_list = []
                precision_list = []
                # print('fs=%g' % fs)
                round_num = 10

                # os.environ['CUDA_VISIBLE_DEVICES'] = '0'


                y_net_out6 = collection[0]
                train_step = collection[1]
                accuracy = collection[2]
                # snr_selection = 15
                for num in range(round_num):
                    generated_audio, position = generate_sim_audio(click_name=click_path, fs=fs, num_click=100,
                                                                max_interval=6000, SNR=snr_selection, verbose=False)
                    audio_filted = high_pass_filter(generated_audio, cutoff=fs/40, fs=fs)[0]

                    audio_norm = local_normalize(audio_filted)

                    audio_norm = audio_norm[0]

                    # temp = np.zeros_like(audio_filted)
                    # for pos in position:
                    #     temp[pos[0]:pos[1]] = 5000
                    # pl.plot(audio_filted)
                    # pl.plot(temp)
                    # pl.show()

                    sio.savemat('/home/fish/ROBB/ClickProject/updated_synthetic_data_supplement/'+class_name+'/simulation_'+str(snr_selection)
                                +'_'+str(num)+'.mat', {'audio': generated_audio})
                    sio.savemat('/home/fish/ROBB/ClickProject/updated_synthetic_data_supplement/'+class_name+'/simulation_'+str(snr_selection)
                                +'_'+str(num)+'_position.mat', {'position': position})

                    time = np.arange(0, audio_filted.shape[0])

                    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    # config = tf.ConfigProto()
                    # config.gpu_options.allow_growth = True
                    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
                    # with tf.Session(config=config) as sess:
                    #     # 导入训练模型
                    #     # saver = tf.train.import_meta_graph('ckpt/cnn4click_shifted.ckpt-199.meta')
                    #     # saver.restore(sess, 'ckpt/cnn4click_shifted.ckpt-199')
                    #     # saver = tf.train.import_meta_graph('ckpt/cnn4click_manual_impulse_flat.ckpt-199.meta')
                    #     # saver.restore(sess, 'ckpt/cnn4click_manual_impulse_flat.ckpt-199')
                    #     # saver = tf.train.import_meta_graph('ckpt/cnn4click_mat_impulse_flat.ckpt-99.meta')
                    #     # saver.restore(sess, 'ckpt/cnn4click_mat_impulse_flat.ckpt-99')
                    #     # saver = tf.train.import_meta_graph('ckpt/cnn4click_mat_flat_impulse.ckpt-99.meta')
                    #     # saver.restore(sess, 'ckpt/cnn4click_mat_flat_impulse.ckpt-99')
                    #
                    #     # saver = tf.train.import_meta_graph('ckpt/allconv_cnn4click_small_clean_3conv_little_shift.ckpt-199.meta')
                    #     # saver.restore(sess, 'ckpt/allconv_cnn4click_small_clean_3conv_little_shift.ckpt-199')
                    #
                    #     saver = tf.train.import_meta_graph('ckpt/allconv_cnn4click_norm_quater_manual.ckpt-300.meta')
                    #     saver.restore(sess, 'ckpt/allconv_cnn4click_norm_quater_manual.ckpt-300')
                    #
                    #     graph = tf.get_default_graph()
                    #
                    #     # 获取模型参数
                    #     x = graph.get_operation_by_name('x').outputs[0]
                    #     y = graph.get_operation_by_name('y').outputs[0]
                    #     is_batch = graph.get_operation_by_name('is_batch').outputs[0]
                    #     keep_pro_l4_l5 = graph.get_operation_by_name('keep_pro_l4_l5').outputs[0]
                    #
                    #     collection = graph.get_collection('saved_module')
                    #
                    #     y_net_out6 = collection[0]
                    #     train_step = collection[1]
                    #     accuracy = collection[2]

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

                    # print('calculating...')
                    # audio_filted = audio[0]
                    detected_visual = np.zeros_like(audio_norm)
                    # reshape_len = math.floor(len_audio/256) * 256
                    # audio_filted = audio_filted[0:256]
                    # audio_filted = energy_normlize(audio_filted.reshape(1, -1))
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
                            start_point = 8*j
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
                    #     # input_signal = high_pass_filter(input_signal, cutoff=5000, fs=192000)
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
                    #     if predict == 0:  # and pro > 0.9:
                    #         detected_visual[start_point:end_point] += 1
                    #         # num_detected = num_detected + 1
                    #     index = index + int(input_length / 8)
                    #     if index > len(audio_clipped)-256:
                    #         break


                    # detected click 定位
                    index_detected = np.where(detected_visual >= 1)[0]
                    detected_list = []
                    is_begin = False
                    pos_start = index_detected[0]
                    for i in range(len(index_detected)):
                        if not is_begin:
                            pos_start = index_detected[i]
                            is_begin = True
                        # 考虑到达list终点时的情况
                        if i+1 == len(index_detected):
                            pos_end = index_detected[i]
                            detected_list.append((pos_start, pos_end))
                            break
                        if index_detected[i+1] - index_detected[i] > 1:
                            pos_end = index_detected[i]
                            detected_list.append((pos_start, pos_end))
                            is_begin = False
                        else:
                            continue

                    # index_to_remove = []
                    # for i in range(len(detected_list)):
                    #     detected_pos = detected_list[i]
                    #     length = detected_pos[1]-detected_pos[0]+1
                    #     confident = np.sum(detected_visual[detected_pos[0]:detected_pos[1]+1])
                    #     if confident <= length:
                    #         detected_visual[detected_pos[0]:detected_pos[1]+1] = 0
                    #         index_to_remove.append(i)
                    # has_removed = 0
                    # for i in index_to_remove:
                    #     detected_list.pop(i-has_removed)
                    #     has_removed = has_removed+1

                    index_to_remove = []
                    for i in range(len(detected_list)):
                        detected_pos = detected_list[i]
                        ## snr estimate
                        signal_clipped = audio_filted[detected_pos[0]:detected_pos[1] + 1]
                        detected_clicks_energy = calcu_click_energy(signal_clipped.reshape(1, -1))
                        noise_estimate1 = audio_filted[detected_pos[0] - 256:detected_pos[0]]
                        noise_estimate2 = audio_filted[detected_pos[1] + 1:detected_pos[1] + 257]
                        noise_estimate = np.hstack((noise_estimate1, noise_estimate2))
                        noise_energy = calcu_energy(noise_estimate)
                        snr = 10 * math.log10(detected_clicks_energy / noise_energy)
                        if snr < 3:
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

                    # print('the number of clicks: %g' % len(position))
                    # print('the number of detected clicks: %g' % len(detected_list))
                    detected_matcher = detection_rate.marked_pool(position)
                    recall, precision = detected_matcher.calcu_detection_rate(detected_list)
                    # print('the recall: %g \nthe precision: %g' % (recall, precision))
                    recall_list.append(recall)
                    precision_list.append(precision)
                    # detected_visual = detected_visual * 1000
                    # # # print('the number of detected click: %g' % num_detected)
                    # #
                    # pl.plot(time, temp)
                    # pl.plot(time, detected_visual)
                    # pl.show()
                recall = sum(recall_list) / round_num
                precision = sum(precision_list) / round_num
                print('the mean recall: %g \nthe mean precision: %g' % (recall, precision))
