import tensorflow as tf
import numpy as np
import pylab as pl
import model_test
import math
import GetData
import random

# 信号生成器
class wav_generator(object):
    def __init__(self, data_size=256):
        self.batch_index = 0
        bottlenose_list = GetData.getFileName('./bottlenose')
        common_list = GetData.getFileName('./common')
        spinner_list = GetData.getFileName('./spinner')
        melon_list = GetData.getFileName('./melon-headed')
        self.file_list = [bottlenose_list, common_list, spinner_list, melon_list]
        self.noise_list = GetData.getFileName('./noise')
        self.data_size = data_size
        self.batch_signal = np.empty(self.data_size)
        self.batch_label = np.empty(1)

    # 生成信号（一个测试信号）
    def generate(self, click_num, verbose=False):
        picked_dolphin_idx = random.randint(0, len(self.file_list)-1)
        picked_noise_idx = random.randint(0, len(self.noise_list)-1)
        noise_propotion = random.randint(3, 6)*0.1
        self.wav, self.pos = self.generate__(picked_dolphin=self.file_list[picked_dolphin_idx],
                                             picked_noise=self.noise_list[picked_noise_idx],
                                             click_num=click_num,
                                             max_interval=10000,
                                             noise_propotion=noise_propotion,
                                             verbose=verbose)
        click_index = np.zeros(len(self.wav))
        for i in self.pos:
            start_pos = i[0]
            half_pos = i[0]+128
            end_pos = i[1]
            click_index[start_pos:half_pos] = 1
            click_index[half_pos:end_pos] = 0.5

        idx = 0
        while True:
            signal = self.wav[idx:idx+self.data_size]
            signal = GetData.energyNormalize(signal)
            crop_index = click_index[idx:idx+self.data_size]
            overlapped_length = np.sum(crop_index)
            if overlapped_length >= int(self.data_size/2):
                label = GetData.label2Vector(0, 2)
            else:
                label = GetData.label2Vector(1, 2)
            if idx == 0:
                batch_signal = signal
                batch_label = label
            else:
                batch_signal = np.vstack((batch_signal, signal))
                batch_label = np.vstack((batch_label, label))
            idx = idx + int(self.data_size/4)
            if idx+self.data_size > len(self.wav):
                break
        self.batch_signal = batch_signal
        self.batch_label = batch_label
        return self.batch_signal, self.batch_label

    # 内部生成信号函数
    def generate__(self, picked_dolphin, picked_noise, click_num, max_interval, noise_propotion=0.6, verbose=False):
        audio = np.zeros((1, 1))
        position = []
        i = 0
        while i < click_num:
            interval = random.randint(256, max_interval)
            interval_blank = np.zeros((1, interval))
            audio = np.hstack((audio, interval_blank))
            pos_start = audio.shape[1]
            pos_end = pos_start + 256
            position.append((pos_start, pos_end))
            picked_click_idx = random.randint(0, len(picked_dolphin)-1)
            click, fs = model_test.readWav(picked_dolphin[picked_click_idx])
            # click = amplitude_norm(click)
            click = click.reshape((1, -1))
            audio = np.hstack((audio, click))

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
            i = i + 1

        length = audio.shape[1]
        noise, fs = model_test.readWav(picked_noise)
        # noise = amplitude_norm(noise)
        len_rate = length / len(noise)
        if len_rate < 1:
            picked_noise = noise[0:length]
        else:
            tile_num = math.ceil(len_rate)
            tiled_noise = np.tile(noise, tile_num)
            picked_noise = tiled_noise[0:length]

            # 随机生成脉冲噪声
        impulse_num = int(click_num/2)
        count = 0
        while True:
            start = random.randint(256, picked_noise.shape[0] - 256)
            noise_clipped = picked_noise[start:start + 30]
            # 生成随机冲击
            idx = np.argmax(noise_clipped)
            idx_in_selected = start + idx
            rand_width = random.randint(1, 4)
            picked_noise[idx_in_selected] = np.abs(picked_noise[idx_in_selected]) * 30
            # print(selected_noise[idx_in_selected])
            for j in range(1, rand_width + 1):
                # 确保放大值均大于0
                # picked_noise[idx_in_selected - j] = np.abs(picked_noise[idx_in_selected - j]) * 4 * (5 - j)
                # picked_noise[idx_in_selected + j] = np.abs(picked_noise[idx_in_selected + j]) * 4 * (5 - j)
                picked_noise[idx_in_selected - j] = picked_noise[idx_in_selected] * j / 4
                lambda_factor = random.random()-0.5
                picked_noise[idx_in_selected + j] = picked_noise[idx_in_selected] * (j + lambda_factor) / 4
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

        generated_audio = (1 - noise_propotion) * audio + picked_noise * noise_propotion
        if verbose:
            # time = np.arange(0, length)
            pl.plot(generated_audio[0])
            # pl.show()
        return generated_audio[0], position


# 错分样本池（保存被错误分类的样本）
class train_data_pool(object):
    def __init__(self, batch_size=128):
        self.supplement_train_data = np.empty((1, 1))
        self.supplement_train_label = np.empty((1, 1))
        self.batch_size = batch_size
        self.batch_index = 0

    # 存储样本及样本标签
    def push(self, data, label):
        if self.supplement_train_data.shape[1]==1:
            self.supplement_train_data = data
            self.supplement_train_label = label
        else:
            self.supplement_train_data = np.vstack((self.supplement_train_data, data))
            self.supplement_train_label = np.vstack((self.supplement_train_label, label))

    # 获取batch
    def next(self):
        batch_start = self.batch_index
        batch_end = self.batch_index + self.batch_size
        if batch_start > self.supplement_train_data.shape[0]:
            self.batch_index = 0
            return None
        elif self.supplement_train_data.shape[0] < batch_end:
            batch_end = self.supplement_train_data.shape[0]
            batch_data = self.supplement_train_data[batch_start:batch_end]
            batch_label = self.supplement_train_label[batch_start:batch_end]
            self.batch_index = self.batch_index + self.batch_size
            return batch_data, batch_label
        else:
            batch_data = self.supplement_train_data[batch_start:batch_end]
            batch_label = self.supplement_train_label[batch_start:batch_end]
            self.batch_index = self.batch_index + self.batch_size
            return batch_data, batch_label

    # 打乱样本池
    def shuffle_data(self):
        indices = np.arange(len(self.supplement_train_data))
        np.random.shuffle(indices)
        self.supplement_train_data = self.supplement_train_data[indices]
        self.supplement_train_label = self.supplement_train_label[indices]

    # 获取batch数量
    def get_num_batch(self):
        num_batch = math.ceil(self.supplement_train_data.shape[0]/self.batch_size)
        return num_batch


# 平衡错分样本数量
def get_balance_data(num, data, label):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    picked_idx = indices[0:num]
    return data[picked_idx], label[picked_idx]


# 计算传入标签中正负样本数量
def count(label):
    true_label = np.argmax(label, axis=1)
    lenth = len(true_label)
    positive_idx = np.where(true_label==0)[0]
    positive_num = len(positive_idx)
    negative_num = lenth - positive_num
    return positive_num, negative_num


if __name__ == '__main__':
    # avoid data imbalance
    click_data = np.loadtxt('./data_txt/train_data.txt')
    click_label = np.loadtxt('shuffle_data_label.txt')
    click_label = np.tile(click_label, (3, 1))


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        # load pre-trained module
        # saver = tf.train.import_meta_graph('ckpt/cnn4click_shifted.ckpt-99.meta')
        # saver.restore(sess, 'ckpt/cnn4click_shifted.ckpt-99')
        saver = tf.train.import_meta_graph('ckpt/cnn4click_manual_impulse_flat_pre.ckpt-199.meta')
        saver.restore(sess, 'ckpt/cnn4click_manual_impulse_flat_pre.ckpt-199')

        graph = tf.get_default_graph()

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
        generator = wav_generator(data_size=256)
        # train_data_storage = train_data_pool(batch_size=128)
        hard_recog_neg = np.zeros((1, 256))
        hard_recog_neg_label = np.zeros((1, 2))
        hard_recog_pos = np.zeros((1, 256))
        hard_recog_pos_label = np.zeros((1, 2))
        for i in range(400):
            data, label = generator.generate(click_num=20, verbose=False)
            net_out, acc = sess.run((y_net_out, accuracy), feed_dict={x: data, y: label, keep_pro_l4_l5: 1, is_testing: True})
            print('generate audio %g, accuracy: %g ' % (i, acc))
            predict = np.argmax(net_out, axis=1)
            truth = np.argmax(label, axis=1)
            positive_idx = np.where((predict != truth) & (truth == 0))
            negetive_idx = np.where((predict != truth) & (truth == 1))
            pos_num = len(positive_idx[0])
            neg_num = len(negetive_idx[0])
            # train_data_storage.push(data=data[idx[0]], label=label[idx[0]])
            # pos_num, neg_num = count(label[idx[0]])
            hard_recog_neg = np.vstack((hard_recog_neg, data[negetive_idx[0]]))
            hard_recog_neg_label = np.vstack((hard_recog_neg_label, label[negetive_idx[0]]))
            hard_recog_pos = np.vstack((hard_recog_pos, data[positive_idx[0]]))
            hard_recog_pos_label = np.vstack((hard_recog_pos_label, label[positive_idx[0]]))
            print('Getting %g positive sample, %g negative num from this epoch.' % (pos_num, neg_num))
            # if neg_num > pos_num:
            #     print('impose balance...')
            #     delta = neg_num-pos_num
            #     balance_data, balance_label = get_balance_data(delta, click_data, click_label)
            #     train_data_storage.push(data=balance_data, label=balance_label)
            # train_data_storage.shuffle_data()
            # step = 0
            # while True:
            #     batch = train_data_storage.next()
            #     if batch is None:
            #         break
            #     train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1],
            #                                                    keep_pro_l4_l5: 0.5, is_testing: True})
            #     print('epoch %g, batch %g, accuracy %g' % (i, step, train_accuracy))
            #     step = step + 1
        np.savetxt('./data_txt/hard_neg_recog.txt', hard_recog_neg[1:])
        np.savetxt('./data_txt/hard_neg_label.txt', hard_recog_neg_label[1:])
        np.savetxt('./data_txt/hard_pos_recog.txt', hard_recog_pos[1:])
        np.savetxt('./data_txt/hard_pos_label.txt', hard_recog_pos_label[1:])

        temp = generator.generate(click_num=20, verbose=True)
        test_wav = generator.wav
        index = 0
        input_length = 256
        detected = []
        audio_clipped = test_wav
        num_detected = 0
        detected_visual = np.zeros_like(test_wav)
        while True:
            start_point = index
            end_point = index + 256
            input_signal = audio_clipped[start_point:end_point]
            input_signal = GetData.energyNormalize(input_signal)
            input_signal = np.reshape(input_signal, (1, 256))
            y_out = sess.run(y_net_out, feed_dict={x: input_signal, keep_pro_l4_l5: 1.0, is_testing: True})
            # test_accuracy, y_out = sess.run([accuracy, y_net_out], feed_dict={x: noise_actual, y: noise_actual_label, keep_pro_l4_l5: 1.0, is_testing: True})
            y_out = model_test.softmax(y_out)
            print(y_out)
            predict = np.argmax(y_out, axis=1)
            pro = y_out[0][predict]
            if predict == 0:
                detected_visual[start_point:end_point] += 10
                num_detected = num_detected + 1
            # elif predict == 1:
            #     detected_visual[start_point:end_point] -= 10
            index = index + int(input_length / 2)
            if index > len(audio_clipped) - 256:
                break

        detected_visual = detected_visual * 100
        print('the number of detected click: %g' % num_detected)

        pl.plot(detected_visual)
        pl.show()