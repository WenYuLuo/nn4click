import os
import wave
import random
# import struct
import numpy as np
import pylab as pl


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

    # 取第一声道信号
    x = wava_data[0]

    # 逐帧取信号
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

    # x = energyNormalize(x) # 归一化留给外部函数处理

    if verbose:
        print(len(time))
        print(len(x))
        pl.plot(time, x)
        pl.xlabel('time')
        pl.show()

    return x


def getFileName(path):
    f_list = os.listdir(path)
    file_list = []
    for i in f_list:
        if os.path.splitext(i)[1] == '.wav':
            full_name = path + '/' + i
            file_list.append(full_name)
    return file_list


def energyNormalize(x):
    norm = np.linalg.norm(x, ord=2, keepdims=False)
    mean = norm**2 / x.shape[0]
    return x/mean


def label2Vector(x, num_class):
    label = np.zeros([1, num_class])
    label[-1, x] = 1
    return label


def SizeEnergyNormalize(input_data, size_t):
    npoints = len(input_data)
    if npoints < size_t:
        # print(input_data.shape)
        delta = size_t - npoints
        appended_array = np.zeros((1, delta))
        # print(appended_array[0].shape)
        clipped = np.hstack((input_data, appended_array[0]))
    else:
        clipped = input_data[0:size_t]
    clipped = energyNormalize(clipped)
    return clipped


def GetClickData(path):
    # 获取click数据, 并向左右偏移
    file_list = getFileName(path)
    # x = readWav(file_list[0])
    # x = SizeEnergyNormalize(x, 256)
    # label = label2Vector(0, 2)
    for i in file_list:
        #readWav('E:/DailyResearch/clicksclassification/5th_workshop/clicks_palmyra092007FS192-070925-224949_00001.wav')
        new_read = readWav(i)
        clipped_left = SizeEnergyNormalize(new_read, 256)
        # pl.plot(clipped_left)
        # roll to center
        clipped_center = np.roll(clipped_left, random.randint(32, 128))
        # pl.plot(clipped_center)
        # roll to right
        clipped_right = np.roll(clipped_center, random.randint(32, 64))
        # pl.plot(clipped_right)
        # pl.show()
        new_label = label2Vector(0, 2)
        label_3class = label2Vector(0, 3)
        if i == file_list[0]:
            x_left = clipped_left
            x_center = clipped_center
            x_right = clipped_right
            label = new_label
            labelfor3 = label_3class
        else:
            x_left = np.vstack((x_left, clipped_left))
            x_center = np.vstack((x_center, clipped_center))
            x_right = np.vstack((x_right, clipped_right))
            label = np.vstack((label, new_label))
            labelfor3 = np.vstack((labelfor3, label_3class))
    return x_left, x_center, x_right, label, labelfor3


def GetNoiseList(path):
    # 读取噪音
    file_list = getFileName(path)
    noise = []
    for i in file_list:
        new_read = readWav(i)
        noise.append(new_read)
    return noise


def ShuffleData(x, label=None):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    shuffled_data = x[indices]
    if label is None:
        shuffled_label = None
    else:
        shuffled_label = label[indices]
    return shuffled_data, shuffled_label



if __name__ == '__main__':
    print('reading spinner\'s clicks...')
    (x3_left, x3_center, x3_right, label3, label3_3) = GetClickData('./spinner')
    print('done')
    print('saving spinner data...')
    np.savetxt('./data_txt/spinner_left.txt', x3_left)
    np.savetxt('./data_txt/spinner_center.txt', x3_center)
    np.savetxt('./data_txt/spinner_right.txt', x3_right)
    np.savetxt('./data_txt/spinner_label.txt', label3)
    np.savetxt('./data_txt/spinner_label3.txt', label3_3)
    print('done')
    print('reading bottlenose\'s clicks...')
    (x1_left, x1_center, x1_right, label1, label1_3) = GetClickData('./bottlenose')
    print('done')
    print('saving bottlenose data...')
    np.savetxt('./data_txt/bottlenose_left.txt', x1_left)
    np.savetxt('./data_txt/bottlenose_center.txt', x1_center)
    np.savetxt('./data_txt/bottlenose_right.txt', x1_right)
    np.savetxt('./data_txt/bottlenose_label.txt', label1)
    np.savetxt('./data_txt/bottlenose_label3.txt', label1_3)
    print('done')
    print('reading common\'s clicks...')
    (x2_left, x2_center, x2_right, label2, label2_3) = GetClickData('./common')
    print('done')
    print('saving common data...')
    np.savetxt('./data_txt/common_left.txt', x2_left)
    np.savetxt('./data_txt/common_center.txt', x2_center)
    np.savetxt('./data_txt/common_right.txt', x2_right)
    np.savetxt('./data_txt/common_label.txt', label2)
    np.savetxt('./data_txt/common_label3.txt', label2_3)
    print('done')
    print('reading melon-headed\'s clicks...')
    (x4_left, x4_center, x4_right, label4, label4_3) = GetClickData('./melon-headed')
    print('done')
    print('saving melon-headed data...')
    np.savetxt('./data_txt/melon_left.txt', x4_left)
    np.savetxt('./data_txt/melon_center.txt', x4_center)
    np.savetxt('./data_txt/melon_right.txt', x4_right)
    np.savetxt('./data_txt/melon_label.txt', label4)
    np.savetxt('./data_txt/melon_label3.txt', label4_3)
    print('done')








