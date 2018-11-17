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
import model_test


def main():
    path = '/home/fish/ROBB/ClickProject/updated_synthetic_data'
    dirs = os.listdir(path)
    for species in dirs:
        file_path = os.path.join(path, species)
        if not os.path.isdir(file_path):
            continue
        audio_list = os.listdir(file_path)
        species_recall = [[], [], [], [], [], [], []]
        species_precision = [[], [], [], [], [], [], []]
        for idx in audio_list:
            audio_name = os.path.splitext(idx)[0]
            if os.path.splitext(idx)[-1] != '.mat':
                continue
            snr_click = idx.split('_')[1]
            # if int(snr_click) < 11:
            #     continue
            db_indx = int((int(snr_click) + 1) / 2 - 2)
            audio_path = os.path.join(file_path, idx)
            audio = sio.loadmat(audio_path)['audio'][0]
            fs = 96000
            len_audio = len(audio)
            fl = fs / 40
            # fs = 192000
            wn = 2 * fl / fs
            b, a = signal.butter(8, wn, 'high')
            # audio = energy_normlize(audio.reshape(1, -1))
            audio_filted = signal.filtfilt(b, a, audio)

            detected_list = sio.loadmat(os.path.join(file_path, 'TKEO_65', idx))['ClickPos']
            # detected_list = sio.loadmat('from_mat/clickpos0.7.mat')['ClickPos']
            # detected_visual = np.zeros_like(audio_filted)
            detected_list = detected_list.tolist()
            if len(detected_list) == 0:
                continue
            # index_to_remove = []
            # for i in range(len(detected_list)):
            #     detected_pos = detected_list[i]
            #     ## snr estimate
            #     signal_cliped = audio_filted[detected_pos[0]:detected_pos[1] + 1]
            #     detected_clicks_energy = model_test.calcu_click_energy(signal_cliped.reshape(1, -1))
            #     noise_estimate1 = audio_filted[detected_pos[0] - 256:detected_pos[0]]
            #     noise_estimate2 = audio_filted[detected_pos[1] + 1:detected_pos[1] + 257]
            #     noise_estimate = np.hstack((noise_estimate1, noise_estimate2))
            #     noise_energy = model_test.calcu_energy(noise_estimate)
            #     snr = 10 * math.log10(detected_clicks_energy / noise_energy)
            #     if snr < 10:
            #         # detected_visual[detected_pos[0]:detected_pos[1] + 1] = 0
            #         index_to_remove.append(i)
            #     # ##
            #     # length = detected_pos[1]-detected_pos[0]+1
            #     # confident = np.sum(detected_visual[detected_pos[0]:detected_pos[1]+1])
            #     # if confident <= length:
            #     #     detected_visual[detected_pos[0]:detected_pos[1]+1] = 0
            #     #     index_to_remove.append(i)
            # has_removed = 0
            # for i in index_to_remove:
            #     detected_list.pop(i-has_removed)
            #     has_removed = has_removed+1
            # for i in detected_list:
            #     detected_visual[i[0]:i[1]] = 1
            #

            # clicks_position = np.zeros_like(audio_filted)
            # for i in position:
            #     clicks_position[i[0]:i[1]] = 6000
            # pl.plot(time, clicks_position)

            position = sio.loadmat(os.path.join(file_path, 'position', audio_name+'_position.mat'))['position']
            # print('the audio of %s' % audio_name)
            # print('the number of detected clicks: %g' % len(detected_list))
            detected_matcher = detection_rate.marked_pool(position)
            recall, precision = detected_matcher.calcu_detection_rate(detected_list)
            # print('the recall: %g \nthe precision: %g' % (recall, precision))
            species_recall[db_indx].append(recall)
            species_precision[db_indx].append(precision)

        # calculating the average accuracy
        print('the %s TKEO results:' % species)
        for i in range(len(species_recall)):
            recall_list = species_recall[i]
            precision_list = species_precision[i]
            db = (i+2)*2 - 1
        # for recall_list, precision_list in species_recall, species_precision:
            if len(recall_list) == 0:
                recall_average = 0
                precision_average = 0
            else:
                recall_average = sum(recall_list)/len(recall_list)
                precision_average = sum(precision_list)/len(precision_list)
            print('the %g-dB by averaging %g times experiment:' % (db, len(recall_list)))
            print('the mean recall: %g , the mean precision: %g' % (recall_average, precision_average))


if __name__ == '__main__':
    main()