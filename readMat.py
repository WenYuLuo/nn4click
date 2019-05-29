import scipy.io as sio
import numpy as np
import os
import GetData
import model_test
import random
from scipy import signal
import math
import pylab as pl


def shift_clip_data(data, num, shift=32, noise=None, noise_energy=None):
    click_start = 128
    shifted_clicks = data[:, click_start:click_start+256]
    # num_clicks = data.shape[0]
    # click_3db = []
    # click_5db = []
    # click_7db = []
    # click_9db = []
    # click_11db = []
    # click_13db = []
    # click_15db = []
    click_snr = []
    for x in data:
        j = 0
        while j < num:
            rand_shift = random.randint(-shift, shift)
            start = click_start + rand_shift
            shifted_click = x[start:start+256]
            # pl.plot(shifted_click)
            # pl.show()
            shifted_click = shifted_click.reshape((1, -1))
            if noise:
                noise_num = noise.shape[0]
                click_energy = calcu_click_energy(x.reshape((1, -1)))
                picked_noise = random.randint(0, noise_num-1)
                weight = calcu_weight(noise_energy=noise_energy[picked_noise], click_energy=click_energy)
                for m in range(len(weight)):
                    click_with_noise = weight[m]*shifted_click+noise[picked_noise]
                    # pl.plot(click_with_noise[0])
                    # pl.show()
                    if len(click_snr) < len(weight):
                        click_snr.append(click_with_noise)
                    else:
                        click_snr[m] = np.vstack((click_snr[m], click_with_noise))
            shifted_clicks = np.vstack((shifted_clicks, shifted_click))
            j = j + 1
    return shifted_clicks, click_snr


def high_pass_filter(audio, cutoff, fs):
    # pl.plot(audio[0])

    wn = 2 * cutoff / fs
    b, a = signal.butter(8, wn, 'high')
    audio_filted = signal.filtfilt(b, a, audio, axis=1)

    # pl.plot(audio_filted[0])
    # pl.show()
    return audio_filted


def calcu_energy(x):
    # x_norm = np.linalg.norm(x, ord=2)
    energy = np.sum(x**2, axis=1)
    energy = np.true_divide(energy, x.shape[1])
    return energy


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


def calcu_weight(noise_energy, click_energy):
    weight = []
    snr = [3, 5, 7, 9, 11, 13, 15]
    for i in snr:
        k = math.sqrt((noise_energy * 10 ** (i / 10)) / click_energy)
        weight.append(k)
    return weight


if __name__ == '__main__':


    ## 读取click（实际训练样本由这种方式生成
    # 数据为人工标注的
    fs_path = '/CNN_detect_data/clicks'
    mat_path = '/CNN_detect_data/snr_clicks/annot_clicks'
    #
    # supplement_path = '/home/fish/ROBB/CNN_click/click'
    #
    assert os.path.exists(mat_path)
    x = mat_path + '/Mesoplodon_Canarylsles-Annotated_fine.mat'

    # 1367 clicks  # normalization 1285
    Mesoplodon = sio.loadmat(mat_path+'/mesoplodon.mat')['click_data']
    print('Mesoplodon:', Mesoplodon.shape)
    Mesoplodon_fs = sio.loadmat(fs_path + '/Mesoplodon_CanaryIsles_fs.mat')['fs']
    # 406 clicks # norm 352
    RightWhale = sio.loadmat(mat_path + '/right.mat')['click_data']
    print('RightWhale:', RightWhale.shape)
    RightWhale_fs = sio.loadmat(fs_path + '/N_RightwhaleDolphin_fs.mat')['fs']
    # 512 clicks # norm 374
    Rissos = sio.loadmat(mat_path+'/rissos.mat')['click_data']
    print('Rissos:', Rissos.shape)
    Rissos_fs = sio.loadmat(fs_path+'/Rissos_SCORE_annot_fs.mat')['fs']
    # 80 clicks # norm 128
    RoughToothed = sio.loadmat(mat_path + '/rough.mat')['click_data']
    print('RoughToothed:', RoughToothed.shape)
    RoughToothed_fs = sio.loadmat(fs_path + '/RoughToothed_Marianas(MISTC)_Annotated_fs.mat')['fs']
    # 483 clicks # norm 505
    Ziphius = sio.loadmat(mat_path + '/ziphius.mat')['click_data']
    print('Ziphius:', Ziphius.shape)
    Ziphius_fs = sio.loadmat(fs_path + '/Ziphius_Italy_Annoated_fine_fs.mat')['fs']
    # 5867 clicks # norm 14
    PacWhite = sio.loadmat(mat_path + '/pacwhite.mat')['click_data']
    print('PacWhite:', PacWhite.shape)
    PacWhite_fs = sio.loadmat(fs_path + '/PacWhitesidedDolphin_fs.mat')['fs']
    # 7123 clicks # norm 6778
    PilotWhale = sio.loadmat(mat_path + '/pilot.mat')['click_data']
    print('PilotWhale:', PilotWhale.shape)
    PilotWhale_fs = sio.loadmat(fs_path + '/Pilot_whales_Bahamas(AUTEC)_Annotated_NUWC_fs.mat')['fs']
    # 58043 clicks
    Sperm = sio.loadmat(mat_path + '/sperm.mat')['click_data']
    print('Sperm:', Sperm.shape)
    Sperm_fs = sio.loadmat(fs_path + '/Sperm_whales_Bahamas(AUTEC)_Annotated_fs.mat')['fs']
    # 108686 clicks
    Spotted = sio.loadmat(mat_path + '/spotted.mat')['click_data']
    print('Spotted:', Spotted.shape)
    Spotted_fs = sio.loadmat(fs_path + '/SpottedDolphin_Bahamas(AUTEC)_Annotated_fs.mat')['fs']
    # 9303 clicks # norm 24949
    Striped = sio.loadmat(mat_path + '/striped.mat')['click_data']
    print('Striped:', Striped.shape)
    Striped_fs = sio.loadmat(fs_path + '/StripedDolphin_Marianas(MISTC)_Annotated_fs.mat')['fs']
    # # 109
    # BBW1 = sio.loadmat(supplement_path + '/BBW1.mat')
    # BBW1_fs = BBW1['fs']
    # BBW1 = BBW1['click_data']
    # pl.plot(BBW1[0])
    # pl.show()

    # print('BBW1:', BBW1.shape)
    # # 1503
    # PACWHITE1 = sio.loadmat(supplement_path + '/PACWHITE1.mat')
    # PACWHITE1_fs = PACWHITE1['fs']
    # PACWHITE1 = PACWHITE1['click_data']
    # print('PACWHITE1:', PACWHITE1.shape)
    # # 2381
    # PILOT3 = sio.loadmat(supplement_path + '/PILOT3.mat')
    # PILOT3_fs = PILOT3['fs']
    # PILOT3 = PILOT3['click_data']
    # print('PILOT3:', PILOT3.shape)
    # # 2495
    # RIGHT1 = sio.loadmat(supplement_path + '/RIGHT1.mat')
    # RIGHT1_fs = RIGHT1['fs']
    # RIGHT1 = RIGHT1['click_data']
    # print('RIGHT1:', RIGHT1.shape)
    # # 1013
    # SPOTTED5 = sio.loadmat(supplement_path + '/SPOTTED5.mat')
    # SPOTTED5_fs = SPOTTED5['fs']
    # SPOTTED5 = SPOTTED5['click_data']
    # print('SPOTTED5:', SPOTTED5.shape)
    # # 13
    # STRIPED1 = sio.loadmat(supplement_path + '/STRIPED1.mat')
    # STRIPED1_fs = STRIPED1['fs']
    # STRIPED1 = STRIPED1['click_data']
    # print('STRIPED1:', STRIPED1.shape)

    # BBW1 = high_pass_filter(BBW1, cutoff=BBW1_fs/40, fs=BBW1_fs)
    # PACWHITE1 = high_pass_filter(PACWHITE1, cutoff=PACWHITE1_fs/40, fs=PACWHITE1_fs)
    # PILOT3 = high_pass_filter(PILOT3, cutoff=PILOT3_fs/40, fs=PILOT3_fs)
    # RIGHT1 = high_pass_filter(RIGHT1, cutoff=RIGHT1_fs/40, fs=RIGHT1_fs)
    # SPOTTED5 = high_pass_filter(SPOTTED5, cutoff=SPOTTED5_fs/40, fs=SPOTTED5_fs)
    # STRIPED1 = high_pass_filter(STRIPED1, cutoff=STRIPED1_fs/40, fs=STRIPED1_fs)

    # shifting sample...
    # print('shifting supplement sample...')
    # BBW1_shifted, n = shift_clip_data(BBW1, num=49)
    # PACWHITE1_shifted, n = shift_clip_data(PACWHITE1, num=2)
    # PILOT3_shifted, n = shift_clip_data(PILOT3, num=1)
    # RIGHT1_shifted, n = shift_clip_data(RIGHT1, num=1)
    # SPOTTED5_shifted, n = shift_clip_data(SPOTTED5, num=4)
    # STRIPED1_shifted, n = shift_clip_data(STRIPED1, num=100)

    Mesoplodon = high_pass_filter(Mesoplodon, cutoff=Mesoplodon_fs/40, fs=Mesoplodon_fs)
    RightWhale = high_pass_filter(RightWhale, cutoff=RightWhale_fs/40, fs=RightWhale_fs)
    Rissos = high_pass_filter(Rissos, cutoff=Rissos_fs/40, fs=Rissos_fs)
    RoughToothed = high_pass_filter(RoughToothed, cutoff=RoughToothed_fs/40, fs=RoughToothed_fs)
    Ziphius = high_pass_filter(Ziphius, cutoff=Ziphius_fs/40, fs=Ziphius_fs)

    PacWhite = high_pass_filter(PacWhite, cutoff=PacWhite_fs/40, fs=PacWhite_fs)
    PilotWhale = high_pass_filter(PilotWhale, cutoff=PilotWhale_fs/40, fs=PilotWhale_fs)
    Sperm = high_pass_filter(Sperm, cutoff=Sperm_fs/40, fs=Sperm_fs)
    Spotted = high_pass_filter(Spotted, cutoff=Spotted_fs/40, fs=Spotted_fs)
    Striped = high_pass_filter(Striped, cutoff=Striped_fs/40, fs=Striped_fs)

    # shifting sample...
    print('shifting sample...')
    Mesoplodon_shifted, n = shift_clip_data(Mesoplodon, num=4)
    RightWhale_shifted, n = shift_clip_data(RightWhale, num=18)
    Rissos_shifted, n = shift_clip_data(Rissos, num=11)
    RoughToothed_shifted, n = shift_clip_data(RoughToothed, num=199)
    Ziphius_shifted, n = shift_clip_data(Ziphius, num=6)

    PacWhite_shifted, n = shift_clip_data(PacWhite, num=7)
    PilotWhale_shifted, n = shift_clip_data(PilotWhale, num=7)
    Sperm_shifted, n = shift_clip_data(Sperm, num=9)
    Spotted_shifted, n = shift_clip_data(Spotted, num=11)
    Striped_shifted, n = shift_clip_data(Striped, num=38)
    print('done')

    # smallset = np.vstack((Mesoplodon_shifted, RightWhale_shifted, Rissos_shifted, RoughToothed_shifted, Ziphius_shifted,
    #                       PacWhite_shifted, PilotWhale_shifted, Sperm_shifted, Spotted_shifted, Striped_shifted,
    #                       BBW1_shifted, PACWHITE1_shifted, PILOT3_shifted, RIGHT1_shifted, SPOTTED5_shifted, STRIPED1_shifted))
    smallset = np.vstack((Mesoplodon_shifted, RightWhale_shifted, Rissos_shifted, RoughToothed_shifted, Ziphius_shifted,
                          PacWhite_shifted, PilotWhale_shifted, Striped_shifted))
    smallset = GetData.ShuffleData(smallset)[0]
    print('num of sample: %g' % len(smallset))

    sio.savemat('little_shift/quater_manual.mat', {'data': smallset, 'num': len(smallset)})


    # 加载来自厦门海域的脉冲噪声
    # xiamen_impulse = sio.loadmat('/CNN_detect_data/xiamen_impulse/xiamen_impulse_2.mat')['clicks']
    # xiamen_impulse = sio.loadmat('/CNN_detect_data/xiamen_impulse/xiamen_impulse_withjt.mat')['clicks']
    xiamen_impulse = sio.loadmat('/CNN_detect_data/xiamen_impulse/xiamen_im.mat')['not_data']

    # highpass with 5khz cutoff
    xiamen_impulse = high_pass_filter(xiamen_impulse, cutoff=10000, fs=400000)

    print('shifting impulse...')
    xiamen_impulse_shift, n = shift_clip_data(xiamen_impulse, shift=64, num=49)
    xiamen_impulse = GetData.ShuffleData(xiamen_impulse_shift)[0]
    print('done')

    sio.savemat('clean_click/xiamen_impulse_quater1.mat', {'data': xiamen_impulse, 'num': len(xiamen_impulse)})




