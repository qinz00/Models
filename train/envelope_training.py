import os
import gc
import pickle
import random
import sys
import json
# sys.path.append('../tools')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.denoise_model import mymodel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from matplotlib import pyplot as plt
import tensorflow as tf
from keras.callbacks import Callback
from scipy.signal import hilbert


class data_sample():
    def __init__(self, noiseE1, noiseE2, noiseE3, mass1_list, mass2_list, spin1z_list, spin2z_list,
                 right_ascension_list,
                 declination_list, signal_E1_list, signal_E2_list, signal_E3_list, snr_E1_list, snr_E2_list,
                 snr_E3_list):
        self.noiseE1 = noiseE1
        self.noiseE2 = noiseE2
        self.noiseE3 = noiseE3
        self.mass1_list = mass1_list
        self.mass2_list = mass2_list
        self.spin1z_list = spin1z_list
        self.spin2z_list = spin2z_list
        self.right_ascension_list = right_ascension_list
        self.declination_list = declination_list
        self.signal_E1_list = signal_E1_list
        self.signal_E2_list = signal_E2_list
        self.signal_E3_list = signal_E3_list
        self.snr_E1_list = snr_E1_list
        self.snr_E2_list = snr_E2_list
        self.snr_E3_list = snr_E3_list

    def print(self):
        print('mass1=' + str(self.mass1_list))
        print('mass2=' + str(self.mass2_list))
        print('spin1z=' + str(self.spin1z_list))
        print('spin2z=' + str(self.spin2z_list))
        print('right_ascension=' + str(self.right_ascension_list))
        print('declination=' + str(self.declination_list))

    def help(self):
        print('noiseE1, noiseE2, noiseE3 are 128 s length noise of E1, E2 and E3')
        print('signal_E1_list, signal_E2_list and signal_E3_list are signals, and each have 1 samples')
        print('mass1_list, mass2_list are masses, and each have 1 masses')
        print('right_ascension_list and declination_list are the directions of the source of the signal')


def file_iter(directory):
    file_list = os.listdir(directory)
    while True:
        for dir_item in file_list:
            if dir_item.endswith('.pkl'):
                yield os.path.join(directory, dir_item)

def sample_iter(datapath):
    data = [1, 2, 3]
    data_file_iter = file_iter(datapath)
    for file in data_file_iter:
        with open(file, 'rb') as f:
            del data
            gc.collect()
            data = pickle.load(f)
        for sample in data:
            yield sample


def baoluo_crop_sample_cut_iter(sample_length, snr_low_list, snr_high_list, peak_range, freq, snr_change_time,
                                datapath):
    data_iter_sample = sample_iter(datapath)
    noise_rand_begin = 2048
    noise_rand_end = int(63 * 2048 - sample_length * freq)
    gen_num = 0
    obj_snr_index = 0
    snr_range_num = len(snr_low_list)
    for data_sample_ in data_iter_sample:
        for signal, snr_s in zip(data_sample_.signal_E1_list, data_sample_.snr_E1_list):
            peak = random.uniform(peak_range[0], peak_range[1])
            if gen_num > 0 and gen_num % int(snr_change_time) == 0:
                obj_snr_index = (obj_snr_index + 1) % snr_range_num
            snr_obj = random.uniform(snr_low_list[obj_snr_index], snr_high_list[obj_snr_index])
            snr_norm = snr_obj / snr_s
            peak_num = int(peak * sample_length * freq)
            signal_peak = int(np.argmax(signal))
            all_num = int(sample_length * freq)
            signal_after_peak = int(np.size(signal) - signal_peak)
            after_peak = int(all_num - peak_num)
            return_data = np.zeros(int(all_num))
            if (peak_num < signal_peak):
                return_data[:peak_num] = signal[signal_peak - peak_num:signal_peak] * snr_norm
            else:
                return_data[peak_num - signal_peak:peak_num] = signal[0:signal_peak] * snr_norm
            if (signal_after_peak < after_peak):
                return_data[peak_num:peak_num + signal_after_peak] = signal[
                                                                     signal_peak:signal_peak + signal_after_peak] * snr_norm
            else:
                return_data[peak_num:] = signal[signal_peak:signal_peak + after_peak] * snr_norm

            noise_begin = random.randint(noise_rand_begin, noise_rand_end)
            noise = data_sample_.noiseE1[int(noise_begin):int(noise_begin + all_num)]
            gen_num = gen_num + 1

            analytic_signal = hilbert(return_data)
            amplitude_envelope = np.abs(analytic_signal)
            num_samples_to_remove = int(freq / 8)
            amplitude_envelope = amplitude_envelope[num_samples_to_remove:-num_samples_to_remove]
            return_data = return_data[num_samples_to_remove:-num_samples_to_remove]
            noise = noise[num_samples_to_remove:-num_samples_to_remove]

            yield return_data + noise, amplitude_envelope

        for signal, snr_s in zip(data_sample_.signal_E2_list, data_sample_.snr_E2_list):
            peak = random.uniform(peak_range[0], peak_range[1])
            if gen_num > 0 and gen_num % int(snr_change_time) == 0:
                obj_snr_index = (obj_snr_index + 1) % snr_range_num
            snr_obj = random.uniform(snr_low_list[obj_snr_index], snr_high_list[obj_snr_index])
            snr_norm = snr_obj / snr_s

            peak_num = int(peak * sample_length * freq)
            signal_peak = int(np.argmax(signal))
            all_num = int(sample_length * freq)
            signal_after_peak = int(np.size(signal) - signal_peak)
            after_peak = int(all_num - peak_num)
            return_data = np.zeros(int(all_num))
            if (peak_num < signal_peak):
                return_data[:peak_num] = signal[signal_peak - peak_num:signal_peak] * snr_norm
            else:
                return_data[peak_num - signal_peak:peak_num] = signal[0:signal_peak] * snr_norm
            if (signal_after_peak < after_peak):
                return_data[peak_num:peak_num + signal_after_peak] = signal[
                                                                     signal_peak:signal_peak + signal_after_peak] * snr_norm
            else:
                return_data[peak_num:] = signal[signal_peak:signal_peak + after_peak] * snr_norm

            noise_begin = random.randint(noise_rand_begin, noise_rand_end)
            noise = data_sample_.noiseE2[int(noise_begin):int(noise_begin + all_num)]
            gen_num = gen_num + 1
            analytic_signal = hilbert(return_data)
            amplitude_envelope = np.abs(analytic_signal)
            num_samples_to_remove = int(freq / 8)
            amplitude_envelope = amplitude_envelope[num_samples_to_remove:-num_samples_to_remove]
            return_data = return_data[num_samples_to_remove:-num_samples_to_remove]
            noise = noise[num_samples_to_remove:-num_samples_to_remove]
            yield return_data + noise, amplitude_envelope

        for signal, snr_s in zip(data_sample_.signal_E3_list, data_sample_.snr_E3_list):
            peak = random.uniform(peak_range[0], peak_range[1])
            if gen_num > 0 and gen_num % int(snr_change_time) == 0:
                obj_snr_index = (obj_snr_index + 1) % snr_range_num
            snr_obj = random.uniform(snr_low_list[obj_snr_index], snr_high_list[obj_snr_index])
            snr_norm = snr_obj / snr_s

            peak_num = int(peak * sample_length * freq)
            signal_peak = int(np.argmax(signal))
            all_num = int(sample_length * freq)
            signal_after_peak = int(np.size(signal) - signal_peak)
            after_peak = int(all_num - peak_num)
            return_data = np.zeros(int(all_num))
            if (peak_num < signal_peak):
                return_data[:peak_num] = signal[signal_peak - peak_num:signal_peak] * snr_norm
            else:
                return_data[peak_num - signal_peak:peak_num] = signal[0:signal_peak] * snr_norm
            if (signal_after_peak < after_peak):
                return_data[peak_num:peak_num + signal_after_peak] = signal[
                                                                     signal_peak:signal_peak + signal_after_peak] * snr_norm
            else:
                return_data[peak_num:] = signal[signal_peak:signal_peak + after_peak] * snr_norm

            noise_begin = random.randint(noise_rand_begin, noise_rand_end)
            noise = data_sample_.noiseE3[int(noise_begin):int(noise_begin + all_num)]
            gen_num = gen_num + 1

            analytic_signal = hilbert(return_data)
            amplitude_envelope = np.abs(analytic_signal)
            num_samples_to_remove = int(freq / 8)
            amplitude_envelope = amplitude_envelope[num_samples_to_remove:-num_samples_to_remove]
            return_data = return_data[num_samples_to_remove:-num_samples_to_remove]
            noise = noise[num_samples_to_remove:-num_samples_to_remove]

            yield return_data + noise, amplitude_envelope


def get_train_batch_iter(batch_size, snr_low_list, snr_high_list, sample_length, peak_range, freq, snr_change_time,
                          datapath):
    my_denoising_iter = baoluo_crop_sample_cut_iter(sample_length, snr_low_list, snr_high_list, peak_range, freq,
                                                    snr_change_time, datapath)
    count = 1
    batch_x = []
    batch_y = []
    for strain, envelope in my_denoising_iter:
        if count == 1:
            batch_x = strain / np.max(strain)
            batch_y = envelope / np.max(envelope)
        else:
            batch_x = np.concatenate((batch_x, strain / np.max(strain)))
            batch_y = np.concatenate((batch_y, envelope / np.max(envelope)))
        if count == batch_size:
            yield (batch_x.reshape(-1, 4 * freq, 1), batch_y.reshape(-1, 4 * freq, 1))
            del batch_x, batch_y
            gc.collect()
        count = count + 1
        if count > batch_size:
            del count
            gc.collect()
            count = 1



class LearningRatePrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(f"Learning rate at epoch {epoch + 1}: {K.get_value(lr)}")

def load_config(config_path):

    with open(config_path, 'r') as f:

        config = json.load(f)

    return config

def normal_train(config_path):
    model = mymodel()
    config = load_config(config_path)
    check_point = ModelCheckpoint(
        'train_epo{epoch:02d}.h5',
        monitor='val_loss', verbose=1,
        save_best_only=False, mode='min', save_weights_only=False)
    Reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001)
    train_path = config['train_path']
    val_path = config['val_path']
    batch_size = config['batch_size']
    sample_length = config['sample_length2']
    snr_low_list = config['snr_low_list']
    snr_high_list = config['snr_high_list']
    peak_range = tuple(config['peak_range2'])
    freq = config['freq']
    snr_change_time = config['snr_change_time']
    lr_printer = LearningRatePrinter()
    history = model.fit_generator(
        generator=get_train_batch_iter(batch_size=batch_size, snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                       sample_length=sample_length, peak_range=peak_range, freq=freq,
                                       snr_change_time=snr_change_time, datapath=train_path),
        steps_per_epoch=2 * 100000 // batch_size,
        epochs=50,
        verbose=1,
        validation_data=get_train_batch_iter(batch_size=batch_size, snr_low_list=snr_low_list,
                                             snr_high_list=snr_high_list,
                                             sample_length=sample_length, peak_range=peak_range, freq=freq,
                                             snr_change_time=snr_change_time, datapath=val_path),
        validation_steps=2 * 10000 // batch_size,
        callbacks=[check_point, Reduce, lr_printer],
        workers=1
    )





if __name__ == '__main__':
    normal_train("config/config.json")


