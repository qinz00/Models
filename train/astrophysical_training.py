import gc
import os
import pickle
import random
import json
import numpy as np
from keras.optimizer_v2.adam import Adam
import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.astrophysical_model import my_model
from model.transformer_model import EncoderModel_6
loss_fn = tf.keras.losses.Hinge()

class data_sample():
    def __init__(self, noiseE1, noiseE2, noiseE3, mass1_list, mass2_list, spin1z_list, spin2z_list, right_ascension_list,
                 declination_list, signal_E1_list, signal_E2_list, signal_E3_list, snr_E1_list, snr_E2_list, snr_E3_list):
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
        print('mass1='+str(self.mass1_list))
        print('mass2=' + str(self.mass2_list))
        print('spin1z='+str(self.spin1z_list))
        print('spin2z='+str(self.spin2z_list))
        print('right_ascension='+str(self.right_ascension_list))
        print('declination='+str(self.declination_list))
    def help(self):
        print('noiseE1, noiseE2, noiseE3 are 128 s length noise of E1, E2 and E3')
        print('signal_E1_list, signal_E2_list and signal_E3_list are signals, and each have 1 samples')
        print('mass1_list, mass2_list are masses, and each have 1 masses')
        print('right_ascension_list and declination_list are the directions of the source of the signal')


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

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
        # print(file)
        with open(file, 'rb') as f:
            del data
            gc.collect()
            data = pickle.load(f)
        for sample in data:
            yield file, sample

def denoising_sample_cut_iter(sample_length, snr_low_list, snr_high_list, peak_range, freq, snr_change_time, datapath):
    data_iter_sample = sample_iter(datapath)
    noise_rand_begin = 2048
    noise_rand_end = int(63 * 2048 - sample_length * freq)
    gen_num = 0
    obj_snr_index = 0
    snr_range_num = len(snr_low_list)
    for file, data_sample_ in data_iter_sample:
        for signal, snr_s in zip(data_sample_.signal_E1_list, data_sample_.snr_E1_list):
            peak = random.uniform(peak_range[0], peak_range[1])
            if gen_num > 0 and gen_num % int(snr_change_time) == 0:
                obj_snr_index = (obj_snr_index + 1) % snr_range_num
            snr_obj = random.uniform(snr_low_list[obj_snr_index], snr_high_list[obj_snr_index])
            snr_norm = snr_obj / snr_s

            peak_num = int(peak * sample_length * freq)
            signal_peak = np.argmax(signal)
            all_num = sample_length * freq
            signal_after_peak = np.size(signal) - signal_peak
            after_peak = all_num - peak_num
            return_data = np.zeros(all_num)
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
            noise = data_sample_.noiseE1[noise_begin:noise_begin + all_num]
            gen_num = gen_num + 1
            yield return_data + noise, noise

        for signal, snr_s in zip(data_sample_.signal_E2_list, data_sample_.snr_E2_list):
            peak = random.uniform(peak_range[0], peak_range[1])
            if gen_num > 0 and gen_num % int(snr_change_time) == 0:
                obj_snr_index = (obj_snr_index + 1) % snr_range_num

            snr_obj = random.uniform(snr_low_list[obj_snr_index], snr_high_list[obj_snr_index])
            snr_norm = snr_obj / snr_s

            peak_num = int(peak * sample_length * freq)
            signal_peak = np.argmax(signal)
            all_num = sample_length * freq
            signal_after_peak = np.size(signal) - signal_peak
            after_peak = all_num - peak_num
            return_data = np.zeros(all_num)
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
            noise = data_sample_.noiseE2[noise_begin:noise_begin + all_num]
            gen_num = gen_num + 1
            yield return_data + noise, noise

        for signal, snr_s in zip(data_sample_.signal_E3_list, data_sample_.snr_E3_list):
            peak = random.uniform(peak_range[0], peak_range[1])
            if gen_num > 0 and gen_num % int(snr_change_time) == 0:
                obj_snr_index = (obj_snr_index + 1) % snr_range_num  # snr_range_num = len(snr_low_list)
            snr_obj = random.uniform(snr_low_list[obj_snr_index], snr_high_list[obj_snr_index])
            snr_norm = snr_obj / snr_s
            peak_num = int(peak * sample_length * freq)
            signal_peak = np.argmax(signal)
            all_num = sample_length * freq
            signal_after_peak = np.size(signal) - signal_peak
            after_peak = all_num - peak_num
            return_data = np.zeros(all_num)
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
            noise = data_sample_.noiseE3[noise_begin:noise_begin + all_num]
            gen_num = gen_num + 1
            yield return_data + noise, noise

def get_train_batch_iter(batch_size, snr_low_list, snr_high_list, sample_length, peak_range, freq, snr_change_time,
                         datapath):
    my_denoising_iter = denoising_sample_cut_iter(sample_length, snr_low_list, snr_high_list, peak_range, freq,
                                                  snr_change_time, datapath)
    count = 1
    batch_x = []
    batch_y = []
    for strain, signal in my_denoising_iter:
        if count == 1:
            batch_x = strain / np.max(strain)
            batch_y = signal / np.max(signal)
        else:
            batch_x = np.concatenate((batch_x, strain / np.max(strain)))
            batch_y = np.concatenate((batch_y, signal / np.max(signal)))
        if count == batch_size:
            yield (batch_x.reshape(-1, sample_length * freq, 1), batch_y.reshape(-1, sample_length * freq, 1))
            del batch_x, batch_y
            gc.collect()
        count = count + 1
        if count > batch_size:
            del count
            gc.collect()
            count = 1


def loss_func1(denoise_model, x, y, training):
    loss_object = tf.keras.losses.MeanSquaredError()
    x_denoise = denoise_model(x, training=training)
    return loss_object(x_denoise, y)

def loss_func2(denoise_model, discriminator, z, fake_labels, training1, training0):
    # loss = tf.keras.losses.CategoricalCrossentropy()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    # print(z.shape)
    z_denoise = denoise_model(z, training=training0)
    y = discriminator(z_denoise, training=training1)

    return loss_object(fake_labels, y)

def loss_func3(denoise_model, discriminator, x, valid_labels, training1, training0):
    # loss = tf.keras.losses.CategoricalCrossentropy()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    x_denoise = denoise_model(x, training=training0)
    y = discriminator(x_denoise, training=training1)

    return loss_object(valid_labels, y)

def loss_func4(denoise_model, z, training):
    y_pred = denoise_model(z, training=training)
    return y_pred ** 2

def train_disc(denoise_model, discriminator, batch_size, snr_low_list, snr_high_list,
               sample_length, peak_range, freq, snr_change_time, train_data_path, val_data_path):
    train_data_loader = get_train_batch_iter(batch_size=batch_size, snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                             sample_length=sample_length, peak_range=peak_range, freq=freq,
                                             snr_change_time=snr_change_time, datapath=train_data_path)
    val_data_loader = get_train_batch_iter(batch_size=batch_size, snr_low_list=snr_low_list,
                                           snr_high_list=snr_high_list,
                                           sample_length=sample_length, peak_range=peak_range, freq=freq,
                                           snr_change_time=snr_change_time, datapath=val_data_path)
    true_labels = np.concatenate((np.ones(shape=(batch_size, 1)), np.zeros(shape=(batch_size, 1))), axis=1)
    fake_labels = np.concatenate((np.zeros(shape=(batch_size, 1)), np.ones(shape=(batch_size, 1))), axis=1)
    denoise_model_1 = denoise_model

    discriminator_model = discriminator
    bce_optimizer = Adam(learning_rate=0.00001)
    train_steps = 1
    for x in train_data_loader:
        with tf.GradientTape() as bce_tape:
            d_loss_fake = loss_func2(denoise_model_1, discriminator_model, x[1], fake_labels, training1=True, training0=False)
            d_loss_true = loss_func3(denoise_model_1, discriminator_model, x[0], true_labels, training1=True, training0=False)
            d_loss_fake = tf.cast(d_loss_fake, dtype=tf.float32)
            d_loss_true = tf.cast(d_loss_true, dtype=tf.float32)
            d_loss = 0.2 *d_loss_true + d_loss_fake
        bce_gradients = bce_tape.gradient(d_loss, discriminator_model.trainable_variables)
        bce_optimizer.apply_gradients(zip(bce_gradients, discriminator_model.trainable_variables))
        print('epoch:', (train_steps // 6250) + 1, 'steps:', train_steps, 'd_loss_real:', round(d_loss_true.numpy(), 5), 'd_loss_fake:', round(d_loss_fake.numpy(), 5)) # 647
        train_steps = train_steps + 1
        if train_steps % 6250 == 0:
            val_loss1_save = []
            val_loss0_save = []
            val_steps = 0
            for x in val_data_loader:
                val_steps = val_steps + 1
                D_loss_fake = loss_func2(denoise_model_1, discriminator_model, x[1], fake_labels, False, False)
                D_loss_true = loss_func3(denoise_model_1, discriminator_model, x[0], true_labels, False, False)
                val_loss0_save.append(D_loss_fake.numpy())
                val_loss1_save.append(D_loss_true.numpy())
                if val_steps == 625:
                    print('epoch:', (train_steps // 6250), 'd_loss_real:', np.round((sum(val_loss1_save) / 625), 5), 'd_loss_fake:', np.round((sum(val_loss0_save) / 625), 5)) # 646 37
                    discriminator_model.save('disc_model_{}.h5'.format(train_steps // 6250))
                    file_path = 'onlydisc_loss.txt'
                    with open(file_path, 'a') as file:
                        losses = \
                            [[np.round((sum(val_loss1_save) / 625), 5), np.round((sum(val_loss0_save) / 625), 5)]]  # 37
                        epoch_loss = ''.join([str(loss) for loss in losses])
                        file.write(epoch_loss + '\n')
                    break



if __name__ == '__main__':
    denoise_model = keras.models.load_model('denoise_model.h5',
                                            custom_objects={"EncoderModel_6": EncoderModel_6})
    config = load_config("config/config.json")
    train_path = config['train_path']
    val_path = config['val_path']
    batch_size = config['batch_size']
    sample_length = config['sample_length1']
    snr_low_list = config['snr_low_list']
    snr_high_list = config['snr_high_list']
    peak_range = tuple(config['peak_range1'])
    freq = config['freq']
    snr_change_time = config['snr_change_time']
    train_disc(denoise_model,
               discriminator=my_model(),
               batch_size=batch_size,
               train_data_path=train_path,
               val_data_path=val_path,
               snr_low_list=snr_low_list,
               snr_high_list=snr_high_list,
               sample_length=sample_length,
               peak_range=peak_range,
               freq=freq,
               snr_change_time=snr_change_time)