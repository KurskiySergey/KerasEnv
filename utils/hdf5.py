import h5py
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from config import TRAIN_HDF5_DIR, SFTP_HDF5_DIR, DATASETS_DIR


class HDF5:
    def __init__(self, file_path=None, settings=None):
        self.file_path = file_path
        self.settings = settings
        self.window = None
        self.file = None
        self.statistics = None
        self.labels = None
        self.transform_stat = None
        self.loaded = False
        self.add_data = {}

    def parse_file_to_windows(self, time_step, channel_step):
        window_config = self.settings['WINDOW']
        time_window_step = int(window_config['time'])
        channel_window_step = int(window_config['channel'])
        origin_window = self.window
        start_time = 0
        start_channel = 0
        while start_time + time_window_step <= self.statistics.shape[0]-1:
            while start_channel + channel_window_step <= self.statistics.shape[1]-1:
                cut_window = [
                    [start_time, start_time+time_window_step],
                    [start_channel, start_channel+channel_window_step],
                    [0, 0]
                ]
                self.window = cut_window
                self.get_transform()
                self.save_transform(name=f"{start_time}_{start_channel}")
                start_channel += channel_step
            start_channel = 0
            start_time += time_step

        self.window = origin_window

    def get_cut_window(self):
        return self.window

    def to_array(self, transform=False):
        if transform:
            return np.asarray(self.transform_stat.getResultData())
        return np.asarray(self.statistics)

    def join_files(self, hdf5_file_list:list, axis=0):
        labels = []
        stats = []
        for hdf5_file in hdf5_file_list:
            hdf5_file.load_file()
            stats.append(hdf5_file.to_array())
            labels.append(hdf5_file.labels)
        stats = np.concatenate(stats, axis=axis)
        labels = np.concatenate(labels, axis=axis)
        self.statistics = stats
        self.labels = labels
        self.get_transform()

    def from_array(self, data):
        self.statistics = data
        self.get_transform()

    def set_labels(self, data):
        self.labels = data

    def get_labels(self):
        return np.copy(np.asarray(self.labels)) if self.labels is not None else None

    def add_new_data(self, label, data):
        self.add_data[label] = data

    def get_data(self, label):
        self.add_data[label] = self.file[label]
        return self.add_data[label]

    def set_cut_window(self, window):
        self.file.attrs['cut_settings'] = window
        self.window = window

    def save_transform(self):
        transform_data = self.transform_stat.getResultData()
        transform_data = np.asarray(transform_data)
        file = h5py.File(self.file_path, 'w')
        file['statistics'] = transform_data
        if self.labels is not None:
            file['label'] = self.labels
        file.close()

    def get_transform(self):
        if self.statistics is not None:
            self.transform_stat = StatisticTransform(origin=self.statistics)
            if self.window is not None:
                self.transform_stat.useSettings(cut_settings=self.window)

    def show_origin_data(self, grayscale=False, show_axis=False, get_img=False, use_plt_show=True):
        img = self.transform_stat.show_img(grayscale=grayscale, show_axis=show_axis, transformed=False, get_img=get_img, use_plt_show=use_plt_show)
        if get_img:
            return img

    def show_transformed_data(self, grayscale=False, show_axis=False, get_img=False, use_plt_show=True):
        img = self.transform_stat.show_img(grayscale=grayscale, show_axis=show_axis, get_img=get_img, use_plt_show=use_plt_show)
        if get_img:
            return img

    def load_file(self):
        if not self.loaded:
            self.file = h5py.File(self.file_path, 'r+')
            cut_settings = self.file.attrs.get("cut_settings")
            if cut_settings is not None:
                self.window = cut_settings
            self.statistics = self.file['statistics']
            try:
                self.labels = self.file["label"]
            except KeyError:
                self.labels = np.zeros(shape=(self.statistics.shape[0]))
            self.get_transform()
            self.loaded = True

    def close(self):
        self.statistics = None
        self.window = None
        self.transform_stat = None
        self.file.close()
        self.loaded = False


class StatisticTransform:
    def __init__(self, origin):
        self.origin = origin
        self.timeLen = (0, self.origin.shape[0]-1)
        self.channelLen = (0, self.origin.shape[1]-1) if len(self.origin.shape) > 1 else (0, 0)
        self.filterLen = (0, self.origin.shape[2]-1) if len(self.origin.shape) > 2 else (0, 0)
        self.result = origin

    def show_img(self, transformed=True, grayscale=False, show_axis=False, get_img=False, use_plt_show=True):
        channels = 3
        filters = self.filterLen[1] - self.filterLen[0]
        filter_step = filters // channels
        if transformed:
            data = self.result
        else:
            data = self.origin

        if filter_step == 0:
            # grayscale
            print(filters, self.filterLen, data.shape)
            if filters == 0:
                np_hdf5 = np.clip(255 - np.squeeze(data, axis=-1) * 255, 0, 255)
            img = Image.fromarray(np_hdf5)
        else:
            start_filter = [i*filter_step for i in range(channels)]
            rgb_channels = [np.sum(data[:, :, start_filter[i]:start_filter[i] + filter_step], axis=2) / filter_step for i in range(channels)]
            rgb_img = np.stack(rgb_channels, axis=2)
            rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
            img = Image.fromarray(rgb_img)
            if grayscale:
                img = img.convert(mode="L")
        if not get_img:
            if show_axis:
                plt.imshow(img)
                if use_plt_show:
                    plt.show()
            else:
                img.show()
        return img

    def reset(self):
        del self.result
        self.result = self.origin
        self.timeLen = (0, self.origin.shape[0]-1)
        self.channelLen = (0, self.origin.shape[1]-1)
        self.filterLen = (0, self.origin.shape[2]-1)

    def cutTime(self, startTime, endTime):
        if startTime < self.timeLen[0]:
            startTime = self.timeLen[0]
        if endTime > self.timeLen[1]+1 or startTime == endTime:
            endTime = self.timeLen[1]+1
        self.timeLen = (startTime, endTime)

    def cutChannels(self, startChannel, endChannel):
        if startChannel < self.channelLen[0]:
            startChannel = self.channelLen[0]
        if endChannel > self.channelLen[1]+1 or endChannel == startChannel:
            endChannel = self.channelLen[1]+1
        self.channelLen = (startChannel, endChannel)

    def cutFilters(self, startFilter, endFilter):
        if startFilter < self.filterLen[0]:
            startFilter = self.filterLen[0]
        if endFilter > self.filterLen[1]+1 or endFilter == startFilter:
            endFilter = self.filterLen[1]+1
        self.filterLen = (startFilter, endFilter)

    def cutData(self):
        self.result = self.origin[self.timeLen[0]:self.timeLen[1], self.channelLen[0]:self.channelLen[1], self.filterLen[0]:self.filterLen[1]]

    def useSettings(self, cut_settings):
        time_cut = cut_settings[0]
        channel_cut = cut_settings[1]
        filter_cut = cut_settings[2]
        self.reset()
        self.cutTime(time_cut[0], time_cut[1])
        self.cutChannels(channel_cut[0], channel_cut[1])
        self.cutFilters(filter_cut[0], filter_cut[1])
        self.cutData()

    def getResultData(self):
        return self.result

    def getOriginData(self):
        return self.origin


if __name__ == "__main__":
    filename = "/mnt/hard_disk_1/PycharmProjects/bts_qt_keras_jupyter/hdf5/union_hdf5/bts_siml_gan_dataset/bts_siml_gan_dataset_patch_20.hdf5"
    hdf5_file = HDF5(file_path=filename)
    hdf5_file.load_file()
    hdf5_file.show_origin_data()

