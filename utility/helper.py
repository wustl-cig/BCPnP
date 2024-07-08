import os
import torch
import numpy as np
from tifffile import imwrite
from collections import defaultdict
import pathlib


def torch_complex_normalize(x):
    x_angle = torch.angle(x)
    x_abs = torch.abs(x)

    x_abs -= torch.min(x_abs)
    x_abs /= torch.max(x_abs)

    x = x_abs * np.exp(1j * x_angle)

    return x


def strip_empties_from_dict(data):
    new_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = strip_empties_from_dict(v)

        if v not in (None, str(), list(), dict(),):
            new_data[k] = v
    return new_data


def get_last_folder(path):
    return pathlib.PurePath(path).name


def convert_pl_outputs(outputs):
    outputs_dict = defaultdict(list)

    for i in range(len(outputs)):
        for k in outputs[i]:
            outputs_dict[k].append(outputs[i][k])

    log_dict, img_dict = {}, {}
    for k in outputs_dict:
        try:
            tmp = torch.Tensor(outputs_dict[k]).detach().cpu()

            log_dict.update({
                k: tmp
            })

        except Exception:
            if outputs_dict[k][0].dim() == 2:
                tmp = torch.stack(outputs_dict[k], 0).detach().cpu()
            else:
                tmp = torch.cat(outputs_dict[k], 0).detach().cpu()

            if tmp.dtype == torch.complex64:
                tmp = torch.abs(tmp)

            img_dict.update({
                k: tmp
            })

    return log_dict, img_dict


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def merge_child_dict(d, ret, prefix=''):

    for k in d:
        if k in ['setting', 'test']:
            continue

        if isinstance(d[k], dict):
            merge_child_dict(d[k], ret=ret, prefix= prefix + k + '/')
        else:
            ret.update({
                prefix + k: d[k]
            })

    return ret


def write_test(save_path, log_dict=None, img_dict=None):

    if log_dict:

        cvs_data = torch.stack([log_dict[k] for k in log_dict], 0).numpy()
        cvs_data = np.transpose(cvs_data, [1, 0])

        cvs_data_mean = cvs_data.mean(0)
        cvs_data_mean.shape = [1, -1]

        num_index = cvs_data.shape[0]
        cvs_index = np.arange(num_index) + 1
        cvs_index.shape = [-1, 1]

        cvs_data_with_index = np.concatenate([cvs_index, cvs_data], 1)

        cvs_header = ''
        for k in log_dict:
            cvs_header = cvs_header + k + ','

        np.savetxt(os.path.join(save_path, 'metrics.csv'), cvs_data_with_index, delimiter=',', fmt='%.5f', header='index,' + cvs_header)

        print("==========================")
        print("HEADER:", cvs_header)
        print("MEAN:", cvs_data_mean)
        print("==========================")

    if img_dict:

        for k in img_dict:

            imwrite(file=os.path.join(save_path, k + '.tiff'), data=np.array(img_dict[k]), imagej=True)
