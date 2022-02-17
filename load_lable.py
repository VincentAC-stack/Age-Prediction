import datetime

import scipy.io as sio
import pandas as pd
import numpy as np
from matplotlib.pylab import num2date


def load_lable(sample_size=62327):
    matlable_struct = sio.loadmat('.\wiki_crop\wiki.mat')
    matlable = matlable_struct['wiki']

    matlable_gender = matlable['gender'][0][0][0]
    matlable_photo_taken = matlable['photo_taken'][0][0][0]
    matlable_dob = matlable['dob'][0][0][0]
    matlable_full_path = matlable['full_path'][0][0][0]
    data_age = []
    data_gender = []
    data_path = []
    for i in range(sample_size):
        date1 = datetime.datetime(matlable_photo_taken[i], 7, 1).date()
        date2 = num2date(matlable_dob[i]).date()
        interval = date1 - date2
        if interval.days <= 0:
            continue
        if np.isnan(matlable_gender[i]):
            continue
        data_age.append(int(interval.days / 365))
        data_gender.append(matlable_gender[i])
        data_path.append(matlable_full_path[i])
    np_age = np.array(data_age)
    np_gender = np.array(data_gender)
    np_data_path = np.array(data_path)
    np_data_path.resize(1, len(np_data_path))
    np_data_path = np_data_path[0]
    return (np_age, np_gender, np_data_path, len(np_data_path))

load_lable()