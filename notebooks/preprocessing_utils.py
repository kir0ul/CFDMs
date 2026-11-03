#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.interpolate import make_splrep


def smooth_traj(traj, nb_points=256, smoothing_factor=0):
    ori_index = np.arange(len(traj))
    # new_index = np.linspace(start=traj.index.start, stop=traj.index.stop, num=nb_points)
    new_index = np.linspace(start=0, stop=len(traj), num=nb_points)
    for idx, col in enumerate(traj.columns):
        if idx == 0:
            traj_smooth = pd.DataFrame(
                {
                    col: make_splrep(x=ori_index, y=traj[col], s=smoothing_factor)(
                        new_index
                    ),
                }
            )
        else:
            traj_smooth[col] = make_splrep(ori_index, traj[col], s=smoothing_factor)(
                new_index
            )
    return traj_smooth


def resample_array(array, nb_points=256, smoothing_factor=1e-3):
    ori_index = np.arange(array.shape[0])
    new_index = np.linspace(start=0, stop=array.shape[0], num=nb_points)
    if len(array.shape) > 1:
        arr_resampled = np.ones((nb_points, array.shape[1])) * np.nan
        for colidx, _ in enumerate(range(array.shape[1])):
            arr_resampled[:, colidx] = make_splrep(
                ori_index, array[:, colidx], s=smoothing_factor
            )(new_index)
    else:
        arr_resampled = make_splrep(ori_index, array, s=smoothing_factor)(new_index)
    return arr_resampled
