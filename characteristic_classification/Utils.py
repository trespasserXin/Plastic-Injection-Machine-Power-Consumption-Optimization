import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import figure
import ruptures as rpt
from collections import defaultdict
from pathlib import Path

THRESHOLD = 1
TIME_INTERVAL = 15

def parse_data(folder_path):
    folder_path = Path(folder_path)
    csv_files = list(folder_path.glob("*.csv"))
    merged_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    return merged_df


def hampel_filter(x, window_size=7, n_sigmas=3):
    """
    Apply a Hampel filter to detect & replace outliers in a 1D array or series.

    Parameters
    ----------
    x : array‐like (list, np.ndarray or pd.Series)
        The input signal.
    window_size : int
        The size of the sliding window (must be odd).
    n_sigmas : float
        How many scaled‐MADs to use as the outlier threshold.

    Returns
    -------
    filtered : np.ndarray
        A copy of x where detected outliers have been replaced by the local median.
    """
    x = np.asarray(x, dtype=float)
    L = window_size // 2
    k = 1.4826  # scale factor so MAD ≃ σ for Gaussian data
    filtered = x.copy()

    for i in range(len(x)):
        start = max(0, i - L)
        end = min(len(x), i + L + 1)
        window = x[start:end]

        med = np.median(window)
        mad = np.median(np.abs(window - med))
        thresh = n_sigmas * k * mad

        if np.abs(x[i] - med) > thresh:
            filtered[i] = med

    return filtered


def get_start_end(data, threshold, if_motor, interval = 15, zero_run_len = 10):
    """
    The function will transform the entries in the data array that is above the threshold as 1, and below
    as 0. Then, the difference between each entry is calculated and stored in "transitions". The "1"s in
    transitions are identified as start points, and "-1"s are seen as end points.

    :param data: Pandas series with length N for identifying the start and end points.
    :param threshold: pre-defined threshold.
    :param if_motor: boolean variable for adjusting the index
    :return: data array with 2 * K dimension. Each column represents a start and end point of a run.
            K idle_slices are identified
    """
    if interval == 1:
        # data_ref = data.copy().to_numpy()
        # i = 0
        # while i <= len(data_ref):
        #     window_start = max(0, i)
        #     window_end = window_start + 15
        #     window = data_ref[window_start: window_end].copy()
        #     if not np.all(window==0):
        #         if np.any(data_ref[window_end-30: window_end-15] > 0):
        #             window[np.argwhere(window == 0)] = 20
        #             data[window_start: window_end] = window
        #     else:
        #         if np.any(data_ref[window_end-30: window_end-15] > 0):
        #             # print(data_ref[window_end-30: window_end-15])
        #             data[window_end-30: window_end] = data_ref[window_end-30: window_end]
        #     i += 15

        intervals = []

        active = False
        start_idx = None
        zero_count = 0

        idx = data.index

        for i, (ix, val) in enumerate(data.items()):
            # normalize for comparisons
            is_zero = pd.notna(val) and (val == 0)

            if not active:
                if pd.notna(val) and (val > threshold):
                    active = True
                    start_idx = ix
                    zero_count = 0
            else:
                # we're inside an interval
                if is_zero:
                    zero_count += 1
                    if zero_count == zero_run_len:
                        # zero-run starts at position i - zero_run_len + 1
                        end_pos = i - zero_run_len  # index of last non-zero before run
                        if end_pos >= 0:
                            end_idx = idx[end_pos]
                            intervals.append((start_idx, end_idx))
                        # reset state to idle
                        active = False
                        start_idx = None
                        zero_count = 0
                else:
                    zero_count = 0  # any non-zero (or NaN) breaks the zero streak

        # close an open interval at the series end
        if active:
            intervals.append((start_idx, idx[-1]))

        intervals = np.array(intervals).reshape(-1, 2).T
        combined_indices = intervals

    else:
        if if_motor:
            data = np.pad(data, (1,), 'constant', constant_values=0)

        motor_on = (data > threshold).astype(int)
        transitions = np.diff(motor_on)

        start_indices = np.array(np.where(transitions == 1))
        end_indices = np.array(np.where(transitions == -1))-int(if_motor)

        combined_indices = np.concatenate((start_indices, end_indices))
    # if interval == 1:
    #     combined_indices[0, :] = combined_indices[0, :] + 1
    #     for run in range(combined_indices.shape[1]):
    #         prev_end = combined_indices[1, run]
    #         post_start = combined_indices[0, run+1]
    #         if post_start - prev_end <= 15:
    #             combined_indices[1, run] = combined_indices[1, run + 1]

    if not if_motor:

        duration = np.diff(combined_indices, axis=0)
        combined_indices = np.concatenate((combined_indices, duration))
        combined_indices = np.delete(combined_indices, np.where(combined_indices[2, :] <= 60/TIME_INTERVAL),
                                     axis=1)

        run_id = np.arange(combined_indices.shape[1]) + 1
        combined_indices = np.vstack((combined_indices, run_id))

    return combined_indices

def startup_correction(power_data):
    """
    Only useful when dealing with 15 min interval cases to modify the abnormal value at the first entry
    of a motor run
    """
    start_index = 0
    pre_start = power_data[start_index]
    if pre_start < power_data[start_index + 1]*0.8 and pre_start != 0:
        power_data[start_index] = power_data[start_index + 1]

    return power_data

def interval_agg(idle_df):

    """
    Interval aggregation function used to combine connected idle intervals(end points of last interval
    equals to the start point of the next interval

    :param idle_df: resulted idle interval dataframe
    :return:aggregated idle interval dataframe
    """

    idle_df = idle_df.sort_values("start").reset_index(drop=True)
    new_group = idle_df["start"].ne(idle_df["end"].shift(1))
    grp = new_group.cumsum()
    agg_dict = {"start": "first","end": "last", "mean": "mean","std": "mean", "start time":"first",
            "end time": "last"}
    merged_idle = idle_df.groupby(grp, as_index=False).agg(agg_dict)
    return merged_idle

def idle_seg_filter(idle_seg, tot_seg, run_length):
    """
    In this function, different idle segments filter are applied to make sure the final idle phase meet
    the desired criteria.

    :param idle_seg: idle segments dataframe without filtering
    :param tot_seg: total segments dataframe resulted from the change point detection algorithm
    :param run_length: Total run length of an identified motor run
    :return: a flag variable, and filtered idle segments dataframe
    """
    valid = False
    # Return False if no idle seg is detected
    if idle_seg.empty:
        return valid, None

    # Idle interval aggregation: combine consecutive intervals
    idle_seg = interval_agg(idle_seg)

    # Drop any idle interval that is smaller than 45 minutes
    idle_dur = idle_seg["end"] - idle_seg["start"]
    idle_seg = idle_seg.drop(idle_seg[idle_dur < 45/TIME_INTERVAL].index)

    """
    Below are optional filter options, not sure if and how they should be implemented.
    """
    # Drop idle seg that starts after 40% of the motor run
    if idle_seg.empty:
        return valid, None

    first_idle = idle_seg["start time"].iat[0] - tot_seg["start time"].iat[0]

    time_to_start = idle_seg["start time"] - tot_seg["start time"][0]
    seg_to_drop = time_to_start > run_length * 0.4
    idle_seg = idle_seg.drop(idle_seg[seg_to_drop].index)

    # If all idle segments are started after 25% of the motor run, return False
    if first_idle > run_length*0.25 or idle_seg.empty:
        return valid, None

    valid = True
    return valid, idle_seg

# Not used in the current version
def therm_motor_diff(motor_data, therm_data, time_data):

    motor_state = (motor_data>THRESHOLD).astype(int)
    therm_state = (therm_data>THRESHOLD).astype(int)
    diff = therm_state - motor_state
    only_therm = diff == 1

    return time_data[only_therm]

def motor_fit(motor_run, timestamp, idle_min, idle_max, sig_threshold):
    """
    Core function for change point detection algorithm
    :param motor_run: a dataframe containing 1 motor run
    :param timestamp: corresponding timestamp for this motor run
    :param idle_min: min threshold for idle segment
    :param idle_max: max threshold for idle segment
    :param sig_threshold: standard deviation threshold for idle segment
    :return: breakpoints for the targeted motor run (change points)
    """
    # Implement the Pelt algorithm for CPD (change point detection)
    # motor_run = hampel_filter(motor_run, window_size=9, n_sigmas=1)
    # sigma2 = np.var(motor_run, ddof=1)  # unbiased global variance
    lam = 1.5/TIME_INTERVAL * np.log(len(motor_run))
    model = rpt.Pelt(model="rbf").fit(motor_run)
    bkps = model.predict(pen=lam)

    bkps = [i-1 for i in bkps]
    """
    Here, we will implement a post-merge algorithm for breakpoints, so that the higher load from
    start-up procedure of the motor is not separated from the rest of the data. This is only used for
    1 min interval
    """
    if TIME_INTERVAL < 15:
        EPS_M = 4                # magnitude below which you merge
        EPS_S = 3
        MIN_PERCENT = 0.15                 # min length to keep
        first_seg = motor_run[:bkps[0]+1]
        second_seg = motor_run[bkps[0]:bkps[1]+1]
        first_seg_mean = np.average(first_seg)
        first_seg_std = np.std(first_seg)
        second_seg_mean = np.average(second_seg)
        second_seg_std = np.std(second_seg)

        if abs(first_seg_mean-second_seg_mean) < EPS_M and abs(first_seg_std-second_seg_std) < EPS_S:
            bkps = bkps[1:]

    # Calculate the mean and standard of each resulted segment, and those meet the criteria will be
    # selected as idle segments
    segments = []
    start = 0
    start_ts = timestamp[0]
    start_time = pd.to_datetime(start_ts, format='%Y-%m-%d %H:%M')
    for end in bkps:

        seg = motor_run[start:end+1]
        mu, sigma = seg.mean(), seg.std()
        end_time = pd.to_datetime(timestamp[end], format='%Y-%m-%d %H:%M')
        segments.append({"start": start, "end": end, "mean": mu, "std": sigma, "start time": start_time,
                         "end time": end_time, "run_id": 0})
        start = end
        start_time = end_time

    seg_df = pd.DataFrame(segments)

    mean_load = seg_df["mean"].max()  # crude estimate from the noisiest slice
    idle_mask = (
            (seg_df["mean"].between(idle_min, idle_max))
            & (seg_df["std"] < mean_load / sig_threshold)
    )

    idle_segments = seg_df[idle_mask]

    return idle_segments, seg_df


def run_cluster(data_array, indices):
    """
    Main function for idle phase identification algorithm
    :param data_array: entire data frame
    :param indices: indices containing start and end points of active idle_slices
    :return:
    """
    motor_data, time_data = data_array
    tot_idle = pd.DataFrame(columns = ["start", "end", "mean", "std",
                                     "start time", "end time", "run_id"])
    # Iterate through each active run
    for index in range(indices.shape[1]):
        # print(indices[0, index]+1, indices[1, index]+1)
        motor_tot = motor_data[indices[0, index]+1: indices[1, index]+1]
        # If in this active run, the motor is not on at all, go to next iteration
        if np.sum(motor_tot) == 0:
            continue
        time_tot = time_data[indices[0, index]+1: indices[1, index]+1]
        # therm_tot = np.sum(therm_data, axis=0)[indices[0, index]+1: indices[1, index]+1]
        # therm_only_t = therm_motor_diff(motor_tot, therm_tot, time_tot)

        #  Get motor start and end points within each active run
        motor_status = get_start_end(motor_tot, THRESHOLD, True)
        # Truncate any motor run that is after half of a complete run
        # motor_status = np.delete(motor_status, motor_status[0,:]>motor_status[1,-1], axis=1)

        # Delete any motor run that is shorter than 45 minutes
        # Delete any motor run that is less than 45 min
        run_dur = motor_status[1,:] - motor_status[0,:]
        motor_status = np.delete(motor_status, run_dur < 45/TIME_INTERVAL, axis=1)

        # Initialize some variables
        idle_seg_sum = pd.DataFrame(columns = ["start", "end", "mean", "std",
                                     "start time", "end time", "run_id"])
        tot_seg_sum = pd.DataFrame(columns = ["start", "end", "mean", "std",
                                     "start time", "end time", "run_id"])
        tot_duration = []
        # print(time_data[indices[0, index]+1], motor_status)
        # Iterate through each motor run within each active run
        for cols in range(motor_status.shape[1]):
            start_idx, end_idx = motor_status[0, cols], motor_status[1, cols]
            # Get necessary data for this motor run
            motor_per_run = motor_tot[start_idx:end_idx+1]
            time_per_run = time_tot[start_idx:end_idx+1]
            motor_corrected = startup_correction(motor_per_run)

            # Pass the motor data into the core change point detection function with min:50, max:54,
            # and std threshold of 10 (this only deals with motor 1)
            idle_seg, tot_seg = (
                motor_fit(motor_corrected, time_per_run,50, 55, 10))
            tot_seg["run_id"] = index + 1

            # Move the idle phase start time to the beginning of an active run
            if not idle_seg.empty and cols == 0 and motor_status[0,0] > 0:
                # print("moving idle start time...", time_data[indices[0, index]+1])
                idle_seg.loc[idle_seg.index[0], "start time"] = time_data[indices[0, index]+1]
                # print(idle_seg)
                # print(idle_seg.loc[0, "start time"])
            # Store resulted idle phases and overall segments
            idle_seg_sum = pd.concat([idle_seg_sum, idle_seg], ignore_index=True)
            tot_seg_sum = pd.concat([tot_seg_sum, tot_seg], ignore_index=True)
            tot_duration.append((tot_seg["end time"] - tot_seg["start time"]).sum())

        motor_duration = np.sum(tot_duration)
        # Pass the crude idle phases into the filter to get selected idles
        idle_flag, corrected_idle = idle_seg_filter(idle_seg_sum, tot_seg_sum, motor_duration)

        # Output idle phase for each active run
        if not idle_flag:
            pass
            # print(tot_seg_sum)
            # print("No Idle Phase Found/No Idle Phase in Starting Phase")
        else:
            corrected_idle["run_id"] = index + 1
            tot_idle = pd.concat([tot_idle, corrected_idle], ignore_index=True)
            # print("Overall seg:", tot_seg_sum)
            # print("Idle seg:", corrected_idle)

    return tot_idle

# Plotting function
def plot_idle_phase(idle_phase_df, runtime_info, data_array):
    time_array, motor_data, therm2_, therm4_ = data_array
    time_array = pd.to_datetime(time_array, format='%Y-%m-%d %H:%M')
    plot_dict = defaultdict(list)
    start_loc = []
    # idle_loc = np.zeros(idle_phase_df.shape[0])

    for start, end in zip(idle_phase_df["start time"], idle_phase_df["end time"]):

        start_index = int(np.argwhere(time_array==start).squeeze())
        # print(start_index)
        start_loc.append(start_index)

    run_start, run_end = runtime_info[0, :], runtime_info[1, :]
    interval = pd.IntervalIndex.from_arrays(run_start, run_end, closed="left")
    loc = interval.get_indexer(start_loc)
    print(len(loc))
    for i in range(len(loc)):
        idle_start = idle_phase_df["start time"][i]
        idle_end = idle_phase_df["end time"][i]
        plot_dict[loc[i]].extend([idle_start, idle_end])

    plot_dict.update((k, np.unique(v)) for k, v in plot_dict.items())

    for run_id in plot_dict.keys():
        start = runtime_info[0, run_id]
        end = runtime_info[1, run_id]+1
        motor_interval = motor_data[start:end]
        therm2_interval = therm2_[start:end]
        therm4_interval = therm4_[start:end]
        ts_interval = time_array[start:end]

        idles = [plot_dict[run_id][0], plot_dict[run_id][1]]
        figure()
        plt.plot(ts_interval, motor_interval, label="Motor 1 Power")
        plt.plot(ts_interval, therm2_interval, label="Therm 2 Power")
        plt.plot(ts_interval, therm4_interval, label="Therm 4 Power")
        plt.vlines(idles, ymin=np.min(motor_interval), ymax=np.max(motor_interval), colors='blue',
                   linestyles='dashed', label='Idle Phases')
        plt.xlabel('Time',fontsize=12, rotation=0, ha='right', va='top',labelpad=5)
        plt.ylabel('Power', fontsize=12)
        locator = mdates.MinuteLocator(interval=TIME_INTERVAL*30)
        formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gcf().autofmt_xdate()
        plt.xlim(ts_interval[0], ts_interval[-1])
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        plt.show()

    return plot_dict