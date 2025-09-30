import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from Utils import get_start_end

def idle_by_variation(folder_path, mode='by_run', step=1080,
    off_level=1.0, off_min_len=10,
    hampel_win=5, n_sigmas=3.0, var_win=5,
    var_threshold=None, var_clip=1, var_percentile=0.15, min_len=15):
    """
    Identifies periods of low variation (idle periods) in time-series data based on specified criteria
    and threshold values. The function performs various filtering and statistical operations to clean

    the data and determine these periods, including outlier detection (Hampel filter), robust variation
    calculation, and exclusion of predefined "off" states.

    :param folder_path: A string specifying the path to the CSV file containing the input data.
    :param data: A pandas DataFrame containing the input time-series data.
    :param mode: A string specifying how the data should be divided: 1. by runs,
                2. by a fixed interval. Options are 'by_run' or 'fixed_interval'.
    :param step: An integer specifying the step size for 'fixed_interval' mode.
    :param off_level: A float threshold value for detecting "off" states in the data.
                      Data values below or equal to this threshold for a sustained period
                      are considered "off."
    :param off_min_len: An integer specifying the minimum length (in number of samples)
                        for sustained "off" states to be considered valid.
    :param hampel_win: An integer specifying the window size for the Hampel filter,
                       used for detecting and fixing outliers in the dataset.
    :param n_sigmas: A float multiplier for the standard deviation threshold in the Hampel filter.
                    A larger value will result in a less strict filter, and vice versa.
    :param var_win: An integer specifying the window size used for rolling calculations
                    of robust statistical indicator: MAD (median absolute deviation).
    :param var_threshold: A float specifying the variation threshold for detecting idle periods.
                          If not provided, it is determined dynamically using a data-driven method.
    :param var_clip: An integer or float specifying the lower bound value for clipping
                     robust variation calculations prior to threshold determination.
    :param var_percentile: A float in the range [0, 1] specifying the percentile value
                           used to compute the variation threshold when not explicitly provided.
                           It will calculate the threshold at this percentile based on all
                           the variations (MAD) of the current data slice.
    :param min_len: An integer specifying the minimum length (in number of samples) for low-variation
                    (idle) periods to be considered valid.
    :return: A 3*N Numpy array containing the start, end indices, and length of identified
             idle periods.
    """
    def detect_off_mask(s: pd.Series, level=1.0, off_len=10):
        """
        OFF = a certain number of the consecutive values below or equal to the level
        """
        # loc_med = s.rolling(win, center=True).median()
        off = s <= level

        grp = (off != off.shift()).astype(int).cumsum()

        runs = (pd.DataFrame({"off": off}).groupby(grp).agg(
            start=("off", lambda x: x.index[0]),
            end=("off",   lambda x: x.index[-1]),
            is_off=("off", "first"),
            length=("off", "size"),
        ))

        # Keep only sustained OFF runs
        keep = runs[(runs.is_off) & (runs.length >= off_len)]
        mask = pd.Series(False, index=s.index)
        for _, r in keep.iterrows():
            mask.loc[r.start:r.end] = True
        return mask, keep[["start","end","length"]]

    def _rolling_mad(x: np.ndarray) -> float:
        med = np.median(x)
        return np.median(np.abs(x - med))

    def _rolling_iqr(x: np.ndarray) -> float:
        q75, q25 = np.percentile(x, [75, 25])
        return q75 - q25

    def var_quantile(var_series: pd.Series, clip_val, val_percentile):
        clipped_series = var_series.clip(lower=clip_val)
        dup_mask = clipped_series.duplicated(keep='first')

        return np.nanquantile(var_series[~dup_mask], val_percentile)

    def find_low_var_periods_excluding_off(
        s: pd.Series,
        _off_level=off_level, _off_min_len=off_min_len,
        _hampel_win=hampel_win, _n_sigmas=n_sigmas, _var_win=var_win,
        _var_threshold=var_threshold, _var_clip=var_clip,
            _var_percentile=var_percentile, _min_len=min_len):

        off_mask, off_periods = detect_off_mask(s, level=_off_level, off_len=_off_min_len)

        # --- Hampel only on ON-data
        med = s.rolling(_hampel_win, center=True).median()
        mad = s.rolling(_hampel_win, center=True).apply(_rolling_mad, raw=True)
        sigma = 1.4826 * mad
        outliers = (s - med).abs() > (_n_sigmas * sigma)

        # Apply the old "zero dropout" rule ONLY when we're ON
        on_mask = ~off_mask
        seg_quantile = var_quantile(med, 1, 0.3)
        if seg_quantile == 0:
            print(med)
        outliers = outliers | ((s == 0) & on_mask)
        # outliers = outliers | ((s == 0) & (med > seg_quantile))
        s_clean = s.mask(outliers & on_mask, med)   # fix outliers in ON regions
        s_clean = s_clean.interpolate(limit_direction="both")

        # --- Robust variation on cleaned data, but ignore OFF regions
        rmad = s_clean.rolling(_var_win, center=True).apply(_rolling_mad, raw=True)
        iqr = s_clean.rolling(_var_win, center=True).apply(_rolling_iqr, raw=True)

        rstd = 1.4826 * rmad
        rstd_iqr = iqr / 1.349

        rstd = rstd.where(~off_mask)  # fill NaN inside OFF blocks
        rstd_iqr = rstd_iqr.where(~off_mask)

        """The following code automate the process of identifying idle (a.k.a. low-variation) periods."""
        # Threshold from ON-only stats unless the user supplies it

        if _var_threshold is None:
            # Filter out duplicated variation indicator values before calculating the threshold
            rstd_clipped = rstd.clip(lower = _var_clip)
            dup_mask = rstd_clipped.duplicated(keep='first')
            if not rstd[~dup_mask].isna().all():
                # print(s.index[0])
                # print(s.index[-1])
                # pass
                _var_threshold = float(np.nanquantile(rstd[~dup_mask], _var_percentile))

        low = (rstd <= _var_threshold)
        low = low.fillna(False)  # OFF can’t start low-variation runs

        grp = (low != low.shift()).astype(int).cumsum()
        runs = (pd.DataFrame({"low": low}).groupby(grp).agg(
            start=("low", lambda x: x.index[0]),
            end=("low",   lambda x: x.index[-1]),
            is_low=("low","first"),
            length=("low","size"),
        ))
        low_periods = runs[(runs.is_low) & (runs.length >= _min_len)][["start", "end", "length"]]
        # Return cleaned series, variation indicators as a series,
        # identified idle periods as a dataframe, and the calculated idle threshold as a float
        return s_clean, rstd, low_periods, _var_threshold


    # Implementation of the main function starts here
    df = pd.read_csv(folder_path)
    # df = pd.read_csv("C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\merged_data\\201_1min.csv")
    df.rename(columns={'?"Time stamp"': 'TimeStamp'}, inplace=True)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y/%m/%d %H:%M')
    # data = df[:124650].copy()
    data = df.copy()
    if mode == 'by_run':
        run_info = get_start_end(data['201 Barrel Heat  Secondary   (kW)'].copy(),
                                 1, False, 1)
        run_info = run_info[:2, :]

    if mode == 'fixed_interval':
        run_info = np.array([np.arange(0, len(data) - step, step + 1), np.arange(step, len(data), step + 1)])

    data.loc[:, 'Motor Sum'] = data[["201 Motor 1   (kW)", "201 Motor 2   (kW)",
                              "201 motor 3   (kW)"]].sum(axis=1)
    info_arr = []

    # Loop over a Numpy array contains start and end indices of each run
    # either by_run or fixed_interval
    for col in run_info.T:

        period = data.loc[col[0]: col[1], 'Motor Sum'].copy()
        if period.eq(0).all():
            continue
        # Pass a slice of the data to the core function to identify low-variation periods
        info_bag = find_low_var_periods_excluding_off(period)
        idle = info_bag[2].to_numpy().flatten()
        info_arr.extend(idle)

    info_arr = np.array(info_arr).reshape(-1, 3)
    return info_arr

if __name__ == "__main__":
    # Change the file path in your device

    folder_path = "C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\merged_data\\201_1min.csv"
    idle_info = idle_by_variation(folder_path, mode='by_run')
# The following code is used to plot the idle periods identified by the function
# but not used in the current version

"""Machine 201"""
# df.drop('201 Therm #3   (kW)', axis = 1, inplace = True)
# df['Motor Sum'] = df[["201 Motor 1   (kW)", "201 Motor 2   (kW)",
#                           "201 motor 3   (kW)"]].sum(axis=1)
# test_slice = df.loc[41464:42246].copy()
# test_slice = df.loc[92689:93560].copy()
# test_slice = df.loc[63371: 64024].copy()
# test_slice = df.loc[64780: 65393].copy()

# """Machine 221"""
# df['Motor Sum'] = df.loc[:, '221 Motor   (kW)']
# test_slice = df.loc[75180: 76210].copy()
#
# off_mask, clean_s, std, std_iqr, idle, idle_thresh = find_low_var_periods_excluding_off(
#     test_slice['Motor Sum'], hampel_win=8, var_win=8, n_sigmas=3,
#     var_threshold=None, var_percentile=0.1, min_len=15)
# print(idle_thresh)
#
# idle_start_ts = np.array(df.loc[idle['start'], 'TimeStamp'])
# idle_end_ts = np.array(df.loc[idle['end'], 'TimeStamp'])
#
# test_slice[["Rolling MAD", 'Cleaned']] = pd.DataFrame({'Rolling MAD': std, 'Cleaned': clean_s})
#
# # # print(std.loc[92836: 92858])
# # # print(std.loc[92887: 92916])
# # ax1 = test_slice.plot(x='TimeStamp', y=["201 Motor 1   (kW)", "201 Motor 2   (kW)",
# #                           "201 motor 3   (kW)", 'Motor Sum', 'Rolling MAD', 'Cleaned'],figsize=(12, 6))
# # # ax1.vlines(x = idle_start_ts, ymin = -10, ymax = 250, color = 'red', linestyles = '--',
# # #            label = 'Idle Periods Start')
# # # ax1.vlines(x = idle_end_ts, ymin = -10, ymax = 250, color = 'blue', linestyles = '-.',
# # #            label = 'Idle Periods End')
# # for start, end in zip(idle_start_ts, idle_end_ts):
# #     ax1.axvspan(start, end, color = 'orange', alpha=0.3)
# #
# # ax1.legend(bbox_to_anchor=(1.1, 1), loc='upper center', ncol=1)
# # # One tick every 12 hours
# # locator = mdates.MinuteLocator(interval=10)  # or HourLocator(interval=12)
# # ax1.xaxis.set_major_locator(locator)
# # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
# # ticks = ax1.get_xticks()
# # tick_dt = mdates.num2date(ticks)  # convert float date numbers -> datetime
# # ax1.set_xticklabels([dt.strftime("%Y-%m-%d %H:%M") for dt in tick_dt], rotation=30, ha="right")
# # ax1.xaxis.get_offset_text().set_visible(False)
# # plt.tight_layout()
# # plt.show()
# # Machine 201
# # cols = [
# #     "201 Motor 1   (kW)",
# #     "201 Motor 2   (kW)",
# #     "201 motor 3   (kW)",
# #     "Motor Sum",
# #     "Rolling MAD",
# #     "Cleaned",
# # ]
#
# # Machine 221
#
# cols = [
#     '221 Motor   (kW)',
#     "Rolling MAD",
#     "Cleaned",
# ]
#
#
# # --- PREP: use TimeStamp as DatetimeIndex (most reliable for date ticks) ---
# ts = test_slice.copy()
# ts["TimeStamp"] = pd.to_datetime(ts["TimeStamp"])
# ts = ts.set_index("TimeStamp").sort_index()
#
# # Keep only columns that actually exist
# cols = [c for c in cols if c in ts.columns]
#
# # --- PLOT ---
# fig, ax = plt.subplots(figsize=(15, 6))
#
# for c in cols:
#     ax.plot(ts.index, ts[c], label=c)
#
# # Shade idle windows (ensure datetime dtype)
# starts = pd.to_datetime(idle_start_ts)
# ends   = pd.to_datetime(idle_end_ts)
# for a, b in zip(starts, ends):
#     ax.axvspan(a, b, color="orange", alpha=0.3)
#
# # --- X ticks: always show full date+time, no offset text ---
# # Choose a sensible locator based on total span
# # span = ts.index.max() - ts.index.min()
# # if span <= pd.Timedelta(days=2):
# #     locator = mdates.MinuteLocator(interval=15)
# # elif span <= pd.Timedelta(days=7):
# #     locator = mdates.HourLocator(interval=6)
# # else:
# #     locator = mdates.DayLocator(interval=1)
#
# locator = mdates.HourLocator(interval=2)
#
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
# ax.xaxis.get_offset_text().set_visible(False)   # prevent the tiny “offset” date
# plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
#
# # Labels/legend
# ax.set_ylabel("Power (kW)")
# ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
#
# fig.tight_layout()
# plt.show()
