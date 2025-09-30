from Utils import get_start_end
from Utils import parse_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

THRESHOLD = 1
TIME_INTERVAL = 15

# In some idle_slices, the barrel heat may stop and then start again with all other sensor being on.
# This function will combine these idle_slices identified using barrel heat as a whole run
def run_correction(df, run_info_):
    new_runs = run_info_.copy()
    for run in run_info_[-1,:]:
        end = run_info_[1,run-1]+1
        after_end = end + int(45/TIME_INTERVAL)

        # if therm2[end + int(45/TIME_INTERVAL)] > 0 and therm4[end + int(45/TIME_INTERVAL)] > 0:
        if df.loc[after_end, [" 201 therm #1   (kW)", "201  Therm #2   (kW)", "201  Therm #4   (kW)"]].all():
            next_start = run_info_[0, run]
            # if (np.count_nonzero(therm2[end:run_info_[0,run]+1] == 0) < 2 and
            #         np.count_nonzero(therm4[end:run_info_[0,run]+1] == 0) < 2):
            # if (not np.any(therm2[end:run_info_[0,run]+2] == 0)
            #         and not np.any(therm4[end :run_info_[0,run]+2] == 0)):
            if not any(df.loc[end:next_start, [" 201 therm #1   (kW)", "201  Therm #2   (kW)",
                                               "201  Therm #4   (kW)"]].eq(0).any()):

                # print(df["Time stamp"][run_info_[0,run-1]])
                new_runs[1,run-1] = run_info_[1,run]
                new_runs[:, run] = 0

    new_runs = np.delete(new_runs, np.where(new_runs[-1,:]==0), axis=1)
    new_runs[-1,:] = np.arange(new_runs.shape[1])+1
    return new_runs

def get_run_info(folder):
    dataframe = parse_data(folder)
    dataframe["Time stamp"] = pd.to_datetime(dataframe["Time stamp"], format='%Y-%m-%d %H:%M')

    run_info = get_start_end(np.array(dataframe["201 Barrel Heat  Secondary   (kW)"]), THRESHOLD, False)
    run_info = np.delete(run_info, np.where(run_info[2, :] < 60 / TIME_INTERVAL), axis=1)

    run_info_corrected = run_correction(dataframe, run_info)

    return run_info_corrected, dataframe

# The function will get the on/off status of input data around the start point of each run
# It will look for 2 data before and after the start point and return the on/off status as 0/1
def get_status(start_point, *data_series ):
    status_array = np.zeros([5, len(data_series)])
    for i, data in enumerate(data_series):

        start_vals = data.loc[start_point-2: start_point+2]

        status_array[:, i] = np.array(start_vals != 0).astype(int)

    return status_array

# Yield the start order of Barrel, Therm and Motor. For any sensor starting before Barrel, 0 will be
# assigned, the Barrel will have a fixed number of 1, any sensor start at the same time with the Barrel
# will have be assigned as 1, and any sensor after the Barrel will be assigned as 2 or 3 depending on
# the corresponding order
def get_order(df, run_info_):
    run_order_dict = {"Therm":[],"Barrel":[], "Motor":[], "Run ID":[]}

    therm1 = df[" 201 therm #1   (kW)"]
    therm2 = df["201  Therm #2   (kW)"]
    therm4 = df["201  Therm #4   (kW)"]
    motor1 = df["201 Motor 1   (kW)"]
    for run in run_info_[-1,:]:
        run_order_dict["Run ID"].append(run)
        run_order = np.ones([1, 3]).squeeze()-2
        run_order[1] = 1

        start = run_info_[0,run-1]+1
        end = run_info_[1,run-1]
        t1 = therm1.loc[start:end]
        t2 = therm2.loc[start:end]
        t4 = therm4.loc[start:end]
        therm = t2 + t4 + t1
        m = motor1.loc[start:end]
        if any(s.eq(0).all() for s in (therm, m)):
            # print(run)
            # print(df.loc[run_info_[0,run-1],"Time stamp"])
            run_order_dict["Therm"].append(-1)
            run_order_dict["Barrel"].append(-1)
            run_order_dict["Motor"].append(-1)
            continue
        try:

            th_start = np.argwhere(therm[therm > 0].index.tolist())[0]
            motor_start = np.argwhere(m[m>0].index.tolist())[0]

        except IndexError:
            print(df["Time stamp"][start])

        # For cases which the therms and motors start before and with the barrel
        anchor_status = get_status(start, therm1, therm2, therm4, motor1)

        if np.any(anchor_status[1:4, 0:-1] == 1):
            # th_start = 0
            run_order[0] = 1
        if np.any(anchor_status[1:4, -1] == 1):
            # motor_start = 0
            run_order[2] = 1

        if np.sum(anchor_status[0,0:-1]) > 0:
            run_order[0] = 0

        if anchor_status[0,-1] > 0:
            run_order[2] = 0

        # For cases where one of the motor or therm or both start after the barrel
        if np.any(run_order < 0):
            if len(np.where(run_order < 0)) > 1:
                if abs(th_start - motor_start) <= 1:
                    run_order[0] = 2
                    run_order[2] = 2
                elif th_start > motor_start:
                    run_order[0] = 3
                    run_order[2] = 2
                else:
                    run_order[0] = 2
                    run_order[2] = 3
            else:
                run_order[np.argwhere(run_order < 0)] = 2

        run_order_dict["Therm"].append(run_order[0])
        run_order_dict["Barrel"].append(run_order[1])
        run_order_dict["Motor"].append(run_order[2])

    run_order_df = pd.DataFrame.from_dict(run_order_dict)
    return run_order_df

def split_normal(run_order_df):
    normal_mask = (run_order_df["Barrel"] == run_order_df["Therm"]) & (run_order_df["Motor"] > 1)
    normal_df = run_order_df[normal_mask]
    unnormal_df = run_order_df[~normal_mask]

    return normal_df, unnormal_df
# print(run_orders.head(5))

# for run_id in run_info_[-1, :]:
#     therm2 = dataframe["201  Therm #2   (kW)"]
#     therm4 = dataframe["201  Therm #4   (kW)"]
#     idx = run_info_[0, run_id - 1]
#     if therm4[idx] or therm2[idx] != 0:
#         th_before_B.append(run_id)
#
# print(th_before_B)

if __name__ == "__main__":
    run_info, run_df = get_run_info("datav2")
    run_orders_df = get_order(run_df, run_info)
    # print(run_info_.shape)
    # print(run_orders_df.head(6))

    normal_df, unnormal_df = split_normal(run_orders_df)

    # Change the mask accordingly to get the information of different subcases

    # mask1 = (run_orders_df["Barrel"] == run_orders_df["Therm"]) & (run_orders_df["Motor"] > 1)
    # runs1 = run_orders_df[mask1]
    #
    # mask2 = (run_orders_df["Barrel"] == run_orders_df["Therm"]) & (run_orders_df["Motor"] == run_orders_df["Barrel"])
    # runs2 = run_orders_df[mask2]
    #
    # mask3 = ((run_orders_df["Barrel"] > run_orders_df["Therm"]) &
    #          (run_orders_df["Therm"]>  run_orders_df["Motor"]))
    # runs3 = run_orders_df[mask3]
    # print(len(runs1), len(runs2), len(runs3))

    run_orders_df['order_tuple'] = list(zip(run_orders_df['Therm'], run_orders_df['Barrel'], run_orders_df['Motor']))
    order_counts= run_orders_df['order_tuple'].value_counts()
    print(order_counts)
    labels = ["Barrel = Therm > Motor", "Barrel = Therm = Motor", "Barrel > Therm = Motor",
              "Barrel = Motor > Therm", "Motor > Barrel > Therm", "Motor > Barrel = Therm",
              "Therm > Barrel > Motor", "Abnormal"]
    order_counts.index = labels
    order_counts.plot(kind='bar', figsize=(8, 6))
    plt.xlabel('Start Order')
    plt.ylabel('Count')
    plt.title('Counts of Start Order Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()