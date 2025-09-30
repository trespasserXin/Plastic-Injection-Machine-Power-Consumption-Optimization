from Utils import parse_data, get_start_end
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
TIME_INTERVAL = 15
THRESHOLD = 1

def run_e_visual(data):

    e_df = data
    summary = e_df.groupby('Run Type')['Energy'].agg(
        count='size',
        mean='mean',
        std='std',
        median='median',
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75)
    )
    print(summary)

    sns.boxplot(data=e_df, x='Run Type', y='Energy', showfliers=False)
    plt.title('Energy by Run Type')
    plt.show()

    # Scatter plot for Tt start vs Energy
    sns.scatterplot(
        data=e_df,
        x='Tt Start',
        y=e_df['Energy'] / e_df['Tt Start'],
        hue='Run Type',  # assigns color by category
        palette='tab10',  # pick any seaborn/matplotlib palette
        s=50  # marker size
    )

    plt.ylabel('Energy')
    plt.xlabel('Time to Motor Start')
    plt.title('Scatter of Energy vs Time to Motor Start, colored by Run Type')
    plt.legend(title='Run Type')
    plt.tight_layout()
    plt.show()

    # 2a. Overlayed kernel‐density estimate
    q_low, q_high = e_df['Energy'].quantile([0.01, 0.99])
    # e_df['Energy'] = e_df['Energy'].clip(lower=q_low, upper=q_high)
    sns.kdeplot(data=e_df, x=e_df['Energy']/e_df['Tt Start'], hue='Run Type', common_norm=False)
    # sns.kdeplot(data=e_df, x='Tt Start', hue='Run Type', common_norm=False, clip=(0, np.inf))
    plt.xlabel('Average Power')
    plt.title('Average Power distribution on Normal vs Non-Normal Runs')
    plt.show()


def get_start_end_df(data, threshold):

    motor_on = (data > threshold).astype(int)
    transitions = motor_on.diff().dropna(inplace=False)

    start_indices = transitions.loc[transitions == 1].index
    end_indices = transitions.loc[transitions == -1].index
    m_duration = end_indices - start_indices

    combined_array = np.vstack((np.array(start_indices), np.array(end_indices),
                                    np.array(m_duration)))

    combined_array = np.delete(combined_array, np.where(combined_array[-1, :] <= 75 / TIME_INTERVAL),
                               axis = 1)

    return combined_array

def get_start(data_series, threshold):
    data_status = (data_series >= threshold).astype(int)
    data_start = data_status.loc[data_status == 1].index[0]
    return data_start

def get_order(m_start, b_start, t_start):
    sensors = ['Barrel', 'Motor', 'Therm']
    index = [b_start, m_start, t_start]
    unique_idx = sorted(set(index))
    rank_map = {val: rank+1 for rank, val in enumerate(unique_idx)}
    order = {key: rank_map[val] for key, val in zip(sensors, index)}

    return order

df = parse_data("data_217")
df.rename(columns={'?"Time stamp"':"Time Stamp"}, inplace=True)

run_array = get_start_end(np.array(df.sum(axis=1, numeric_only = True)), THRESHOLD, False)
run_array[0,:] = run_array[0,:]+1

energy_dict = {"Energy":[], "Tt Start":[], "Run Length": [], "Run ID":[]}
order_dict = {'Barrel':[], 'Motor':[], 'Therm':[], 'Run ID':[]}

for run in run_array[-1,:]:

    start = run_array[0,run-1]
    end = run_array[1,run-1]
    motor = df.loc[start-1: end+1,'217 motor   (kW)']
    motor_array = get_start_end_df(motor, THRESHOLD)
    therms = df['217  therm 1   (kW)'] + df['217  therm 2   (kW)'] + df['217  therm 3   (kW)']
    therms_seg = therms.loc[start: end]
    barrel_seg = df.loc[start: end, '217 Barrel Heats   (kW)']
    if therms_seg.eq(0).all() or barrel_seg.eq(0).all():
        print("No Production")
        continue
    therm_start = get_start(therms_seg, THRESHOLD)
    barrel_start = get_start(barrel_seg, THRESHOLD)
    # For now, the motor start will account for any timestamp that motor is started no matter if the
    # run is valid or not
    motor_array = np.delete(motor_array, np.where(motor_array[0, :] < therm_start), axis=1)

    if motor_array.size == 0:
        print("No Production")
        continue
    try:
        motor_start = get_start(motor, THRESHOLD)
    except IndexError:
        print(motor_array.size)

    orders = get_order(motor_start, barrel_start, therm_start)
    for k in list(order_dict.keys())[:3]:
        order_dict[k].append(orders[k])

    order_dict['Run ID'].append(run)

    idle_len = (motor_array[0, 0] - start + 1)*0.25
    energy = df.loc[start: motor_array[0, 0]].select_dtypes(float).to_numpy().sum()

    energy_dict['Tt Start'].append(idle_len)
    energy_dict['Energy'].append(energy*0.25)
    energy_dict['Run ID'].append(run)
    energy_dict['Run Length'].append(run_array[-2,run-1]*0.25)

energy_df = pd.DataFrame(energy_dict)
order_df = pd.DataFrame(order_dict)

order_df['order_tuple'] = list(zip(order_df['Barrel'], order_df['Motor'], order_df['Therm']))
order_counts = order_df['order_tuple'].value_counts()
print(order_counts)
labels = ["Barrel = Therm > Motor", "Barrel > Therm > Motor", "Barrel = Motor = Therm",
              "Barrel = Motor > Therm", "Barrel > Motor > Therm", "Barrel > Motor = Therm",
              "Motor > Barrel > Therm", "Motor > Therm > Barrel", "Motor = Therm > Barrel",
          "Therm > Barrel > Motor", "Therm > Barrel = Motor"]
order_counts.index = labels
figure(1)
order_counts.plot(kind='bar', figsize=(8, 6))
plt.xlabel('Start Order')
plt.ylabel('Count')
plt.title('Counts of Start Order Combinations--Machine 217')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

figure(2, figsize=(12, 6))
counts, bins1, _ = plt.hist(np.array(energy_df['Tt Start']), align='mid', rwidth=0.8, bins=30,
                            cumulative=True)
plt.xlabel('Time to Motor start')
plt.xticks(bins1)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.title('Histogram of Time to Motor Start')
plt.tight_layout()
plt.show()

figure(3, figsize=(10, 6))
_, bins1, _ = plt.hist(np.array(energy_df['Run Length']), align='mid', rwidth=0.8, bins=30,
                       cumulative=True)
plt.xlabel('Run Length')
plt.xticks(bins1)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.title('Histogram of Run Length')
plt.tight_layout()
plt.show()


normal_mask = (((order_df['Barrel'] == 1) & (order_df['Motor'] == 2) & (order_df['Therm'] == 1)) |
               ((order_df['Barrel'] == 1) & (order_df['Motor'] == 3) & (order_df['Therm'] == 2)))

energy_df.loc[normal_mask, 'Run Type'] = 'Normal'
energy_df.loc[~normal_mask, 'Run Type'] = 'Abnormal'
# print(energy_df.head(5))
# print(order_df.head(5))

# run_e_visual(energy_df)
