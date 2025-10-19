import pandas as pd
import paramiko
import time

prev_file = None
end_ts = None

# def no_run_patch(dataframe: pd.DataFrame, labels):
#     df = dataframe.copy()
#     time_str = df[' measurement_time(UTC)'].copy()
#     dup_msk = time_str.duplicated(keep='first')
#     time_str_unique = time_str[~dup_msk]
#     ts = pd.to_datetime(time_str_unique)
#     ts_df = ts.to_frame()
#     ts_df = ts_df.sort_values(by=' measurement_time(UTC)')
#     ts_df.loc[:, labels] = 0
#     ts_df.set_index(' measurement_time(UTC)', inplace=True)
#     return ts_df

def no_run_patch(start_ts):
    _end_ts = start_ts + pd.Timedelta(minutes=14)
    rng = pd.date_range(start=start_ts, end=_end_ts, freq='1min')
    motor_sum = pd.Series(0, index=rng)
    return motor_sum

def fill_gaps(fill_df):
    full_index = pd.date_range(start=fill_df.index.min(), end=fill_df.index.max(), freq='1min')
    filled = fill_df.reindex(full_index, fill_value=0)
    return filled

def get_file(_prev_file, _end_ts, file_num):

    function_call = True
    new_file_detected = False
    first_file = False
    # SFTP server details
    hostname = "ftp.panpwrws.com"
    port = 22
    username = "magnaplastcoat"
    password = "a1D0NJu8bO2tR"

    # Create SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # accept unknown host keys

    try:
        # Connect to server
        client.connect(hostname, port, username, password)

        # Open SFTP session
        sftp = client.open_sftp()
        sftp.chdir("/Measurements")
        # print("Current remote directory:", sftp.getcwd())

        last_file = sftp.listdir()[-file_num]
        if _prev_file is None:
            _prev_file = last_file
            new_file_detected = True
            first_file = True
        else:
            if _prev_file == last_file:
                function_call = False
                print("No new file detected")
                return _prev_file, _end_ts, function_call, None, last_file
            else:
                _prev_file = last_file
                new_file_detected = True

        print(last_file)
        with sftp.open(last_file, 'r') as f:
            df = pd.read_csv(f)
            f.close()
        # print(df.head(10))
        # df.to_csv('test1.csv')

        # Get end time of the new file
        if new_file_detected:
            # end_ts = get_end_time(df)

            label_201 = ['201 Barrel Heat & Secondary', '201 Motor 1', '201 Motor 2','201 motor 3',
                         '201 Screw Retraction Motor', ' 201 therm #1','201  Therm #2','201 Therm #3',
                         '201  Therm #4']
            df_201 = df[df[' device_name'].isin(label_201)].copy().loc[:, [' device_name',
                                                                ' measurement_time(UTC)',' power(W)']]

            # print(df_201.head())
            # Case 1: No data for 201 and first file, so we don't do anything
            if df_201.empty & first_file:
                print("No data since program start, skipping")
                function_call = False
                return None, None, function_call, None, last_file
            # Case 2: No data for 201 and not the first file
            if df_201.empty:
                print("No data in the last 15 minutes")
                # df_201_empty = no_run_patch(df, label_201)
                motor_sum = no_run_patch(_end_ts)
                _end_ts = motor_sum.index[-1]
                print(motor_sum)
                motor_sum.to_csv('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\Online_Data\\{}'
                                 .format(last_file))
                return _prev_file, _end_ts, function_call, motor_sum, last_file

            else:
                print("New file detected")

            # Case 3: 201 data exists (may not be complete)
            reformat_201 = df_201.pivot(index=' measurement_time(UTC)', columns=' device_name',
                                        values=' power(W)')
            reformat_201.columns.name = None
            # reformat_201 = reformat_201.reset_index()
            rename_dict = {' measurement_time(UTC)': '?"Time stamp"',
                           '201 Barrel Heat & Secondary': '201 Barrel Heat  Secondary   (kW)',
                           '201 Screw Retraction Motor': '201 Screw Retraction Motor   (kW)',
                           '201 Motor 1': '201 Motor 1   (kW)', '201 Motor 2': '201 Motor 2   (kW)',
                           '201 motor 3': '201 motor 3   (kW)'}
            reformat_201.rename(columns=rename_dict, inplace=True)

            # reformat_201.to_csv('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\Online_Data\\{}'
            #                     .format(last_file))
            # This section deals with missing columns
            renamed_label = ['201 Barrel Heat  Secondary   (kW)', '201 Screw Retraction Motor   (kW)',
                             '201 Motor 1   (kW)', '201 Motor 2   (kW)', '201 motor 3   (kW)',
                             ' 201 therm #1','201  Therm #2','201 Therm #3', '201  Therm #4']
            missing_col = [col for col in renamed_label if col not in reformat_201.columns]
            reformat_201.loc[:, missing_col] = 0
            reformat_201.index = pd.to_datetime(reformat_201.index)
            reformat_201.loc[:, 'Motor Sum'] = reformat_201.loc[:, ["201 Motor 1   (kW)",
                                                                    "201 Motor 2   (kW)",
                                                                    '201 motor 3   (kW)']].sum(axis=1)*0.001


            # First file with data (drop outliers & forward fill & fill gaps)
            if _end_ts is None:
                _end_ts = reformat_201.index[-1]
                guess_start = _end_ts - pd.Timedelta(minutes=14)
                reformat_201 = reformat_201.loc[reformat_201.index >= guess_start, :]
                fill_end = reformat_201.index[0] - pd.Timedelta(minutes=1)
                idx_rng = pd.date_range(start=guess_start, end=fill_end, freq='1min')
                df_zeros = pd.DataFrame(0, index=idx_rng, columns=reformat_201.columns)
                reformat_201 = pd.concat([df_zeros, reformat_201])
                reformat_201 = fill_gaps(reformat_201)
                reformat_201.fillna(0, inplace=True)
                print(reformat_201.head())
                print(reformat_201['Motor Sum'])
                reformat_201['Motor Sum'].to_csv('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\Online_Data\\{}'
                                 .format(last_file))
                # print(type(reformat_201['Motor Sum']))
                return _prev_file, _end_ts, function_call, reformat_201['Motor Sum'], last_file

            # Non-first file with data
            # Truncate any ts that is before the expected start time
            next_start_ts = _end_ts + pd.Timedelta(minutes=1)
            invalid_ts = reformat_201.index < next_start_ts
            reformat_201 = reformat_201.loc[~invalid_ts,:]

            # We want to make sure each file contains at least 15 entries, if less,
            # we will fill in the missing entries with zeros
            file_start_ts = reformat_201.index[0]
            if file_start_ts > _end_ts + pd.Timedelta(minutes=1):
                print("File start time is earlier than the end time, filling in zeros")
                fill_start_ts = _end_ts + pd.Timedelta(minutes=1)
                fill_end_ts = file_start_ts - pd.Timedelta(minutes=1)
                rng = pd.date_range(start=fill_start_ts, end=fill_end_ts, freq='1min')
                df_zeros = pd.DataFrame(0, index=rng, columns=reformat_201.columns)
                reformat_201 = pd.concat([df_zeros, reformat_201])

            # If there are timestamps that are not consecutive, fill in the gaps with zeros
            reformat_201 = fill_gaps(reformat_201)
            reformat_201.fillna(0, inplace=True)
            _end_ts = reformat_201.index[-1]

            print(reformat_201.head())
            motor_sum = reformat_201['Motor Sum']
            print(motor_sum)
            # print(type(motor_sum))
            motor_sum.to_csv('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\Online_Data\\{}'
                             .format(last_file))
        # Close SFTP session
        sftp.close()

    finally:
        # client.close()
        pass
    return _prev_file, _end_ts, function_call, motor_sum, last_file

if __name__ == "__main__":
    motor_sum_all = pd.Series()
    for i in range(401, 320, -1):
        current_info = get_file(prev_file, end_ts, 25)
        prev_file = current_info[0]
        end_ts = current_info[1]
        print('END_TS:', end_ts)
        motor_sum = current_info[3]
        if motor_sum is not None:
            if motor_sum_all.empty:
                motor_sum_all = motor_sum
            else:
                motor_sum_all = pd.concat([motor_sum_all, motor_sum])
        time.sleep(5)

    print(motor_sum_all)
    motor_sum_all.to_csv("demo.csv")
