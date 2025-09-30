from Utils import parse_data
import pandas as pd

df_217 = parse_data('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\1min_data\\data_217_1min')
df_201 = parse_data('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\1min_data\\M201')
df_221 = parse_data('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\1min_data\\M221')

# print(len(df_217))
# print(len(df_201))
# print(len(df_221))

df_217.to_csv('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\merged_data\\217_1min_v2.csv', index=False)
# df_201.to_csv('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\merged_data\\201_1min.csv', index=False)
# df_221.to_csv('C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\merged_data\\221_1min.csv', index=False)

