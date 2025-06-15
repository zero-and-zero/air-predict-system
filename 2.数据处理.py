import pandas as pd
file_path = '空气质量-changsha_day.csv'
data = pd.read_csv(file_path)
data['日期'] = pd.to_datetime(data['日期'])
data['年'] = data['日期'].dt.year
data['月'] = data['日期'].dt.month
data['日'] = data['日期'].dt.day
data['星期'] = data['日期'].dt.dayofweek
data.drop_duplicates(inplace=True)
data['AQI_1天前'] = data['AQI指数'].shift(1);data['AQI_2天前'] = data['AQI指数'].shift(2)
data['PM2.5_1天前'] = data['PM2.5'].shift(1);data['PM2.5_2天前'] = data['PM2.5'].shift(2)
data['PM10_1天前'] = data['PM10'].shift(1);data['PM10_2天前'] = data['PM10'].shift(2)
data['So2_1天前'] = data['So2'].shift(1);data['So2_2天前'] = data['So2'].shift(2)
data['No2_1天前'] = data['No2'].shift(1);data['No2_2天前'] = data['No2'].shift(2)
data['O3_1天前'] = data['O3'].shift(1);data['O3_2天前'] = data['O3'].shift(2)
data['Co_1天前'] = data['Co'].shift(1);data['Co_2天前'] = data['Co'].shift(2)
data = data.dropna()
data.to_csv("dataset.csv", index=False)
print("数据处理完毕，存储位置dataset.csv")
print(data.head(7))