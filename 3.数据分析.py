import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 创建图片保存目录
if not os.path.exists('analysis_plots'):
    os.makedirs('analysis_plots')
# 加载数据
file_path = 'dataset.csv'
data = pd.read_csv(file_path)
data['日期'] = pd.to_datetime(data['日期'])
# 相关性分析
numeric_columns = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numeric_columns].corr()
print(correlation_matrix['AQI指数'].sort_values(ascending=False))
# 时间序列图
plt.figure(figsize=(14, 7))
plt.plot(data['日期'], data['AQI指数'], label='AQI指数')
plt.xlabel('日期')
plt.ylabel('AQI指数')
plt.title('AQI指数随时间变化')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig('analysis_plots/time_series_aqi.png')  # 保存图片
plt.close()  # 关闭图形，避免重叠
# 箱线图
plt.figure(figsize=(14, 7))
sns.boxplot(x='月', y='AQI指数', data=data)
plt.xlabel('月份')
plt.ylabel('AQI指数')
plt.title('AQI指数的月度分布')
plt.savefig('analysis_plots/monthly_boxplot.png')  # 保存图片
plt.close()
# 热力图
plt.figure(figsize=(14, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('特征相关性热力图')
plt.savefig('analysis_plots/correlation_heatmap.png')  # 保存图片
plt.close()

print("分析完成，图片已保存至 analysis_plots 目录")