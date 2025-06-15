"""
数据来源网址
http://www.tianqihoubao.com/aqi/changsha-2025xx.html (xx代表月份)
"""


import pandas as pd
import requests
import warnings

warnings.filterwarnings("ignore")
for page in range(1, 13):
    if page < 10:
        url = f'http://www.tianqihoubao.com/aqi/changsha-20240{page}.html'
        res = requests.get(url)
        html = res.text
        df = pd.read_html(html, encoding='utf-8')[0]
        if page == 1:
            df.to_csv('空气质量-changsha_day.csv', mode='a+', index=False, header=False)
        else:
            df.iloc[1:, ::].to_csv('空气质量-changsha_day.csv', mode='a+', index=False, header=False)
        print(f"{page}月数据采集完毕")
    else:
        url = f'http://www.tianqihoubao.com/aqi/changsha-2024{page}.html'
        res = requests.get(url)
        html = res.text
        df = pd.read_html(html, encoding='utf-8')[0]
        df.iloc[1:, ::].to_csv('空气质量-changsha_day.csv', mode='a+', index=False, header=False)
        print(f"{page}月数据采集完毕")

print("长沙2024空气质量数据采集完毕！\n存储文件:空气质量-changsha_day.csv")

