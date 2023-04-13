import pandas as pd 
import pandas_Datareader as pdr #這裡有問題 沒有datareader
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
# Pandas和Pandas DataReader
# 下載股票價格數據
symbol = 'AAPL'  # Apple Inc.的代號
start_date = '2020-01-01'
end_date = '2022-04-13'
df = pdr.get_data_yahoo(symbol, start_date, end_date)

# 計算收益率
df['Returns'] = df['Adj Close'].pct_change()

# 設定圖形風格
plt.style.use('seaborn')

# 創建子圖表
fig, ax = plt.subplots(figsize=(12, 6))

# 繪製收益率折線圖
ax.plot(df['Returns'], label='Returns')

# 添加標籤和標題
ax.set_xlabel('Date')
ax.set_ylabel('Returns')
ax.set_title(f'Returns for {symbol}')

# 添加日期格式
date_form = DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_form)

# 添加網格線
ax.grid(True)

# 添加漲跌顏色
ax.fill_between(df.index, 0, df['Returns'], where=df['Returns']>0, interpolate=True, color='green', alpha=0.4)
ax.fill_between(df.index, 0, df['Returns'], where=df['Returns']<0, interpolate=True, color='red', alpha=0.4)

# 顯示圖形
plt.show()
