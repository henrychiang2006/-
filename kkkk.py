import yfinance as yf
import pandas as pd
import plotly.graph_objects as go


# Download the stock data
print('請輸入國家,美國,.TW:台灣上市,.TWO:台灣上櫃')
country_code = input('輸入國家地區')
ticker = input('輸入股票代碼')
stock_data = yf.download(ticker + country_code, period='1y', interval='1d')


# Clean and format the data
data = stock_data.reset_index()
data.columns = ['現在時間', '開盤價', '最高價', '最低價', '收價', '調整後收盤價', '成交量']
data['現在時間'] = pd.to_datetime(data['現在時間'].dt.strftime('%Y-%m-%d %H:%M'))
data['成交量'] //= 1000  # 把單位從成交股數換成成交張數


# Calculate the RSI
n = 14  # Number of periods to consider for the RSI calculation
data['Change'] = data['收價'].pct_change(
) * 100  # Calculate the change in price as a percentage
data['Gain'] = data['Change'].apply(lambda x: x
                                    if x > 0 else 0)  # Calculate the gains
data['Loss'] = data['Change'].apply(lambda x: abs(x)
                                    if x < 0 else 0)  # Calculate the losses


# Calculate the average gain and loss over the n periods
data['Avg Gain'] = data['Gain'].rolling(n).mean()
data['Avg Loss'] = data['Loss'].rolling(n).mean()


# Calculate the RSI as the ratio of the average gain to the average loss
data['RSI'] = data['Avg Gain'] / (data['Avg Loss'] + data['Avg Gain']) * 100


# Create the figure object
fig = go.Figure()


# Add the RSI trace
fig.add_trace(
    go.Scatter(x=data['現在時間'], y=data['RSI'], name='RSI', mode='lines'))


# Update the layout
fig.update_layout(title=ticker + country_code,
                  hovermode='x unified',
                  yaxis=dict(title='RSI'),
                  font=dict(size=20))
fig.update_yaxes(fixedrange=True)


config = dict({'scrollZoom': True})
fig.show(config=config)