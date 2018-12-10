import pandas as pd
from datetime import datetime
from iexfinance.stocks import get_historical_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



output = pd.read_csv('submission.csv')

start = datetime(2013, 12, 5)
end = datetime(2018, 12, 5)
market_data = get_historical_data("SPY", start, end, output_format='pandas')
market_data['Scaled_SPY_Daily_Return'] = (market_data.open/market_data.close)-1
market_data.drop(['volume', 'high', 'low', 'open', 'close'], axis=1, inplace=True)

new = output["assetCode"].str.split(".", n=1, expand=True)
output['assetCode'] = new
group_by_day = output.groupby('time').agg('mean')
group_by_day.rename(columns={'confidenceValue': 'Avg_confidence_value'}, inplace=True)

max_cv = group_by_day.Avg_confidence_value.max()
print(max_cv)
min_cv = group_by_day.Avg_confidence_value.min()


scaler = MinMaxScaler(feature_range=(min_cv, max_cv))
market_data[['Scaled_SPY_Daily_Return']] = scaler.fit_transform(market_data[['Scaled_SPY_Daily_Return']])

combined = group_by_day.join(market_data, how='inner')

combined.plot()
plt.title('S&P500 Return vs. Avg Confidence Value', fontweight="bold")
plt.xlabel('Date: 01/03/2017 - 12/05/2018', fontweight="bold")
plt.ylabel('Avg. Confidence Value/Scaled SPY return)', fontweight="bold")
plt.show()
