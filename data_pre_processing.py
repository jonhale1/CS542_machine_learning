import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler0 = StandardScaler()

#Importing sample data
news_data = pd.read_csv("/Users/jonathanhale/Documents/Courses/Machine Learning/example_data/news_sample (1).csv")
mkt_data = pd.read_csv("/Users/jonathanhale/Documents/Courses/Machine Learning/example_data/marketdata_sample (1).csv")

print(mkt_data.returnsOpenNextMktres10.describe())

def pre_process_data (news_data, mkt_data):

    # Delete unnecessary columns
    news_data.drop(['time',
                    'sourceTimestamp',
                    'sourceId'],
                   axis=1, inplace=True)

    mkt_data.drop(['time',
                   'close',
                   'open',
                   'returnsClosePrevMktres1',
                   'returnsOpenPrevMktres1',
                   'returnsClosePrevMktres10',
                   'returnsOpenPrevMktres10'],
                  axis=1, inplace=True)

    # Format time stamp data to numeric epoch time dates
    news_data['firstCreated'] = pd.to_datetime(news_data["firstCreated"])
    news_data['firstCreated'] = (pd.DataFrame((news_data['firstCreated'] -
                                               pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')))

    # Create a list of news_data columns to be processed and scaled
    news_headers = list(news_data)
    news_process_block = news_headers[15:32]
    news_data_list = ['bodySize',
                      'urgency',
                      'firstCreated',
                      'takeSequence',
                      'sentenceCount',
                      'companyCount',
                      'wordCount'] + news_process_block

    # Create a list of mkt_data columns to be processed and scaled
    mkt_headers = list(mkt_data)
    mkt_process_block = mkt_headers[4:8]
    mkt_data_list = ['volume'] + mkt_process_block

    # IMPLEMENT SCALING/NORMALIZATION OPTIONS:

    # Uncomment to implement MinMaxScaler between 0 and 1
    # news_data[news_data_list] = scaler1.fit_transform(news_data[news_data_list])
    # mkt_data[mkt_data_list] = scaler1.fit_transform(mkt_data[mkt_data_list])

    # Uncomment to implement StandardScaler (mean = 0, std = +/-1)
    news_data[news_data_list] = scaler0.fit_transform(news_data[news_data_list])
    mkt_data[mkt_data_list] = scaler0.fit_transform(mkt_data[mkt_data_list])

    return news_data, mkt_data

[processed_news_data, processed_mkt_data] = pre_process_data(news_data, mkt_data)

# print(processed_news_data)
# print(processed_mkt_data)




