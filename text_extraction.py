from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
import pandas as pd

news_data = pd.read_csv("/Users/jonathanhale/Documents/Courses/Machine Learning/example_data/news_sample (1).csv")

def text_feat_extract (news_data):
    """function to process news text data from specific csv file."""
    news_data.headline = news_data.headline.str.lower()
    news_data.subjects = news_data.subjects.str.lower()
    news_data.audiences = news_data.audiences.str.lower()

    n_features_h = 20
    n_features_sub = 10
    n_features_aud = 10

    hv_h = HashingVectorizer(n_features=n_features_h, strip_accents='ascii')
    hv_sub = HashingVectorizer(n_features=n_features_sub, strip_accents='ascii')
    hv_aud = HashingVectorizer(n_features=n_features_aud, strip_accents='ascii')

    news_data['headline'] = list(hv_h.transform(news_data.headline))
    column_names = ["h_feat{0}".format(i) for i in range(1, n_features_h+1)]
    h_features = np.array(list(map(lambda x: x.toarray(), news_data.headline)))
    h_features = h_features.reshape((h_features.shape[0],n_features_h))
    news_data[column_names] = pd.DataFrame(h_features, index=news_data.headline.index)
    del h_features
    del column_names
    news_data.drop(['headline'], axis=1, inplace=True)

    news_data['subjects'] = list(hv_sub.transform(news_data.subjects))
    column_names = ["sub_feat{0}".format(i) for i in range(1, n_features_sub+1)]
    sub_features = np.array(list(map(lambda x: x.toarray(), news_data.subjects)))
    sub_features = sub_features.reshape((sub_features.shape[0],n_features_sub))
    news_data[column_names] = pd.DataFrame(sub_features, index=news_data.subjects.index)
    del sub_features
    del column_names
    news_data.drop(['subjects'], axis=1, inplace=True)

    news_data['audiences'] = list(hv_aud.transform(news_data.audiences))
    column_names = ["aud_feat{0}".format(i) for i in range(1, n_features_aud+1)]
    aud_features = np.array(list(map(lambda x: x.toarray(), news_data.audiences)))
    aud_features = aud_features.reshape((aud_features.shape[0],n_features_aud))
    news_data[column_names] = pd.DataFrame(aud_features, index=news_data.audiences.index)
    del aud_features
    del column_names
    news_data.drop(['audiences'], axis=1, inplace=True)

    return news_data


text_feat_extract(news_data)


# FOR TESTING

print(news_data.h_feat1[0])
print('-'*20)
print(news_data.sub_feat2[5])
print('-'*20)
print(news_data.aud_feat3[2])

print(list(news_data.columns))