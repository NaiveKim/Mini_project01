import datetime

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import os
import glob

# 처음 시도

# df_culture_social = pd.read_csv('./crawling_data/crawling_data_Culture_78.csv')
# df_c = pd.DataFrame(df_culture_social, columns=df_culture_social.keys())
#
# df_economic = pd.read_csv('./crawling_data/crawling_data_Economic_110.csv')
# df_e = pd.DataFrame(df_economic, columns=df_economic.keys())
#
# df_world = pd.read_csv('./crawling_data/crawling_data_World_last.csv')
# df_w = pd.DataFrame(df_world, columns=df_world.keys())
#
# df_IT = pd.read_csv('./crawling_data/crawling_data_IT_last.csv')
# df_i = pd.DataFrame(df_IT, columns=df_IT.keys())
#
# category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
# df_section_titles = pd.DataFrame(titles, columns=['titles'])
# df_section_titles['category'] = category[i]
# df_All_titles = pd.concat([df_e, df_c,df_w, df_i, columns = ['category']], ignore_index=True)
# df_All_titles.to_csv('./crawling_data/title_crawling_data_Golden_shoe.csv'), index=False)

# 강사님 수업

data_path = glob.glob('./crawling_data/*')
print(data_path)

df = pd.DataFrame()
for path in data_path[1:]:
    df_temp = pd.read_csv(path)
    df = pd.concat([df, df_temp], ignore_index=True)
df.dropna(inplace = True)
df.reset_index(inplace=True, drop=True)
print(df.head())
print(df.tail())
print(df['category'].value_counts())
df.info()
df.to_csv('./crawling_data/naver_news_titles_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)

# 에러 없었을 때
# df = pd.read_csv('./crawling_data/crawling_data.csv')
# df_headline = pd.read_csv('./crawling_data/naver_headline_news_20220525.csv')
# df_all = pd.concat([df, df_headline])
#
# print(df_all.head())
# print(df_all.tail())
# print(df_all['category'].value_counts())
# df_all.info()
# df_all.to_csv('./crawling_data/naver_news_titles_{}.csv'.format(
#     datetime.datetime.now().strftime('%Y%m%d')), index=False)