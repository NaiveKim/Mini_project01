from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']

url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100'
# url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=101'
# url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=102'
# url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=103'
# url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=104'
# url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=105'


headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36'}

df_titles = pd.DataFrame()

for i in range(6):
    url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(i)
    resp = requests.get(url, headers=headers)
    # print(resp)
    # print(list(resp))
    # print(type(resp))

    soup = BeautifulSoup(resp.text, 'html.parser')
    # print(soup)

    title_tags = soup.select('.cluster_text_headline')
    print(title_tags[0].text)
    titles = []

    for title_tag in title_tags:
        title = re.compile('[^가-힣 ]').sub("",title_tag.text) #^가-힣 => 모든 한글, 띄어쓰기 필수
        # 해석 한글과 띄어쓰기를 제외한 모든 것을 뺀다.
        # 뺀것들에 빈칸을 넣어라
        titles.append(title)
    df_section_titles = pd.DataFrame(titles, columns = ['titles'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles, df_section_titles], axis='rows',
                          ignore_index=True)
print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index = False)
