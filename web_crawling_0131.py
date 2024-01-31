import requests
from bs4 import BeautifulSoup
import pandas as pd
from IPython.display import display

query = '국내 여행'
url = f'https://search.naver.com/search.naver?where=view&sm=tab_jum&query={query}'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

results = soup.select('a.title_link')

data = []
for result in results:
    title = result.text
    link = result['href']
    data.append([title, link])

df = pd.DataFrame(data, columns=['Title', 'Link'])
df.to_csv('holiday_recommendations.csv', index=False, encoding='utf-8-sig')
df = pd.read_csv('holiday_recommendations.csv')
display(df)