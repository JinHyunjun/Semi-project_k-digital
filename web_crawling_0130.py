import requests
from bs4 import BeautifulSoup

# f'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={query}'

query = '휴가지 추천'
url = f'https://search.naver.com/search.naver?where=view&sm=tab_jum&query={query}'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

results = soup.select('a.title_link')
for result in results:
    title = result.text
    link = result['href']
    print(f"{title} {link} \n")
    
