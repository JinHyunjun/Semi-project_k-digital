import requests
from bs4 import BeautifulSoup

# 웹 사이트 주소를 변수에 할당
url = "https://www.google.co.kr/travel/search?q=%EC%A0%9C%EC%A3%BC%EB%8F%84&ts=CAEaHBIaEhQKBwjoDxABGB4SBwjoDxABGB8YATICEAAqBwoFOgNLUlc&ved=0CAAQ5JsGahgKEwiwtt2tmYKEAxUAAAAAHQAAAAAQigE&ictx=3&qs=CAAgACgA&ap=MAA"

# requests 라이브러리를 이용해 웹 페이지 정보를 가져옴
response = requests.get(url)

# 웹 페이지의 HTML 정보를 BeautifulSoup 객체로 변환
soup = BeautifulSoup(response.text, 'html.parser')

# BeautifulSoup의 find_all 함수를 이용해 원하는 정보가 담긴 HTML 태그를 찾음
# 아래의 'your-tag'와 'your-class' 부분은 실제 사이트의 구조에 맞게 변경해야 합니다.
hotel_list = soup.find_all('div', {'class': 'pjDrrc'})

for hotel in hotel_list:
    # get_text() 함수를 이용해 HTML 태그 안의 텍스트 정보만 추출
    print(hotel.get_text())
