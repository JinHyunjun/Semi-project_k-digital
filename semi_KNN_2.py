from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# LabelEncoder 객체 생성
encoder = LabelEncoder()


df = pd.read_csv("C:/Users/SAMSUNG/Desktop/20240126145505_전국_202301-202312_데이터랩_다운로드/20240126145505_방문자 체류특성.csv", encoding='cp949', low_memory=False)

# 피처와 타겟 설정
X = df[['평균 체류시간']]
y = df[['평균 숙박일수']]


# 학습 데이터셋과 테스트 데이터셋으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결측치 제거
X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train.dropna()
y_test = y_test.dropna()


# KNN 모델 학습
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = knn.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))

print("RMSE: ", rmse)

plt.figure(figsize=(10, 8))
plt.scatter(df['평균 체류시간'], df['평균 숙박일수'], alpha=0.5)
plt.title('Average length of stay vs the average number of nights spent', fontsize=20)
plt.xlabel('Average length of stay', fontsize=15)
plt.ylabel('the average number of nights spent', fontsize=15)
plt.show()