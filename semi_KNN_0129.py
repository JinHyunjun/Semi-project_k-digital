from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# LabelEncoder 객체 생성
encoder = LabelEncoder()

# 데이터 불러오기
df = pd.read_csv("C:/Users/SAMSUNG/Desktop/DATA_2022년_국민여행조사_원자료.csv", encoding='cp949', low_memory=False)

# 피처와 타겟 설정
X = df[['A9D']]
y = df[['A9C']]

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
plt.scatter(df['A9D'], df['A9C'], alpha=0.5)
plt.title('Scatter Plot of Restaurant Cost vs Accommodation Cost per Person', fontsize=20)
plt.xlabel('Restaurant Cost per Person', fontsize=15)
plt.ylabel('Accommodation Cost per Person', fontsize=15)
plt.show()