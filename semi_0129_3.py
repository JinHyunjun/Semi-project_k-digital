# 필요한 라이브러리 import
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# LabelEncoder 객체 생성
encoder = LabelEncoder()

# 데이터 불러오기
df = pd.read_csv("C:/Users/SAMSUNG/Desktop/DATA_2022년_국민여행조사_원자료.csv", encoding='cp949', low_memory=False)
# 필요한 칼럼만 추출
df = df[['D_TRA1_CASE', 'D_TRA1_COST', 'D_TRA1_ONE_COST', 'A9D', 'A9C']]

# 결측치 제거
df = df.dropna()

# 'A9C'와 'A9D' 칼럼이 문자열이라면 숫자로 변환
if df['A9C'].dtype == 'object':
    df['A9C'] = encoder.fit_transform(df['A9C'])

if df['A9D'].dtype == 'object':
    df['A9D'] = encoder.fit_transform(df['A9D'])

# 필요한 칼럼만으로 새로운 데이터프레임 생성
df_new = df[['A9C', 'A9D']]

# 데이터 정규화
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)

# k-means 클러스터링
kmeans = KMeans(n_clusters=3, init='k-means++').fit(df_scaled)

# 클러스터링 결과 추가
df['Cluster'] = kmeans.labels_

# scatter plot 그리기
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(df['A9C'], df['A9D'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('Accommodation Cost per Person', fontsize=15)
ax.set_ylabel('Restaurant Cost per Person', fontsize=15)
ax.set_title('K-means Clustering', fontsize=20)

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.show()

# '여행 유형' 칼럼을 숫자로 변환
df['D_TRA1_CASE'] = encoder.fit_transform(df['D_TRA1_CASE'])

# 상관계수 행렬 계산
corr_matrix = df.corr()

# Heatmap으로 상관계수 행렬을 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.title('Correlation Heatmap', fontsize=20)
plt.show()