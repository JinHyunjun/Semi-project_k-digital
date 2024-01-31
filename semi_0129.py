import pandas as pd
import numpy as np

# CSV 파일 로드
df = pd.read_csv("C:/Users/SAMSUNG/Desktop/DATA_2022년_국민여행조사_원자료.csv", encoding='cp949', low_memory=False)

# 숫자가 아닌 칼럼 제거
df = df.select_dtypes(include=[np.number])

# 상관계수 계산
corr_matrix = df.corr()

print(corr_matrix)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 독립 변수와 종속 변수 설정
X = df.drop('target_column', axis=1)
y = df['target_column']

# 알고리즘 설정
models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(),
}

# 하이퍼파라미터 설정
params = {
    'RandomForestClassifier': {'n_estimators': [50, 100, 200], 'max_depth': [None, 30, 15, 5]},
    'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
}

# 최적의 모델 찾기
for model_name in models:
    clf = GridSearchCV(models[model_name], params[model_name], cv=5)
    clf.fit(X, y)
    print(f'Best parameters for {model_name}:', clf.best_params_)
