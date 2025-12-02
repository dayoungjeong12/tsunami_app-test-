import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# -----------------------------------------------------------------------------
# 1. 앱 설정 및 제목
# -----------------------------------------------------------------------------
st.set_page_config(page_title="쓰나미 예측 시스템", page_icon="🌊")

st.title("🌊 지진 발생 시 쓰나미 예측 시스템")
st.markdown("""
이 앱은 지진 데이터를 학습한 **Random Forest 모델**을 사용하여, 
입력된 지진 정보(규모, 깊이, 위치)를 바탕으로 **쓰나미 발생 가능성**을 예측합니다.
""")

# -----------------------------------------------------------------------------
# 2. 한글 폰트 설정 (⭐수정된 부분: koreanize_matplotlib 대신 직접 설정⭐)
# Streamlit Cloud 환경에서 Matplotlib 한글 깨짐을 방지합니다.
# -----------------------------------------------------------------------------
try:
    # Streamlit Cloud에서 NanumGothic을 사용하도록 설정
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    # NanumGothic이 없을 경우 fallback
    plt.rcParams['font.family'] = 'sans-serif' 
    st.warning("경고: Streamlit Cloud 환경에서 한글 폰트 설정에 문제가 있을 수 있습니다.")

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# -----------------------------------------------------------------------------
# 3. 데이터 로드 및 모델 학습 (캐싱 기능 사용)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # 데이터 불러오기 (파일 경로가 같은 디렉토리에 있다고 가정)
    try:
        df = pd.read_csv("earthquake_data_tsunami.csv")
    except FileNotFoundError:
        st.error("❌ 'earthquake_data_tsunami.csv' 파일을 찾을 수 없습니다.")
        st.stop()
    return df

@st.cache_resource
def train_model(df):
    # 필요한 열 선택
    X = df[["magnitude", "depth", "latitude", "longitude"]]
    y = df["tsunami"]
    
    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 정확도 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, acc, X.columns.tolist()

# 데이터 로드 및 학습 실행
df = load_data()
model, accuracy, feature_names = train_model(df)
st.success(f"✅ 모델 학습 완료! (모델 정확도: {accuracy:.2f})")

st.divider()

# -----------------------------------------------------------------------------
# 4. 사이드바: 사용자 입력 (슬라이더)
# -----------------------------------------------------------------------------
st.sidebar.header("🌍 지진 정보 입력")
st.sidebar.write("지진 정보를 슬라이더로 조절하세요.")

# 데이터프레임의 min/max 값을 기반으로 슬라이더 범위 설정
magnitude_min, magnitude_max = df['magnitude'].min(), df['magnitude'].max()
depth_min, depth_max = df['depth'].min(), df['depth'].max()
latitude_min, latitude_max = df['latitude'].min(), df['latitude'].max()
longitude_min, longitude_max = df['longitude'].min(), df['longitude'].max()

# 슬라이더 설정
magnitude = st.sidebar.slider("지진 규모 (Magnitude)", 
                              min_value=magnitude_min, max_value=magnitude_max, 
                              value=min(6.0, magnitude_max), step=0.1)
depth = st.sidebar.slider("깊이 (Depth, km)", 
                          min_value=int(depth_min), max_value=int(depth_max), 
                          value=min(50, int(depth_max)), step=1)
latitude = st.sidebar.slider("위도 (Latitude)", 
                             min_value=latitude_min, max_value=latitude_max, 
                             value=np.mean([latitude_min, latitude_max]), step=0.1)
longitude = st.sidebar.slider("경도 (Longitude)", 
                              min_value=longitude_min, max_value=longitude_max, 
                              value=np.mean([longitude_min, longitude_max]), step=0.1)

# 입력 데이터를 데이터프레임으로 변환
input_data = pd.DataFrame({
    'magnitude': [magnitude],
    'depth': [depth],
    'latitude': [latitude],
    'longitude': [longitude]
})

# -----------------------------------------------------------------------------
# 5. 메인 화면: 예측 및 시각화
# -----------------------------------------------------------------------------

# 5-1. 입력 위치 지도 표시
st.subheader("📍 지진 발생 위치")
st.map(input_data)

# 5-2. 예측하기 버튼 및 결과 출력
if st.button("🚨 쓰나미 발생 예측하기", type="primary"):
    with st.spinner('예측 중입니다...'):
        prediction = model.predict(input_data)[0]
        # 쓰나미 발생 확률 (클래스 1)
        probability = model.predict_proba(input_data)[0][1] 

    st.subheader("예측 결과")
    
    if prediction == 1:
        st.error(f"⚠️ **경고: 쓰나미 발생 위험이 높습니다!** (확률: {probability*100:.1f}%)")
        st.write("즉시 대피 정보를 확인하고 안전한 곳으로 이동하세요.")
    else:
        st.success(f"✅ **안전: 쓰나미 발생 확률이 낮습니다.** (확률: {probability*100:.1f}%)")
        st.write("지진 피해 상황을 주시하세요.")

# 5-3. 중요 변수 시각화
with st.expander("📊 모델이 중요하게 생각하는 특성 보기"):
    fig, ax = plt.subplots()
    importances = model.feature_importances_
    
    # 한글 제목/라벨 적용
    ax.bar(feature_names, importances, color='skyblue')
    ax.set_title("Feature Importance (특성이 쓰나미 예측에 미치는 영향)")
    ax.set_ylabel("중요도")
    
    st.pyplot(fig)
