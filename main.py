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
입력된 지진 정보(규모, 깊이, 위치)를 바탕으로 **쓰나미 발생 가능성**을 예측하는 시스템입니다.
왼쪽 사이드바에서 지진 정보를 조절하고 예측 결과를 확인하세요.
""")

# -----------------------------------------------------------------------------
# 2. 한글 폰트 설정
# 오류를 피하기 위해 koreanize_matplotlib 대신 직접 Matplotlib 폰트를 설정합니다.
# -----------------------------------------------------------------------------
try:
    # Streamlit Cloud 환경에서 NanumGothic을 사용하도록 설정
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    # NanumGothic이 없을 경우 fallback
    plt.rcParams['font.family'] = 'sans-serif' 
    st.warning("경고: Matplotlib 차트의 한글 폰트 설정에 문제가 있을 수 있습니다.")

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# -----------------------------------------------------------------------------
# 3. 데이터 로드 및 모델 학습 (캐싱 기능 사용)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("earthquake_data_tsunami.csv")
    except FileNotFoundError:
        st.error("❌ 'earthquake_data_tsunami.csv' 파일을 찾을 수 없습니다. 파일을 확인해주세요.")
        st.stop()
    return df

@st.cache_resource
def train_model(df):
    X = df[["magnitude", "depth", "latitude", "longitude"]]
    y = df["tsunami"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 학습: Random Forest Classifier 사용
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, acc, X.columns.tolist()

# 데이터 로드 및 학습 실행
df = load_data()
model, accuracy, feature_names = train_model(df)
st.success(f"✅ 모델 학습 완료! (테스트 데이터 정확도: {accuracy:.2f})")

st.divider()

# -----------------------------------------------------------------------------
# 4. 사이드바: 사용자 입력 (슬라이더)
# -----------------------------------------------------------------------------
st.sidebar.header("🌍 지진 정보 입력")
st.sidebar.write("지진 정보를 슬라이더로 조절하여 쓰나미 예측에 필요한 데이터를 입력합니다.")

# 데이터프레임의 min/max 값을 기반으로 슬라이더 범위 설정
magnitude_min, magnitude_max = df['magnitude'].min(), df['magnitude'].max()
depth_min, depth_max = df['depth'].min(), df['depth'].max()
latitude_min, latitude_max = df['latitude'].min(), df['latitude'].max()
longitude_min, longitude_max = df['longitude'].min(), df['longitude'].max()

# 슬라이더 설정
magnitude = st.sidebar.slider("지진 규모 (Magnitude)", 
                              min_value=magnitude_min, max_value=magnitude_max, 
                              value=min(6.5, magnitude_max), step=0.1)
depth = st.sidebar.slider("깊이 (Depth, km)", 
                          min_value=int(depth_min), max_value=int(depth_max), 
                          value=min(30, int(depth_max)), step=1)
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
# 5. 메인 화면: 예측 및 결과 출력
# -----------------------------------------------------------------------------

st.subheader("📍 지진 발생 위치")
st.map(input_data)

if st.button("🚨 쓰나미 발생 예측하기", type="primary"):
    with st.spinner('예측 중입니다...'):
        prediction = model.predict(input_data)[0]
        # 쓰나미 발생 확률 (클래스 1)
        probability = model.predict_proba(input_data)[0][1] 

    st.subheader("예측 결과")
    
    if prediction == 1:
        st.error(f"⚠️ **경고: 쓰나미 발생 위험이 높습니다!** (확률: {probability*100:.1f}%)")
        st.write("쓰나미 발생이 예측되었습니다. 아래 **대응책**을 확인하고 즉시 대피하세요!")
    else:
        st.success(f"✅ **안전: 쓰나미 발생 확률이 낮습니다.** (확률: {probability*100:.1f}%)")
        st.write("쓰나미 발생 확률은 낮지만, 지진 발생 시에는 항상 주의하고 재난 방송에 귀 기울여야 합니다.")

# -----------------------------------------------------------------------------
# 6. 모델 설명 및 대응책 섹션 추가 (사용자 요청 사항)
# -----------------------------------------------------------------------------

st.divider()

## 🛠️ 모델 분석 및 설명

with st.expander("모델에 대한 자세한 정보 보기"):
    st.markdown(
        """
        ### 사용 모델: Random Forest Classifier (랜덤 포레스트 분류기)
        
        **Random Forest**는 여러 개의 **결정 트리(Decision Tree)**를 만들고, 
        그 결정 트리의 예측 결과를 모아 다수결로 최종 예측을 결정하는 **앙상블(Ensemble) 학습** 기법입니다.
        
        #### 특징
        * **높은 정확도**: 다양한 트리의 의견을 종합하기 때문에 단일 모델보다 정확도가 높습니다.
        * **과적합(Overfitting) 방지**: 여러 무작위 표본을 사용하므로 데이터에 지나치게 맞춰지는 것을 방지합니다.
        * **변수 중요도 제공**: 각 특성(규모, 깊이 등)이 예측에 얼마나 중요한지 파악할 수 있습니다.
        """
    )
    
    # 중요 변수 시각화
    fig, ax = plt.subplots()
    importances = model.feature_importances_
    ax.bar(feature_names, importances, color='skyblue')
    ax.set_title("Feature Importance (특성이 쓰나미 예측에 미치는 영향)")
    ax.set_ylabel("중요도")
    st.pyplot(fig)


## 🚨 쓰나미 발생 시 대응책

with st.expander("쓰나미 발생 시 행동 요령"):
    st.markdown(
        """
        ### 🌊 쓰나미 경보 시 즉각적인 대피 요령
        
        쓰나미는 지진이 발생한 후 수분에서 수시간 내에 해안에 도달할 수 있습니다.
        
        1.  **즉시 대피**: 지진 발생 후 해안가에 있다면, 지진의 규모나 공식적인 경보 여부와 상관없이 즉시 가장 높은 곳으로 이동합니다.
        2.  **높은 곳으로**: 해안에서 멀리 떨어진 **고지대**나 튼튼한 **높은 건물 3층 이상**으로 대피합니다.
        3.  **이동 수단**: 차량 정체로 대피가 늦어질 수 있으므로, 가능한 한 **도보**로 대피합니다.
        4.  **정보 경청**: 정부, 언론, 재난 방송 등을 통해 공식적인 쓰나미 정보를 지속적으로 확인합니다.
        5.  **경보 해제까지**: 쓰나미는 한 번으로 끝나지 않고 여러 차례 반복될 수 있으므로, **경보가 공식적으로 해제될 때까지** 해안가로 돌아가지 않습니다.
        """
    )
