import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

st.set_page_config(page_title="Namhoon Lab OS", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    div.stButton > button {
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        font-weight: bold;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 2px 5px 10px rgba(0,0,0,0.5);
        color: #4da6ff;
    }
</style>
""", unsafe_allow_html=True)

# 비밀번호 해싱 (2735)
hashed_passwords = Hasher(['2735']).generate()

credentials = {
    'usernames': {
        'aaa': {
            'email': 'namhoon@lab.com',
            'name': 'Nam-hoon Kim',
            'password': hashed_passwords[0]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    'lab_dashboard_cookie',
    'secret_signature_key',
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login('main')

if authentication_status == False:
    st.error('❌ 아이디 또는 비밀번호가 틀렸습니다.')
elif authentication_status == None:
    st.warning('🔒 연구실 OS에 접근하려면 로그인하세요. (ID: aaa, PW: 2735)')
elif authentication_status:
    col1, col2 = st.columns([8, 2])
    with col1:
        st.title(f"👋 Welcome to Namhoon's Private Lab OS")
    with col2:
        authenticator.logout('로그아웃', 'main')

    st.markdown("---")
    st.markdown("""
    > *"데이터는 직관보다 강하다. 하나의 완벽한 이론보다, 천 번의 흔들림 없는 데이터가 공정을 지배한다."*  
    > **- Semiconductor Metrology & Process Engineering -**
    """)
    st.markdown("### 📌 연구실 주요 기능 안내")
    st.info("👈 **좌측 사이드바 메뉴를 클릭하여 작업을 시작하세요.**")
