import streamlit as st
from supabase import create_client, Client
import os

st.title("📂 연구 데이터 클라우드 아카이브")
st.markdown("연구실에서 측정한 장비 데이터(.csv, .txt)를 안전하게 영구 보관하고 언제든 다운로드하세요.")

# Supabase 연동 시도
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
    db_connected = True
except Exception as e:
    st.warning("⚠️ 아직 Supabase 클라우드 DB가 연결되지 않았습니다. Streamlit 설정에서 Secrets를 입력해주세요.")
    db_connected = False

if db_connected:
    st.markdown("---")
    st.subheader("📤 새로운 데이터 업로드")
    uploaded_file = st.file_uploader("보관할 데이터 파일 선택", type=["csv", "txt", "xlsx"])

    if uploaded_file is not None:
        if st.button("☁️ 클라우드에 안전하게 저장하기", use_container_width=True):
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

            with st.spinner('암호화하여 업로드 중...'):
                try:
                    # Supabase 'lab_data' 버킷에 업로드
                    res = supabase.storage.from_("lab_data").upload(
                        file=file_bytes,
                        path=file_name,
                        file_options={"content-type": uploaded_file.type}
                    )
                    st.success(f"✅ '{file_name}' 파일이 클라우드에 성공적으로 저장되었습니다!")
                except Exception as e:
                    st.error(f"업로드 실패: 이미 동일한 이름의 파일이 있거나 권한 오류입니다. ({e})")

    st.markdown("---")
    st.subheader("📥 보관된 내 데이터 목록")
    
    if st.button("🔄 파일 목록 불러오기"):
        with st.spinner('클라우드에서 목록을 가져오는 중...'):
            try:
                files = supabase.storage.from_("lab_data").list()
                if files:
                    for f in files:
                        if f['name'] == '.emptyFolderPlaceholder': continue
                        
                        col1, col2 = st.columns([8, 2])
                        col1.write(f"📄 **{f['name']}**")
                        
                        # 파일 다운로드를 위한 Public URL 생성
                        file_url = supabase.storage.from_("lab_data").get_public_url(f['name'])
                        col2.link_button("⬇️ 다운로드", file_url, use_container_width=True)
                else:
                    st.info("아직 저장된 파일이 없습니다.")
            except Exception as e:
                st.error(f"목록을 불러오는 중 오류가 발생했습니다: {e}")
