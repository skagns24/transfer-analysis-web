import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="Transfer Data Analysis Tool")

st.title("📊 반도체 소자 Transfer 특성 분석 웹앱")
st.markdown("측정 장비에서 얻은 CSV 파일들을 드래그 앤 드롭으로 업로드하면, 통합 그래프와 최고 성능 하이라이트 표, 그리고 **단위 변환이 완료된 개별 데이터 엑셀**을 제공합니다.")

# ---------------------------------------------------------
# 좌측 사이드바: 그래프 축 범위 조절 컨트롤러
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 그래프 축 설정")
    st.write("마우스로 슬라이더를 움직여 X축 범위를 실시간으로 조절하세요.")
    x_min = st.number_input("X축 최소값 (V)", value=-5.0, step=0.5)
    x_max = st.number_input("X축 최대값 (V)", value=1.0, step=0.5)

# ---------------------------------------------------------
# 메인 화면: 파일 업로드 기능
# ---------------------------------------------------------
uploaded_files = st.file_uploader("측정 데이터 CSV 파일들을 여러 개 선택해서 올려주세요.", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    st.success(f"총 {len(uploaded_files)}개의 파일이 업로드 되었습니다. 분석을 시작합니다!")

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.linewidth'] = 1.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1_twin = ax1.twinx() 
    cmap = plt.colormaps['tab10'] 
    
    all_summaries = []
    processed_dfs = {}

    for idx, file in enumerate(uploaded_files):
        file_name = file.name.replace('.csv', '')
        c = cmap(idx % 10) 
        
        try:
            df = pd.read_csv(file, skiprows=1, encoding='cp949')
            col_vd = 'Drain Voltage (Vd)'
            col_vg = 'Gate Voltage (Vg)'
            col_id_raw = ' Drain Current (Id)'
            col_ig_raw = 'Gate Current (Ig)'
            
            col_id_norm = 'Drain Current (mA/mm)'
            df[col_id_norm] = df[col_id_raw] * (1000 / 0.22)
            col_ig_norm_abs = '|Gate Current| (mA/mm)'
            df[col_ig_norm_abs] = np.abs(df[col_ig_raw] * (1000 / 0.22))
            
            df['Gm (mS/mm)'] = np.nan
            summary_list = []
            
            for vd, group in df.groupby(col_vd):
                group = group.sort_values(col_vg)
                vg = group[col_vg].values
                id_norm = group[col_id_norm].values
                id_abs = np.abs(id_norm)
                ig_abs = group[col_ig_norm_abs].values
                
                gm = np.gradient(id_norm, vg)
                df.loc[group.index, 'Gm (mS/mm)'] = gm
                
                # 그래프 선 추가
                label_name = f"{file_name}"
                ax1.plot(vg, id_norm, label=label_name, color=c, linewidth=2)
                ax1_twin.plot(vg, gm, color=c, linestyle='--', linewidth=2, alpha=0.6)
                
                ax2.plot(vg, id_abs, label=label_name, color=c, linewidth=2)
                ax2.plot(vg, ig_abs, color=c, linestyle=':', linewidth=2, alpha=0.6)

                # 파라미터 추출
                gm_max = np.max(gm)
                id_max = np.max(id_norm)
                idx_max_gm = np.argmax(gm)
                vth_linear = vg[idx_max_gm] - (id_norm[idx_max_gm] / gm_max)
                
                def find_vth_cc(target_current):
                    crossings = np.where((id_abs[:-1] < target_current) & (id_abs[1:] >= target_current))[0]
                    if len(crossings) > 0:
                        i = crossings[-1] + 1
                        v1, v2 = vg[i-1], vg[i]
                        i1, i2 = id_abs[i-1], id_abs[i]
                        if i1 > 0 and i2 > 0:
                            return v1 + (v2 - v1) * (np.log10(target_current) - np.log10(i1)) / (np.log10(i2) - np.log10(i1))
                    return np.nan

                vth_cc_1 = find_vth_cc(1e-4)
                vth_cc_2 = find_vth_cc(0.1)
                
                log_id = np.log10(id_abs + 1e-15) 
                dlogId_dVg = np.gradient(log_id, vg)
                max_slope = np.max(dlogId_dVg)
                ss_min = 1000 / max_slope if max_slope > 0 else np.nan
                on_off_ratio = np.max(id_abs) / np.min(id_abs)
                
                # 개별 파일 요약 데이터 생성
                summary_list.append({
                    'Vd (V)': vd, 'Vth (Linear Ex) (V)': vth_linear, 'Vth (0.1 uA/mm) (V)': vth_cc_1,
                    'Vth (0.1 mA/mm) (V)': vth_cc_2, 'Min SS (mV/dec)': ss_min,
                    'Id max (mA/mm)': id_max, 'Gm max (mS/mm)': gm_max, 'On/Off Ratio': f"{on_off_ratio:.4E}"
                })
                
                # 전체 비교 엑셀용 데이터 수집
                all_summaries.append({
                    'File Name': file_name,
                    'Vth (Linear Ex) (V)': vth_linear,
                    'Vth (0.1 uA/mm) (V)': vth_cc_1,
                    'Vth (0.1 mA/mm) (V)': vth_cc_2,
                    'Id max (mA/mm)': id_max,
                    'Gm max (mS/mm)': gm_max,
                    'Min SS (mV/dec)': ss_min,
                    'On/Off Ratio': on_off_ratio
                })
                
            # 단위 변환된 메인 데이터와 요약을 병합하여 저장
            main_df = df[[col_vg, col_id_norm, col_ig_norm_abs, col_vd, 'Gm (mS/mm)']].reset_index(drop=True)
            sum_df = pd.DataFrame(summary_list).reset_index(drop=True)
            final_df = pd.concat([main_df, sum_df], axis=1)
            processed_dfs[file_name] = final_df

        except Exception as e:
            st.error(f"오류 발생: [{file.name}] 파일 처리 중 문제가 생겼습니다. 에러 메시지: {e}")

    # ---------------------------------------------------------
    # 1. 그래프 시각화 및 다운로드
    # ---------------------------------------------------------
    ax1.set_xlabel('Gate Voltage (V)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Drain Current (mA/mm)', fontsize=12, fontweight='bold')
    ax1_twin.set_ylabel('Transconductance (mS/mm)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True)
    ax1_twin.tick_params(axis='y', direction='in', labelsize=10, width=1.5)
    
    ax1.set_xlim(x_min, x_max)  
    ax1.legend(loc='upper left', frameon=True, fontsize=9, title="Solid: $I_d$, Dashed: $G_m$")

    ax2.set_yscale('log')
    ax2.set_xlabel('Gate Voltage (V)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Current (mA/mm)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True, right=True)
    
    ax2.set_xlim(x_min, x_max)
    ax2.legend(loc='lower right', frameon=True, fontsize=9, title="Solid: $|I_d|$, Dotted: $|I_g|$")

    plt.tight_layout()
    
    st.subheader("📈 통합 Transfer 커브 시각화")
    st.pyplot(fig)

    buf_img = BytesIO()
    fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
    st.download_button(
        label="📥 고화질 통합 그래프 다운로드 (.png)",
        data=buf_img.getvalue(),
        file_name="Combined_Transfer_Plot.png",
        mime="image/png"
    )

    # ---------------------------------------------------------
    # 2. 소자별 성능 비교 엑셀 표 생성 및 다운로드
    # ---------------------------------------------------------
    if len(all_summaries) > 0:
        st.markdown("---")
        st.subheader("📋 소자별 핵심 파라미터 비교 요약")
        comp_df = pd.DataFrame(all_summaries)
        
        def highlight_max(s):
            is_max = s == s.max(skipna=True)
            return ['color: red; font-weight: bold' if v else '' for v in is_max]
            
        def highlight_min(s):
            is_min = s == s.min(skipna=True)
            return ['color: red; font-weight: bold' if v else '' for v in is_min]

        styled_df = comp_df.style \
            .apply(highlight_max, subset=['Id max (mA/mm)', 'Gm max (mS/mm)', 'On/Off Ratio']) \
            .apply(highlight_min, subset=['Min SS (mV/dec)']) \
            .format({'On/Off Ratio': '{:.4E}'}) 
            
        st.dataframe(styled_df, use_container_width=True)

        buf_excel = BytesIO()
        with pd.ExcelWriter(buf_excel, engine='openpyxl') as writer:
            styled_df.to_excel(writer, index=False)
            
        st.download_button(
            label="📥 비교 요약 엑셀 다운로드 (.xlsx)",
            data=buf_excel.getvalue(),
            file_name="Comparison_Summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    # ---------------------------------------------------------
    # 3. 개별 소자별 Raw Data (mA/mm 변환 데이터) 다운로드
    # ---------------------------------------------------------
    if processed_dfs:
        st.markdown("---")
        st.subheader("💾 개별 소자 단위 변환 데이터 다운로드")
        st.write("단위 변환($I_d$, $|I_g|$ mA/mm), $G_m$ 미분 값 및 1줄 요약이 포함된 원본 엑셀 파일들입니다. 오리진(Origin)에서 개별 그래프를 그릴 때 사용하세요.")
        
        # 다운로드 버튼 가로 배치
        num_cols = max(1, min(len(processed_dfs), 4))
        cols = st.columns(num_cols) 
        
        for idx, (name, final_df) in enumerate(processed_dfs.items()):
            buf_raw = BytesIO()
            with pd.ExcelWriter(buf_raw, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False)
            
            with cols[idx % num_cols]:
                st.download_button(
                    label=f"📥 {name} 데이터 (.xlsx)",
                    data=buf_raw.getvalue(),
                    file_name=f"result_{name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_{name}"
                )
