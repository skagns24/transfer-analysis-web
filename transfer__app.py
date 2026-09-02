import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# 웹페이지 기본 설정 (항상 최상단에 위치해야 함)
st.set_page_config(layout="wide", page_title="Semiconductor Data Analysis Tool")

# ---------------------------------------------------------
# 좌측 사이드바: 모드 선택 및 공통 설정
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 제어판 (Control Panel)")
    st.markdown("---")
    
    # 1. 분석 모드 선택 (라디오 버튼)
    analysis_mode = st.radio(
        "분석할 데이터 종류를 선택하세요.",
        ("1. Transfer 특성 분석", "2. Output 특성 분석", "3. TLM 분석 (추가 예정)")
    )
    st.markdown("---")
    
    # 2. 공통 X축 설정 (Transfer용)
    if analysis_mode == "1. Transfer 특성 분석":
        st.write("마우스로 슬라이더를 움직여 X축 범위를 조절하세요.")
        x_min = st.number_input("X축 최소값 (V)", value=-5.0, step=0.5)
        x_max = st.number_input("X축 최대값 (V)", value=1.0, step=0.5)

# 메인 타이틀
st.title("📊 반도체 소자 특성 통합 분석 웹앱")

# =====================================================================
# [모드 1] Transfer 특성 분석
# =====================================================================
if analysis_mode == "1. Transfer 특성 분석":
    st.markdown("측정 장비에서 얻은 CSV 파일들을 업로드하면, 통합 그래프와 최고 성능 하이라이트 표, 그리고 **단위 변환이 완료된 개별 데이터 엑셀**을 제공합니다.")
    
    uploaded_files = st.file_uploader("Transfer 데이터 CSV 파일들을 올려주세요.", type=['csv'], accept_multiple_files=True, key="transfer")

    if uploaded_files:
        st.success(f"총 {len(uploaded_files)}개의 파일이 업로드 되었습니다. Transfer 분석을 시작합니다!")

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
                    
                    label_name = f"{file_name}"
                    ax1.plot(vg, id_norm, label=label_name, color=c, linewidth=2)
                    ax1_twin.plot(vg, gm, color=c, linestyle='--', linewidth=2, alpha=0.6)
                    
                    ax2.plot(vg, id_abs, label=label_name, color=c, linewidth=2)
                    ax2.plot(vg, ig_abs, color=c, linestyle=':', linewidth=2, alpha=0.6)

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
                    
                    summary_list.append({
                        'Vd (V)': vd, 'Vth (Linear Ex) (V)': vth_linear, 'Vth (0.1 uA/mm) (V)': vth_cc_1,
                        'Vth (0.1 mA/mm) (V)': vth_cc_2, 'Min SS (mV/dec)': ss_min,
                        'Id max (mA/mm)': id_max, 'Gm max (mS/mm)': gm_max, 'On/Off Ratio': f"{on_off_ratio:.4E}"
                    })
                    
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
                    
                main_df = df[[col_vg, col_id_norm, col_ig_norm_abs, col_vd, 'Gm (mS/mm)']].reset_index(drop=True)
                sum_df = pd.DataFrame(summary_list).reset_index(drop=True)
                processed_dfs[file_name] = pd.concat([main_df, sum_df], axis=1)

            except Exception as e:
                st.error(f"오류 발생: [{file.name}] 파일 처리 중 문제가 생겼습니다. 에러 메시지: {e}")

        # 그래프 세팅
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
        st.download_button("📥 고화질 통합 그래프 다운로드 (.png)", data=buf_img.getvalue(), file_name="Combined_Transfer_Plot.png", mime="image/png")

        # 비교표 생성
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
            st.download_button("📥 비교 요약 엑셀 다운로드 (.xlsx)", data=buf_excel.getvalue(), file_name="Comparison_Summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
        # 개별 데이터 다운로드
        if processed_dfs:
            st.markdown("---")
            st.subheader("💾 개별 소자 단위 변환 데이터 다운로드")
            num_cols = max(1, min(len(processed_dfs), 4))
            cols = st.columns(num_cols) 
            for idx, (name, final_df) in enumerate(processed_dfs.items()):
                buf_raw = BytesIO()
                with pd.ExcelWriter(buf_raw, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False)
                with cols[idx % num_cols]:
                    st.download_button(label=f"📥 {name} 데이터", data=buf_raw.getvalue(), file_name=f"result_{name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_t_{name}")

# =====================================================================
# [모드 2] Output 특성 분석
# =====================================================================
elif analysis_mode == "2. Output 특성 분석":
    st.markdown("Output(출력) 특성 CSV 파일들을 업로드하세요. 게이트 전압($V_g$)별 드레인 전류 곡선을 시각화하고 단위가 변환된 엑셀을 제공합니다.")
    
    uploaded_files = st.file_uploader("Output 데이터 CSV 파일들을 올려주세요.", type=['csv'], accept_multiple_files=True, key="output")

    if uploaded_files:
        st.success(f"총 {len(uploaded_files)}개의 파일이 업로드 되었습니다. Output 분석을 시작합니다!")
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.linewidth'] = 1.5
        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = plt.colormaps['tab10'] 
        
        processed_dfs_output = {}

        for idx, file in enumerate(uploaded_files):
            file_name = file.name.replace('.csv', '')
            c = cmap(idx % 10) # 파일별로 다른 기본 색상 배정
            
            try:
                df = pd.read_csv(file, skiprows=1, encoding='cp949')
                col_vd = 'Drain Voltage (Vd)'
                col_vg = 'Gate Voltage (Vg)'
                col_id_raw = ' Drain Current (Id)'
                
                # Id 단위 변환 (A -> mA/mm)
                df['Id_norm'] = df[col_id_raw] * (1000 / 0.22)
                
                # 그래프 그리기 (같은 파일의 Vg 스텝들은 동일한 색상으로 그리되 범례에 Vg 표시)
                for vg_val, group in df.groupby(col_vg):
                    group = group.sort_values(col_vd)
                    vd_vals = group[col_vd].values
                    id_vals = group['Id_norm'].values
                    
                    # 선 굵기와 투명도를 살짝 조절해 구분감 형성
                    ax.plot(vd_vals, id_vals, color=c, linewidth=2, label=f"{file_name} ($V_g$={vg_val:g}V)")

                # 엑셀 데이터 가공 (Pivot)
                pivot_df = df.pivot(index=col_vd, columns=col_vg, values='Id_norm')
                new_columns = [f'Vg {vg:g}V Drain Current (mA/mm)' for vg in pivot_df.columns]
                pivot_df.columns = new_columns
                pivot_df = pivot_df.reset_index()
                
                processed_dfs_output[file_name] = pivot_df

            except Exception as e:
                st.error(f"오류 발생: [{file.name}] 파일 처리 중 문제가 생겼습니다. 에러 메시지: {e}")

        # 그래프 디자인
        ax.set_xlabel('Drain Voltage (V)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drain Current (mA/mm)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True, right=True)
        
        # 범례가 그래프를 가리지 않도록 바깥쪽(우측)으로 빼기
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
        plt.tight_layout()
        
        st.subheader("📈 통합 Output 커브 시각화")
        st.pyplot(fig)

        buf_img = BytesIO()
        fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 고화질 통합 그래프 다운로드 (.png)", data=buf_img.getvalue(), file_name="Combined_Output_Plot.png", mime="image/png")

        # 개별 데이터 다운로드
        if processed_dfs_output:
            st.markdown("---")
            st.subheader("💾 개별 소자 Output 데이터 다운로드 (Pivot 정리본)")
            num_cols = max(1, min(len(processed_dfs_output), 4))
            cols = st.columns(num_cols) 
            for idx, (name, final_df) in enumerate(processed_dfs_output.items()):
                buf_raw = BytesIO()
                with pd.ExcelWriter(buf_raw, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False)
                with cols[idx % num_cols]:
                    st.download_button(label=f"📥 {name} 데이터", data=buf_raw.getvalue(), file_name=f"result_{name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_o_{name}")

# =====================================================================
# [모드 3] TLM 분석
# =====================================================================
elif analysis_mode == "3. TLM 분석 (추가 예정)":
    st.info("TLM(Transmission Line Method) 분석 기능은 다음 업데이트에 바로 추가될 예정입니다!")
