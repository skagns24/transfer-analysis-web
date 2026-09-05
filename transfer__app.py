import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from io import BytesIO
from scipy.signal import savgol_filter
import plotly.graph_objects as go # [추가됨] 3D 지형도 렌더링용

# 웹페이지 기본 설정 (최상단 배치)
st.set_page_config(layout="wide", page_title="Semiconductor Data Analysis Tool")

# ---------------------------------------------------------
# Session State 초기화
# ---------------------------------------------------------
if 'transfer_files_data' not in st.session_state:
    st.session_state['transfer_files_data'] = []
if 'output_files_data' not in st.session_state:
    st.session_state['output_files_data'] = []
if 'tlm_files_data' not in st.session_state:
    st.session_state['tlm_files_data'] = []
if 'afm_files_data' not in st.session_state:
    st.session_state['afm_files_data'] = []
    
if 't_uploader_key' not in st.session_state:
    st.session_state['t_uploader_key'] = 0
if 'o_uploader_key' not in st.session_state:
    st.session_state['o_uploader_key'] = 0
if 'tlm_uploader_key' not in st.session_state:
    st.session_state['tlm_uploader_key'] = 0
if 'afm_uploader_key' not in st.session_state:
    st.session_state['afm_uploader_key'] = 0

# ---------------------------------------------------------
# 좌측 사이드바: 분석 모드 선택 및 축 범위 컨트롤러
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 제어판 (Control Panel)")
    st.markdown("---")
    
    analysis_mode = st.radio(
        "분석할 데이터 종류를 선택하세요.",
        ("1. Transfer 특성 분석", "2. Output 특성 분석", "3. TLM 특성 분석", "4. AFM 표면 분석"),
        key="selected_mode"
    )
    st.markdown("---")
    
    # 1. Transfer 축 범위 설정
    if analysis_mode == "1. Transfer 특성 분석":
        st.header("⚙️ Transfer 축 범위 설정")
        t_x_auto = st.checkbox("X축 자동 조절", value=True, key="t_x_auto")
        if not t_x_auto:
            t_x_min = st.number_input("X축 최소값 (V)", value=-5.0, step=0.5, key="t_xmin")
            t_x_max = st.number_input("X축 최대값 (V)", value=1.0, step=0.5, key="t_xmax")
        
        t_y_lin_auto = st.checkbox("Linear Y축 자동 조절", value=True, key="t_ylin_auto")
        if not t_y_lin_auto:
            t_y_lin_min = st.number_input("Linear Y축 최소값 (mA/mm)", value=0.0, step=5.0, key="t_ylin_min")
            t_y_lin_max = st.number_input("Linear Y축 최대값 (mA/mm)", value=200.0, step=10.0, key="t_ylin_max")
            
        t_y_log_auto = st.checkbox("Log Y축 자동 조절", value=True, key="t_ylog_auto")
        if not t_y_log_auto:
            t_y_log_min_exp = st.number_input("Log Y축 최소값 (10^x)", value=-8.0, step=1.0, key="t_ylog_min")
            t_y_log_max_exp = st.number_input("Log Y축 최대값 (10^x)", value=2.0, step=1.0, key="t_ylog_max")

        st.markdown("---")
        st.subheader("📌 Gm 노이즈 필터링 (Savitzky-Golay)")
        apply_smoothing = st.checkbox("Gm 노이즈 필터링 적용", value=False, key="apply_smoothing", help="체크하면 엑셀 다운로드 시 'Smoothed Gm' 열이 추가됩니다.")
        
        if apply_smoothing:
            window_length = st.slider("필터 강도 (Window Length, 홀수)", min_value=3, max_value=51, value=11, step=2, key="sg_window")
            poly_order = st.slider("다항식 차수 (Poly Order)", min_value=1, max_value=5, value=2, key="sg_poly")

    # 2. Output 축 범위 설정
    elif analysis_mode == "2. Output 특성 분석":
        st.header("⚙️ Output 축 범위 설정")
        o_x_auto = st.checkbox("X축(Vd) 자동 조절", value=True, key="o_x_auto")
        if not o_x_auto:
            o_x_min = st.number_input("X축(Vd) 최소값 (V)", value=0.0, step=0.5, key="o_xmin")
            o_x_max = st.number_input("X축(Vd) 최대값 (V)", value=5.0, step=0.5, key="o_xmax")
            
        o_y_auto = st.checkbox("Y축(Id) 자동 조절", value=True, key="o_y_auto")
        if not o_y_auto:
            o_y_min = st.number_input("Y축(Id) 최소값 (mA/mm)", value=0.0, step=5.0, key="o_ymin")
            o_y_max = st.number_input("Y축(Id) 최대값 (mA/mm)", value=200.0, step=10.0, key="o_ymax")
            
        st.subheader("📌 온저항(Ron) 분석 설정")
        target_vg = st.number_input("Ron 추출 기준 Vg (V)", value=3.0, step=1.0, key="target_vg")

    # 3. TLM 설정
    elif analysis_mode == "3. TLM 특성 분석":
        st.header("⚙️ TLM 설정")
        t_w_um = st.number_input("전극 폭 (W, um)", value=220.0, step=10.0, key="tlm_w")
        
        st.subheader("📌 Ohmic I-V X축 설정")
        tlm_x_auto = st.checkbox("X축 자동 조절", value=True, key="tlm_x_auto")
        if not tlm_x_auto:
            tlm_x_min = st.number_input("X축 최소값 (V)", value=-5.0, step=0.5, key="tlm_xmin")
            tlm_x_max = st.number_input("X축 최대값 (V)", value=5.0, step=0.5, key="tlm_xmax")
            
        st.subheader("📌 Ohmic I-V Y축 설정")
        tlm_y_auto = st.checkbox("Y축 자동 조절", value=True, key="tlm_y_auto")
        if not tlm_y_auto:
            tlm_y_min = st.number_input("Y축 최소값 (mA)", value=-15.0, step=1.0, key="tlm_ymin")
            tlm_y_max = st.number_input("Y축 최대값 (mA)", value=15.0, step=1.0, key="tlm_ymax")

    # 4. AFM 설정
    elif analysis_mode == "4. AFM 표면 분석":
        st.header("⚙️ AFM 3D 렌더링 설정")
        color_theme = st.selectbox("컬러 맵 선택", ["earth", "hot", "viridis", "plasma", "inferno", "magma", "cividis"], index=0)
        st.info("💡 텍스트(TXT) 파일에 포함된 X, Y, Z (nm) 데이터를 기반으로 3D 지형도와 거칠기를 계산합니다.")

    # ==========================================
    # 🤖 사이드바 하단: 외부 Gemini 창 열기 버튼
    # ==========================================
    st.markdown("---")
    st.header("🤖 AI 연구 어시스턴트")
    st.write("분석 결과 해석이 필요할 때 아래 버튼을 눌러 평소 사용하시던 Gemini 창을 띄워보세요.")
    st.link_button("💬 Gemini 새 창으로 열기", "https://gemini.google.com/app", use_container_width=True)

# 메인 타이틀
st.title("📊 반도체 소자 특성 통합 분석 웹앱")

# =====================================================================
# [모드 1] Transfer 특성 분석
# =====================================================================
if analysis_mode == "1. Transfer 특성 분석":
    st.markdown("Transfer CSV 파일들을 업로드하세요. 통합 그래프, 파라미터 비교표, **개별 단위 변환 엑셀**을 제공합니다.")
    
    uploaded_transfer = st.file_uploader(
        "Transfer 데이터 CSV 파일들을 올려주세요.", 
        type=['csv'], 
        accept_multiple_files=True, 
        key=f"transfer_uploader_{st.session_state['t_uploader_key']}"
    )
    
    if uploaded_transfer:
        st.session_state['transfer_files_data'] = uploaded_transfer

    files_to_process = st.session_state['transfer_files_data']

    if files_to_process:
        col1, col2 = st.columns([8, 2])
        with col1:
            st.success(f"현재 {len(files_to_process)}개의 Transfer 파일이 유지/분석 중입니다.")
        with col2:
            if st.button("🗑️ 전체 파일 삭제", use_container_width=True, key="del_t"):
                st.session_state['transfer_files_data'] = []
                st.session_state['t_uploader_key'] += 1 
                st.rerun() 

        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.linewidth'] = 1.5
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1_twin = ax1.twinx() 
        cmap = plt.colormaps['tab10'] 
        
        all_summaries = []
        processed_dfs = {}

        for idx, file in enumerate(files_to_process):
            file_name = file.name.replace('.csv', '')
            c = cmap(idx % 10) 
            
            try:
                file.seek(0)
                df = pd.read_csv(file, skiprows=1, encoding='cp949')
                col_vd = 'Drain Voltage (Vd)'
                col_vg = 'Gate Voltage (Vg)'
                col_id_raw = ' Drain Current (Id)'
                col_ig_raw = 'Gate Current (Ig)'
                
                col_id_norm = 'Drain Current (mA/mm)'
                df[col_id_norm] = df[col_id_raw] * (1000 / 0.22)
                col_ig_norm_abs = '|Gate Current| (mA/mm)'
                df[col_ig_norm_abs] = np.abs(df[col_ig_raw] * (1000 / 0.22))
                
                summary_list = []
                
                for vd, group in df.groupby(col_vd):
                    group = group.sort_values(col_vg)
                    vg = group[col_vg].values
                    id_norm = group[col_id_norm].values
                    id_abs = np.abs(id_norm)
                    ig_abs = group[col_ig_norm_abs].values
                    
                    # Gm 원본 추출
                    gm_raw = np.gradient(id_norm, vg)
                    df.loc[group.index, 'Raw Gm (mS/mm)'] = gm_raw
                    
                    # 스무딩(필터링) 적용 여부에 따른 최종 Gm 선택 및 엑셀 데이터 저장
                    if apply_smoothing and len(gm_raw) > window_length:
                        actual_poly = min(poly_order, window_length - 1)
                        gm_final = savgol_filter(gm_raw, window_length, actual_poly)
                        df.loc[group.index, 'Smoothed Gm (mS/mm)'] = gm_final
                    else:
                        gm_final = gm_raw
                    
                    # 그래프 시각화
                    label_name = f"{file_name}"
                    ax1.plot(vg, id_norm, label=label_name, color=c, linewidth=2)
                    
                    if apply_smoothing and len(gm_raw) > window_length:
                        ax1_twin.plot(vg, gm_raw, color=c, linestyle=':', linewidth=1.5, alpha=0.3) # 원본(흐리게)
                        ax1_twin.plot(vg, gm_final, color=c, linestyle='--', linewidth=2.5, alpha=0.8) # 필터링됨(진하게)
                    else:
                        ax1_twin.plot(vg, gm_final, color=c, linestyle='--', linewidth=2, alpha=0.6)
                    
                    ax2.plot(vg, id_abs, label=label_name, color=c, linewidth=2)
                    ax2.plot(vg, ig_abs, color=c, linestyle=':', linewidth=2, alpha=0.6)

                    # 파라미터 추출
                    gm_max = np.max(gm_final)
                    id_max = np.max(id_norm)
                    idx_max_gm = np.argmax(gm_final)
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
                    
                # 엑셀 다운로드용 데이터프레임 정리 (옵션에 따라 Smoothed Gm 포함)
                cols_to_export = [col_vg, col_id_norm, col_ig_norm_abs, col_vd, 'Raw Gm (mS/mm)']
                if apply_smoothing and 'Smoothed Gm (mS/mm)' in df.columns:
                    cols_to_export.append('Smoothed Gm (mS/mm)')
                    
                main_df = df[cols_to_export].reset_index(drop=True)
                sum_df = pd.DataFrame(summary_list).reset_index(drop=True)
                processed_dfs[file_name] = pd.concat([main_df, sum_df], axis=1)

            except Exception as e:
                st.error(f"오류 발생: [{file.name}] 파일 처리 중 문제가 생겼습니다. 에러 메시지: {e}")

        ax1.set_xlabel('Gate Voltage (V)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Drain Current (mA/mm)', fontsize=12, fontweight='bold')
        ax1_twin.set_ylabel('Transconductance (mS/mm)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True)
        ax1_twin.tick_params(axis='y', direction='in', labelsize=10, width=1.5)
        
        if not t_x_auto:
            ax1.set_xlim(t_x_min, t_x_max)  
            ax2.set_xlim(t_x_min, t_x_max)
        if not t_y_lin_auto:
            ax1.set_ylim(t_y_lin_min, t_y_lin_max)
        if not t_y_log_auto:
            ax2.set_ylim(10**t_y_log_min_exp, 10**t_y_log_max_exp)

        # 범례 동적 변경
        legend_title = "Solid: $I_d$, Dashed: Smoothed $G_m$, Dotted: Raw $G_m$" if apply_smoothing else "Solid: $I_d$, Dashed: $G_m$"
        ax1.legend(loc='upper left', frameon=True, fontsize=9, title=legend_title)

        ax2.set_yscale('log')
        ax2.set_xlabel('Gate Voltage (V)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Current (mA/mm)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True, right=True)
        ax2.legend(loc='lower right', frameon=True, fontsize=9, title="Solid: $|I_d|$, Dotted: $|I_g|$")

        plt.tight_layout()
        st.subheader("📈 통합 Transfer 커브 시각화")
        st.pyplot(fig)

        buf_img = BytesIO()
        fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 고화질 통합 그래프 다운로드 (.png)", data=buf_img.getvalue(), file_name="Combined_Transfer_Plot.png", mime="image/png")

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
    st.markdown("Output CSV 파일들을 업로드하세요. 게이트 전압($V_g$)별 드레인 전류 곡선과 타겟 $V_g$의 온저항($R_{on}$)을 자동 추출합니다.")
    
    uploaded_output = st.file_uploader(
        "Output 데이터 CSV 파일들을 올려주세요.", 
        type=['csv'], 
        accept_multiple_files=True, 
        key=f"output_uploader_{st.session_state['o_uploader_key']}"
    )

    if uploaded_output:
        st.session_state['output_files_data'] = uploaded_output

    files_to_process_out = st.session_state['output_files_data']

    if files_to_process_out:
        col1, col2 = st.columns([8, 2])
        with col1:
            st.success(f"현재 {len(files_to_process_out)}개의 Output 파일이 유지/분석 중입니다.")
        with col2:
            if st.button("🗑️ 전체 파일 삭제", use_container_width=True, key="del_o"):
                st.session_state['output_files_data'] = []
                st.session_state['o_uploader_key'] += 1 
                st.rerun() 
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.linewidth'] = 1.5
        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = plt.colormaps['tab10'] 
        
        processed_dfs_output = {}
        output_summaries = []

        for idx, file in enumerate(files_to_process_out):
            file_name = file.name.replace('.csv', '')
            c = cmap(idx % 10) 
            
            try:
                file.seek(0)
                df = pd.read_csv(file, skiprows=1, encoding='cp949')
                col_vd = 'Drain Voltage (Vd)'
                col_vg = 'Gate Voltage (Vg)'
                col_id_raw = ' Drain Current (Id)'
                
                df['Id_norm'] = df[col_id_raw] * (1000 / 0.22)
                
                unique_vgs = df[col_vg].unique()
                closest_vg = unique_vgs[np.argmin(np.abs(unique_vgs - target_vg))]
                
                file_ron = np.nan
                file_slope = np.nan
                file_intercept = np.nan
                
                for vg_val, group in df.groupby(col_vg):
                    group = group.sort_values(col_vd)
                    vd_vals = group[col_vd].values
                    id_vals = group['Id_norm'].values
                    
                    ax.plot(vd_vals, id_vals, color=c, linewidth=2, label=f"{file_name} ($V_g$={vg_val:g}V)")

                    if vg_val == closest_vg:
                        mask = (vd_vals >= 0.0) & (vd_vals <= 0.5)
                        if np.sum(mask) >= 2:
                            slope, intercept = np.polyfit(vd_vals[mask], id_vals[mask], 1)
                            if slope > 0:
                                file_slope = slope
                                file_intercept = intercept
                                file_ron = (1 / slope) * 1000 
                                
                if not np.isnan(file_slope):
                    max_i_global = df['Id_norm'].max()
                    v_max_line = (max_i_global * 1.1 - file_intercept) / file_slope if file_slope > 0 else max(df[col_vd])
                    v_line = np.linspace(0, min(v_max_line, max(df[col_vd])), 10)
                    i_line = file_slope * v_line + file_intercept
                    
                    line_color = 'red' if len(files_to_process_out) == 1 else c
                    ax.plot(v_line, i_line, color=line_color, linestyle='--', linewidth=1.5, label=f"Linear fit ($R_{{on}}$: {file_ron:.1f} $\Omega\cdot mm$)")

                pivot_df = df.pivot(index=col_vd, columns=col_vg, values='Id_norm')
                new_columns = [f'Vg {vg:g}V Drain Current (mA/mm)' for vg in pivot_df.columns]
                pivot_df.columns = new_columns
                pivot_df = pivot_df.reset_index()
                
                processed_dfs_output[file_name] = pivot_df
                
                output_summaries.append({
                    'Sample Name': file_name,
                    'Analyzed Vg (V)': closest_vg,
                    'On-Resistance, Ron (Ohm.mm)': file_ron
                })

            except Exception as e:
                st.error(f"오류 발생: [{file.name}] 처리 실패 ({e})")

        ax.set_xlabel('Drain Voltage (V)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drain Current Density (mA/mm)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True, right=True)
        
        if not o_x_auto:
            ax.set_xlim(o_x_min, o_x_max)
        if not o_y_auto:
            ax.set_ylim(o_y_min, o_y_max)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
        plt.tight_layout()
        
        st.subheader("📈 통합 Output 커브 시각화")
        st.pyplot(fig)

        buf_img = BytesIO()
        fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 고화질 통합 그래프 다운로드 (.png)", data=buf_img.getvalue(), file_name="Combined_Output_Plot.png", mime="image/png")

        if output_summaries:
            st.markdown("---")
            st.subheader("📋 소자별 On-Resistance ($R_{on}$) 요약 비교")
            out_sum_df = pd.DataFrame(output_summaries)
            
            def highlight_min_ron(s):
                is_min = s == s.min(skipna=True)
                return ['color: red; font-weight: bold' if v else '' for v in is_min]
                
            styled_out_sum = out_sum_df.style.apply(highlight_min_ron, subset=['On-Resistance, Ron (Ohm.mm)'])
            st.dataframe(styled_out_sum, use_container_width=True)
            
            buf_out_sum = BytesIO()
            with pd.ExcelWriter(buf_out_sum, engine='openpyxl') as writer:
                out_sum_df.to_excel(writer, index=False)
            st.download_button("📥 Output (Ron) 요약 엑셀 다운로드", data=buf_out_sum.getvalue(), file_name="Output_Ron_Summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
# [모드 3] TLM 특성 분석
# =====================================================================
elif analysis_mode == "3. TLM 특성 분석":
    st.markdown("TLM 측정을 위해 얻은 CSV 파일들을 업로드하세요. **파일 이름에 전극 간격(10, 20, 30, 40, 80) 숫자가 반드시 포함되어야 자동으로 인식됩니다.**")
    st.info("💡 **샘플 단위 분리 그래프 안내:**\n\n여러 샘플이 섞이는 것을 방지하려면 파일 이름 앞에 **샘플명**을 적어주세요. (예: `SampleA_10.csv`, `SampleA_20.csv` / `SampleB_10.csv`). 프로그램이 이름 앞부분이 같은 파일끼리 묶어서 **각 샘플마다 독립적인 그래프와 결과 표를 별도로 생성**해 줍니다!")
    
    uploaded_tlm = st.file_uploader(
        "TLM 데이터 CSV 파일들을 올려주세요.", 
        type=['csv'], 
        accept_multiple_files=True, 
        key=f"tlm_uploader_{st.session_state['tlm_uploader_key']}"
    )

    if uploaded_tlm:
        st.session_state['tlm_files_data'] = uploaded_tlm

    files_to_process_tlm = st.session_state['tlm_files_data']

    if files_to_process_tlm:
        col1, col2 = st.columns([8, 2])
        with col1:
            st.success(f"현재 {len(files_to_process_tlm)}개의 TLM 파일이 유지/분석 중입니다.")
        with col2:
            if st.button("🗑️ 전체 파일 삭제", use_container_width=True, key="del_tlm"):
                st.session_state['tlm_files_data'] = []
                st.session_state['tlm_uploader_key'] += 1
                st.rerun()
                
        tlm_groups = {}
        for file in files_to_process_tlm:
            basename = file.name
            match = re.search(r'(?<!\d)(10|20|30|40|80)(?!\d)', basename)
            if match:
                gap = int(match.group(1))
                idx = match.start()
                prefix = basename[:idx]
                group_name = re.sub(r'[\_\-\s]+$', '', prefix)
                
                if not group_name:
                    group_name = "기본 샘플 (이름 없음)"
                
                if group_name not in tlm_groups:
                    tlm_groups[group_name] = {}
                tlm_groups[group_name][gap] = file
                
        if not tlm_groups:
            st.warning("업로드된 파일 중 이름에 10, 20, 30, 40, 80이 포함된 CSV 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
        else:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['axes.linewidth'] = 1.5
            
            all_tlm_summaries = []
            
            for group_name, gap_dict in tlm_groups.items():
                gaps_found = sorted(gap_dict.keys())
                
                if not all(g in gaps_found for g in [10, 20, 30, 40]):
                    st.warning(f"[{group_name}] 샘플은 TLM 분석을 위해 최소 10, 20, 30, 40 파일이 모두 필요합니다. (현재 업로드된 간격: {gaps_found}um)")
                    continue
                    
                st.markdown("---")
                st.markdown(f"### 🧪 분석 샘플: **{group_name}**")
                
                iv_data = {}
                R_dict = {}
                voltages = None
                
                for gap in gaps_found:
                    f = gap_dict[gap]
                    f.seek(0)
                    try:
                        df = pd.read_csv(f, skiprows=1, encoding='cp949')
                        col_v = [c for c in df.columns if 'Voltage' in c][0]
                        col_i = [c for c in df.columns if 'Current' in c][0]
                        col_r = [c for c in df.columns if 'Resistance' in c][0]
                        
                        v = df[col_v].values
                        i_mA = df[col_i].values * 1000
                        r = df[col_r].values
                        
                        if voltages is None:
                            voltages = v
                            iv_data['Voltage (V)'] = voltages
                            
                        iv_data[f'{gap}um Current (mA)'] = i_mA
                        
                        idx_m01 = np.argmin(np.abs(v - (-0.1)))
                        idx_p01 = np.argmin(np.abs(v - 0.1))
                        min_R = min(r[idx_m01], r[idx_p01])
                        R_dict[gap] = min_R
                        
                    except Exception as e:
                        st.error(f"오류: [{f.name}] 파일 처리 실패 ({e})")
                        continue
                        
                L = np.array(gaps_found)
                R = np.array([R_dict[g] for g in gaps_found])
                slope, intercept = np.polyfit(L, R, 1)
                
                Rs = slope * t_w_um
                Rc = intercept / 2
                Lt = intercept / (2 * slope)
                rho_c = Rs * (Lt * 1e-4)**2 if Lt > 0 else np.nan
                
                all_tlm_summaries.append({
                    'Sample Name': group_name,
                    'Measured Gaps (um)': str(gaps_found),
                    'Sheet Resistance, Rs (Ohm/sq)': Rs,
                    'Contact Resistance, Rc (Ohm)': Rc,
                    'Specific Resistivity, rho_c (Ohm.cm2)': rho_c,
                    '10um R (Ohm)': R_dict.get(10),
                    '20um R (Ohm)': R_dict.get(20),
                    '30um R (Ohm)': R_dict.get(30),
                    '40um R (Ohm)': R_dict.get(40),
                    '80um R (Ohm)': R_dict.get(80)
                })
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                cmap = plt.colormaps['tab10']
                
                for idx, gap in enumerate(gaps_found):
                    if f'{gap}um Current (mA)' in iv_data:
                        ax1.plot(iv_data['Voltage (V)'], iv_data[f'{gap}um Current (mA)'], color=cmap(idx % 10), linewidth=2, label=f'{gap} $\mu m$')
                        
                ax1.set_title("Ohmic Characteristics (I-V)", fontsize=13, fontweight='bold')
                ax1.set_xlabel("Voltage (V)", fontsize=11, fontweight='bold')
                ax1.set_ylabel("Current (mA)", fontsize=11, fontweight='bold')
                ax1.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True, right=True)
                
                if not tlm_x_auto:
                    ax1.set_xlim(tlm_x_min, tlm_x_max)
                if not tlm_y_auto:
                    ax1.set_ylim(tlm_y_min, tlm_y_max)
                    
                ax1.legend()
                
                ax2.plot(L, R, 'ko', markersize=8, label='Measured Resistance')
                L_line = np.array([0, max(L)]) 
                R_line = slope * L_line + intercept
                ax2.plot(L_line, R_line, 'r--', linewidth=2, label=f'Linear Fit (R² ≒ {np.corrcoef(L, R)[0,1]**2:.4f})')
                
                ax2.set_title("TLM Plot (Resistance vs Gap)", fontsize=13, fontweight='bold')
                ax2.set_xlabel("Gap Distance ($\mu m$)", fontsize=11, fontweight='bold')
                ax2.set_ylabel("Total Resistance ($\Omega$)", fontsize=11, fontweight='bold')
                ax2.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True, right=True)
                
                text_str = f"$R_s$: {Rs:.2f} $\\Omega/sq$\n$R_c$: {Rc:.2f} $\\Omega$\n$\\rho_c$: {rho_c:.2e} $\\Omega\\cdot cm^2$"
                props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
                ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props)
                ax2.legend(loc='lower right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                col_dl1, col_dl2 = st.columns(2)
                
                buf_img_tlm = BytesIO()
                fig.savefig(buf_img_tlm, format="png", dpi=300, bbox_inches='tight')
                with col_dl1:
                    st.download_button(
                        label=f"🖼️ [{group_name}] 그래프 이미지 다운로드 (.png)",
                        data=buf_img_tlm.getvalue(),
                        file_name=f"TLM_{group_name}_Plot.png",
                        mime="image/png",
                        key=f"dl_img_tlm_{group_name}"
                    )
                
                iv_df = pd.DataFrame(iv_data)
                buf_tlm = BytesIO()
                with pd.ExcelWriter(buf_tlm, engine='openpyxl') as writer:
                    iv_df.to_excel(writer, index=False, sheet_name='I-V Data')
                with col_dl2:
                    st.download_button(
                        label=f"📥 [{group_name}] 전극별 I-V 데이터 다운로드 (.xlsx)",
                        data=buf_tlm.getvalue(),
                        file_name=f"TLM_{group_name}_IV_Data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_excel_tlm_{group_name}"
                    )
            
            if all_tlm_summaries:
                st.markdown("---")
                st.subheader("📋 전체 샘플 TLM 분석 결과 요약 비교")
                sum_df = pd.DataFrame(all_tlm_summaries)
                
                def highlight_min_tlm(s):
                    is_min = s == s.min(skipna=True) 
                    return ['color: red; font-weight: bold' if v else '' for v in is_min]
                    
                styled_sum = sum_df.style.apply(
                    highlight_min_tlm, 
                    subset=[
                        'Sheet Resistance, Rs (Ohm/sq)', 
                        'Contact Resistance, Rc (Ohm)', 
                        'Specific Resistivity, rho_c (Ohm.cm2)'
                    ]
                ).format({'Specific Resistivity, rho_c (Ohm.cm2)': '{:.4E}'})
                
                st.dataframe(styled_sum, use_container_width=True)
                
                buf_sum = BytesIO()
                with pd.ExcelWriter(buf_sum, engine='openpyxl') as writer:
                    sum_df.to_excel(writer, index=False)
                st.download_button("📥 통합 TLM 요약 비교 엑셀 다운로드", data=buf_sum.getvalue(), file_name="Combined_TLM_Summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =====================================================================
# [모드 4] AFM 표면 분석 (새로 추가됨!)
# =====================================================================
elif analysis_mode == "4. AFM 표면 분석":
    st.markdown("AFM 장비에서 Export한 **텍스트(.txt) 파일**을 업로드하세요. 표면 거칠기 정량화 및 3D 지형도를 제공합니다.")
    
    uploaded_afm = st.file_uploader(
        "AFM Raw Data 텍스트 파일을 올려주세요 (.txt)", 
        type=['txt', 'csv', 'dat'], 
        accept_multiple_files=True, 
        key=f"afm_uploader_{st.session_state['afm_uploader_key']}"
    )

    if uploaded_afm:
        st.session_state['afm_files_data'] = uploaded_afm

    files_to_process_afm = st.session_state['afm_files_data']

    if files_to_process_afm:
        col1, col2 = st.columns([8, 2])
        with col1:
            st.success(f"현재 {len(files_to_process_afm)}개의 AFM 파일이 분석 중입니다.")
        with col2:
            if st.button("🗑️ 전체 파일 삭제", use_container_width=True):
                st.session_state['afm_files_data'] = []
                st.session_state['afm_uploader_key'] += 1
                st.rerun()

        afm_summaries = []

        for file in files_to_process_afm:
            file_name = file.name
            st.markdown("---")
            st.subheader(f"🔬 샘플 분석: {file_name}")
            
            try:
                # 1. 텍스트 파일 파싱 (헤더 영역 건너뛰고 순수 데이터만 추출)
                file.seek(0)
                lines = file.readlines()
                
                # 데이터가 시작되는 "X \t Y \t Z" 라인을 찾음
                start_idx = 0
                for i, line in enumerate(lines):
                    if b'X' in line and b'Y' in line and b'Z' in line:
                        start_idx = i + 1
                        break
                
                # 순수 데이터만 Pandas로 불러오기
                file.seek(0)
                df_afm = pd.read_csv(file, skiprows=start_idx, sep=r'\s+', names=['X', 'Y', 'Z'])
                df_afm = df_afm.dropna()
                
                # 2. X, Y 픽셀 사이즈(해상도) 자동 계산 (예: 256x256)
                x_unique = len(df_afm['X'].unique())
                y_unique = len(df_afm['Y'].unique())
                
                # 1D 배열을 2D 매트릭스로 변환 (Z축 높이 행렬)
                Z_matrix = df_afm['Z'].values.reshape((y_unique, x_unique))
                
                # 3. 거칠기(Roughness) 수치 계산 (수학 공식 적용)
                z_mean = np.mean(Z_matrix)
                z_centered = Z_matrix - z_mean # 기준면 평탄화(Leveling)
                
                Ra = np.mean(np.abs(z_centered)) # Average Roughness
                Rq = np.sqrt(np.mean(z_centered**2)) # RMS Roughness
                Rpv = np.max(z_centered) - np.min(z_centered) # Peak-to-Valley
                
                afm_summaries.append({
                    'Sample Name': file_name,
                    'Ra (nm)': Ra,
                    'Rq (RMS, nm)': Rq,
                    'Rpv (Peak-to-Valley, nm)': Rpv
                })
                
                # 4. 화면 분할 시각화 (좌: 거칠기 결과 / 우: 3D 모델)
                col_res, col_plot = st.columns([1, 2])
                
                with col_res:
                    st.markdown("### 📊 표면 거칠기(Roughness)")
                    st.info(f"**$R_q$ (RMS)** : {Rq:.3f} nm\n\n**$R_a$ (Average)** : {Ra:.3f} nm\n\n**$R_{{pv}}$ (Max-Min)** : {Rpv:.3f} nm")
                    st.write(f"- 스캔 해상도: {x_unique} x {y_unique} 픽셀")
                    st.write(f"- 스캔 크기: {df_afm['X'].max():.1f} $\mu m$ x {df_afm['Y'].max():.1f} $\mu m$")
                    
                with col_plot:
                    # Plotly를 이용한 고해상도 인터랙티브 3D 렌더링
                    fig = go.Figure(data=[go.Surface(
                        z=z_centered,
                        x=df_afm['X'].unique(),
                        y=df_afm['Y'].unique(),
                        colorscale=color_theme,
                        colorbar=dict(title='Height (nm)')
                    )])
                    
                    fig.update_layout(
                        title='3D Topography Surface Map',
                        scene=dict(
                            xaxis_title='X (um)',
                            yaxis_title='Y (um)',
                            zaxis_title='Z Height (nm)',
                            aspectratio=dict(x=1, y=1, z=0.4) # Z축을 살짝 압축하여 시각적 안정감 부여
                        ),
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"데이터 파싱 오류: {file_name} 파일의 양식이 올바르지 않습니다. ({e})")

        # 분석 요약표 엑셀 추출
        if afm_summaries:
            st.markdown("---")
            st.subheader("📋 전체 샘플 AFM 거칠기 요약 비교")
            afm_sum_df = pd.DataFrame(afm_summaries)
            
            st.dataframe(afm_sum_df.style.format({'Ra (nm)': '{:.3f}', 'Rq (RMS, nm)': '{:.3f}', 'Rpv (Peak-to-Valley, nm)': '{:.3f}'}), use_container_width=True)
            
            buf_afm = BytesIO()
            with pd.ExcelWriter(buf_afm, engine='openpyxl') as writer:
                afm_sum_df.to_excel(writer, index=False)
            st.download_button("📥 통합 거칠기 요약 엑셀 다운로드", data=buf_afm.getvalue(), file_name="AFM_Roughness_Summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
