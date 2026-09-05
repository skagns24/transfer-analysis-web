]import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from io import BytesIO
from scipy.signal import savgol_filter
import plotly.graph_objects as go

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="Semiconductor Data Analysis Tool")

# ---------------------------------------------------------
# Session State 초기화
# ---------------------------------------------------------
if 'transfer_files_data' not in st.session_state: st.session_state['transfer_files_data'] = []
if 'output_files_data' not in st.session_state: st.session_state['output_files_data'] = []
if 'tlm_files_data' not in st.session_state: st.session_state['tlm_files_data'] = []
if 'afm_files_data' not in st.session_state: st.session_state['afm_files_data'] = []
    
if 't_uploader_key' not in st.session_state: st.session_state['t_uploader_key'] = 0
if 'o_uploader_key' not in st.session_state: st.session_state['o_uploader_key'] = 0
if 'tlm_uploader_key' not in st.session_state: st.session_state['tlm_uploader_key'] = 0
if 'afm_uploader_key' not in st.session_state: st.session_state['afm_uploader_key'] = 0

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
        apply_smoothing = st.checkbox("Gm 노이즈 필터링 적용", value=False, key="apply_smoothing")
        
        if apply_smoothing:
            window_length = st.slider("필터 강도 (Window Length, 홀수)", min_value=3, max_value=51, value=11, step=2, key="sg_window")
            poly_order = st.slider("다항식 차수 (Poly Order)", min_value=1, max_value=5, value=2, key="sg_poly")

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

    elif analysis_mode == "4. AFM 표면 분석":
        st.header("⚙️ AFM 3D 렌더링 설정")
        color_theme = st.selectbox("컬러 맵 선택", ["earth", "hot", "viridis", "plasma", "inferno", "magma", "cividis"], index=0)
        st.info("💡 메인 화면의 슬라이더를 조절하여 '기준면(Baseline)'을 설정하면, 해당 영역을 0으로 맞춘 완벽한 단차(Step Height) 계산이 가능합니다.")

    st.markdown("---")
    st.header("🤖 AI 연구 어시스턴트")
    st.link_button("💬 Gemini 새 창으로 열기", "https://gemini.google.com/app", use_container_width=True)

st.title("📊 반도체 소자 특성 통합 분석 웹앱")

# =====================================================================
# [모드 1] Transfer 특성 분석
# =====================================================================
if analysis_mode == "1. Transfer 특성 분석":
    st.markdown("Transfer CSV 파일들을 업로드하세요. 통합 그래프, 파라미터 비교표, 개별 단위 변환 엑셀을 제공합니다.")
    
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
                    
                    gm_raw = np.gradient(id_norm, vg)
                    df.loc[group.index, 'Raw Gm (mS/mm)'] = gm_raw
                    
                    if apply_smoothing and len(gm_raw) > window_length:
                        actual_poly = min(poly_order, window_length - 1)
                        gm_final = savgol_filter(gm_raw, window_length, actual_poly)
                        df.loc[group.index, 'Smoothed Gm (mS/mm)'] = gm_final
                    else:
                        gm_final = gm_raw
                    
                    label_name = f"{file_name}"
                    ax1.plot(vg, id_norm, label=label_name, color=c, linewidth=2)
                    
                    if apply_smoothing and len(gm_raw) > window_length:
                        ax1_twin.plot(vg, gm_raw, color=c, linestyle=':', linewidth=1.5, alpha=0.3)
                        ax1_twin.plot(vg, gm_final, color=c, linestyle='--', linewidth=2.5, alpha=0.8)
                    else:
                        ax1_twin.plot(vg, gm_final, color=c, linestyle='--', linewidth=2, alpha=0.6)
                    
                    ax2.plot(vg, id_abs, label=label_name, color=c, linewidth=2)
                    ax2.plot(vg, ig_abs, color=c, linestyle=':', linewidth=2, alpha=0.6)

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

        legend_title = "Solid: Id, Dashed: Smoothed Gm, Dotted: Raw Gm" if apply_smoothing else "Solid: Id, Dashed: Gm"
        ax1.legend(loc='upper left', frameon=True, fontsize=9, title=legend_title)

        ax2.set_yscale('log')
        ax2.set_xlabel('Gate Voltage (V)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Current (mA/mm)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True, right=True)
        ax2.legend(loc='lower right', frameon=True, fontsize=9, title="Solid: |Id|, Dotted: |Ig|")

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
    st.markdown("Output CSV 파일들을 업로드하세요. 게이트 전압(Vg)별 드레인 전류 곡선과 타겟 Vg의 온저항(Ron)을 자동 추출합니다.")
    
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
                    
                    ax.plot(vd_vals, id_vals, color=c, linewidth=2, label=f"{file_name} (Vg={vg_val:g}V)")

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
                    ax.plot(v_line, i_line, color=line_color, linestyle='--', linewidth=1.5, label=f"Linear fit (Ron: {file_ron:.1f} Ohm.mm)")

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
            st.subheader("📋 소자별 On-Resistance (Ron) 요약 비교")
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
            st.warning("업로드된 파일 중 이름에 10, 20, 30, 40, 80이 포함된 CSV 파일을 찾을 수 없습니다.")
        else:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['axes.linewidth'] = 1.5
            
            all_tlm_summaries = []
            
            for group_name, gap_dict in tlm_groups.items():
                gaps_found = sorted(gap_dict.keys())
                
                if not all(g in gaps_found for g in [10, 20, 30, 40]):
                    st.warning(f"[{group_name}] 샘플은 TLM 분석을 위해 최소 10, 20, 30, 40 파일이 모두 필요합니다.")
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
                        ax1.plot(iv_data['Voltage (V)'], iv_data[f'{gap}um Current (mA)'], color=cmap(idx % 10), linewidth=2, label=f'{gap} um')
                        
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
                ax2.set_xlabel("Gap Distance (um)", fontsize=11, fontweight='bold')
                ax2.set_ylabel("Total Resistance (Ohm)", fontsize=11, fontweight='bold')
                ax2.tick_params(axis='both', direction='in', labelsize=10, width=1.5, top=True, right=True)
                
                text_str = f"Rs: {Rs:.2f} Ohm/sq\nRc: {Rc:.2f} Ohm\nrho_c: {rho_c:.2e} Ohm.cm2"
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
                        label=f"🖼️ [{group_name}] 그래프 다운로드 (.png)",
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
                        label=f"📥 [{group_name}] 데이터 다운로드 (.xlsx)",
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
# [모드 4] AFM 표면 분석 (기준면 레벨링 + 단차 프로파일 + 영역 크롭)
# =====================================================================
elif analysis_mode == "4. AFM 표면 분석":
    st.markdown("AFM 장비에서 Export한 **텍스트(.txt) 파일**을 업로드하세요. 부분 영역(ROI) 평탄화 기반 거칠기 정량화 및 정밀 단면 프로파일(Line Profile)을 제공합니다.")
    
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
                # 1. 텍스트 파일 파싱
                file.seek(0)
                lines = file.readlines()
                
                start_idx = 0
                for i, line in enumerate(lines):
                    if b'X' in line and b'Y' in line and b'Z' in line:
                        start_idx = i + 1
                        break
                
                file.seek(0)
                df_afm = pd.read_csv(file, skiprows=start_idx, sep=r'\s+', names=['X', 'Y', 'Z'])
                df_afm = df_afm.dropna()
                
                # 2. 고유 X, Y 및 원본 Z_matrix 생성
                x_coords = np.sort(df_afm['X'].unique())
                y_coords = np.sort(df_afm['Y'].unique())
                Z_matrix_raw = df_afm['Z'].values.reshape((len(y_coords), len(x_coords)))
                
                x_min_real, x_max_real = float(x_coords[0]), float(x_coords[-1])
                y_min_real, y_max_real = float(y_coords[0]), float(y_coords[-1])

                # --- 3. UI 슬라이더: 기준면(Baseline) 지정 및 단차 계산 ---
                st.markdown("#### 📏 단면 프로파일 및 단차(Step Height) 측정 설정")
                st.info("단차를 잴 때 기준이 되는 평평한 바닥면을 '기준면(Baseline)' 슬라이더로 지정하세요. 이 구간을 0으로 완벽하게 수평 피팅합니다.")
                
                # Y축 라인 프로파일 위치 선택
                profile_y = st.slider(f"단면을 자를 기준 Y축 위치 (um)", y_min_real, y_max_real, y_min_real + (y_max_real-y_min_real)/2, step=0.1, key=f"p_slide_{file_name}")
                y_idx = np.argmin(np.abs(y_coords - profile_y))
                raw_z_line = Z_matrix_raw[y_idx, :]
                
                c_base, c_step = st.columns(2)
                with c_base:
                    base_range = st.slider("🟩 기준면(Baseline, Z=0으로 펴질 영역)", x_min_real, x_max_real, (x_min_real, x_min_real + 2.0), step=0.1, key=f"base_slide_{file_name}")
                with c_step:
                    step_range = st.slider("🟥 단차 측정(Step Height 계산 영역)", x_min_real, x_max_real, (x_max_real - 2.0, x_max_real), step=0.1, key=f"step_slide_{file_name}")

                # 지정된 기준면을 이용해 해당 Line의 기울기를 계산하고 완벽하게 0으로 폄 (Baseline Leveling)
                base_mask = (x_coords >= base_range[0]) & (x_coords <= base_range[1])
                step_mask = (x_coords >= step_range[0]) & (x_coords <= step_range[1])
                
                if sum(base_mask) > 1:
                    slope, intercept = np.polyfit(x_coords[base_mask], raw_z_line[base_mask], 1)
                    leveled_z_line = raw_z_line - (slope * x_coords + intercept)
                else:
                    leveled_z_line = raw_z_line # 마스킹 영역이 너무 좁을 경우 원본 유지
                    
                # 단차(Step Height) 계산
                if sum(step_mask) > 0:
                    step_height = np.mean(leveled_z_line[step_mask])
                else:
                    step_height = 0.0

                # --- 4. 3D 지형도용 전체 평탄화 (선택된 Baseline을 전체 영역에 적용) ---
                Z_matrix_flattened = np.zeros_like(Z_matrix_raw)
                for i in range(len(y_coords)):
                    z_line = Z_matrix_raw[i, :]
                    if sum(base_mask) > 1:
                        s, inc = np.polyfit(x_coords[base_mask], z_line[base_mask], 1)
                        Z_matrix_flattened[i, :] = z_line - (s * x_coords + inc)
                    else:
                        Z_matrix_flattened[i, :] = z_line

                # --- 5. 거칠기를 계산할 ROI(관심 영역) 크롭 ---
                st.markdown("#### ✂️ 표면 거칠기(Roughness) 계산 영역 지정")
                c1, c2 = st.columns(2)
                with c1:
                    sel_x = st.slider(f"거칠기 계산 X축 범위", x_min_real, x_max_real, (x_min_real, x_max_real), step=0.1, key=f"x_crop_{file_name}")
                with c2:
                    sel_y = st.slider(f"거칠기 계산 Y축 범위", y_min_real, y_max_real, (y_min_real, y_max_real), step=0.1, key=f"y_crop_{file_name}")

                x_crop_mask = (x_coords >= sel_x[0]) & (x_coords <= sel_x[1])
                y_crop_mask = (y_coords >= sel_y[0]) & (y_coords <= sel_y[1])
                crop_Z = Z_matrix_flattened[y_crop_mask, :][:, x_crop_mask]
                
                # 거칠기 계산
                Ra = np.mean(np.abs(crop_Z))
                Rq = np.sqrt(np.mean(crop_Z**2))
                Rpv = np.max(crop_Z) - np.min(crop_Z)
                
                afm_summaries.append({
                    'Sample Name': file_name,
                    'Step Height (nm)': step_height,
                    'Ra (nm)': Ra,
                    'Rq (RMS, nm)': Rq,
                    'Rpv (Peak-to-Valley, nm)': Rpv
                })
                
                # 6. 시각화 (좌: 수치 및 단면 프로파일, 우: 3D 플롯)
                col_res, col_plot = st.columns([1, 2])
                
                with col_res:
                    st.markdown("### 📊 분석 결과")
                    st.success(f"**측정된 단차(Step Height)** : {abs(step_height):.3f} nm")
                    st.info(f"**Rq (RMS 거칠기)** : {Rq:.3f} nm\n\n**Ra (Average)** : {Ra:.3f} nm\n\n**Rpv (Max-Min)** : {Rpv:.3f} nm")
                    
                    # 2D 라인 플롯 생성 (Matplotlib)
                    plt.rcParams['font.family'] = 'Arial'
                    fig_prof, ax_prof = plt.subplots(figsize=(6, 4))
                    
                    # 라인 프로파일 그리기
                    ax_prof.plot(x_coords, leveled_z_line, color='blue', linewidth=1.5, label='Leveled Profile')
                    
                    # 기준면(초록색)과 측정면(빨간색) 음영 하이라이트
                    ax_prof.axvspan(base_range[0], base_range[1], color='green', alpha=0.2, label='Baseline (Z=0)')
                    ax_prof.axvspan(step_range[0], step_range[1], color='red', alpha=0.2, label='Step Target')
                    
                    # 단차 텍스트 표시
                    ax_prof.axhline(0, color='black', linestyle='--', linewidth=1)
                    ax_prof.axhline(step_height, color='black', linestyle='--', linewidth=1)
                    
                    ax_prof.set_title(f"Leveled Line Profile (Cut at Y = {y_coords[y_idx]:.2f} um)", fontsize=11, fontweight='bold')
                    ax_prof.set_xlabel("X Distance (um)", fontsize=10)
                    ax_prof.set_ylabel("Height (nm)", fontsize=10)
                    ax_prof.grid(True, linestyle='--', alpha=0.6)
                    ax_prof.legend(loc='best', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_prof)
                    
                with col_plot:
                    # Plotly를 이용한 3D 지형도 (전체 영역 렌더링)
                    fig_3d = go.Figure(data=[go.Surface(
                        z=Z_matrix_flattened,
                        x=x_coords,
                        y=y_coords,
                        colorscale=color_theme,
                        colorbar=dict(title='Height (nm)')
                    )])
                    
                    # 3D 맵 위에 사용자가 선택한 라인 프로파일 위치를 붉은 선으로 표시
                    fig_3d.add_trace(go.Scatter3d(
                        x=x_coords,
                        y=np.full_like(x_coords, y_coords[y_idx]),
                        z=leveled_z_line + (np.max(Z_matrix_flattened) * 0.05), # 지형도 위로 살짝 띄워서 선명하게 보임
                        mode='lines',
                        line=dict(color='red', width=5),
                        name='Profile Line'
                    ))
                    
                    fig_3d.update_layout(
                        title=f'3D Topography (Baseline Leveled)',
                        scene=dict(
                            xaxis_title='X (um)',
                            yaxis_title='Y (um)',
                            zaxis_title='Z Height (nm)',
                            aspectratio=dict(x=1, y=1, z=0.4) 
                        ),
                        margin=dict(l=0, r=0, b=0, t=40),
                        showlegend=False
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
            except Exception as e:
                st.error(f"데이터 파싱 오류: {file_name} 파일의 양식이 올바르지 않습니다. ({e})")

        if afm_summaries:
            st.markdown("---")
            st.subheader("📋 전체 샘플 단차 및 거칠기 요약 비교")
            afm_sum_df = pd.DataFrame(afm_summaries)
            
            st.dataframe(afm_sum_df.style.format({
                'Step Height (nm)': '{:.3f}', 
                'Ra (nm)': '{:.3f}', 
                'Rq (RMS, nm)': '{:.3f}', 
                'Rpv (Peak-to-Valley, nm)': '{:.3f}'
            }), use_container_width=True)
            
            buf_afm = BytesIO()
            with pd.ExcelWriter(buf_afm, engine='openpyxl') as writer:
                afm_sum_df.to_excel(writer, index=False)
            st.download_button("📥 통합 요약 엑셀 다운로드", data=buf_afm.getvalue(), file_name="AFM_Step_Roughness_Summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
