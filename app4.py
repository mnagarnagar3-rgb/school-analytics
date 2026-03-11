import streamlit as st
import pandas as pd
import plotly.express as px
import os
from fpdf import FPDF

# --- 1. CONFIG ---
st.set_page_config(page_title="EduInsight Pro | Class I", layout="wide")

# --- 2. DATA ENGINE (FIXED FOR CLASS_1.XLSX) ---
@st.cache_data
def load_data():
    file_path = "CLASS_1.xlsx" # Change to CLASS_1.xlsx if that's the filename on GitHub
    if not os.path.exists(file_path): return pd.DataFrame()
    
    # Load the file - No skipping needed for this specific file
    df = pd.read_excel(file_path)
    
    # Clean column names
    df.columns = df.columns.astype(str).str.strip()
    
    # Logic to handle the UT mapping based on your column positions
    # UT 4: Cols 1-4 | UT 5: Cols 5-8 | UT 6: Cols 9-12
    mapping = {
        df.columns[1]: 'UT 4', df.columns[2]: 'UT 4', df.columns[3]: 'UT 4', df.columns[4]: 'UT 4',
        df.columns[5]: 'UT 5', df.columns[6]: 'UT 5', df.columns[7]: 'UT 5', df.columns[8]: 'UT 5',
        df.columns[9]: 'UT 6', df.columns[10]: 'UT 6', df.columns[11]: 'UT 6', df.columns[12]: 'UT 6'
    }
    
    # Melt data
    df_melted = pd.melt(df, id_vars=[df.columns[0]], value_vars=list(mapping.keys()), 
                        var_name='Raw_Col', value_name='Marks')
    
    # Rename first column to 'Student Name'
    df_melted = df_melted.rename(columns={df.columns[0]: 'Student Name'})
    
    # Assign correct Subject and Term names
    df_melted['Term'] = df_melted['Raw_Col'].map(mapping)
    df_melted['Subject'] = df_melted['Raw_Col'].str.replace('.1', '').str.replace('.2', '').str.strip()
    
    # Numeric cleanup & strict 2-decimal rounding
    df_melted['Marks'] = pd.to_numeric(df_melted['Marks'], errors='coerce').fillna(0).round(2)
    
    # Group by Mean to handle duplicate students like Atharv Singh
    return df_melted.groupby(['Student Name', 'Subject', 'Term'], as_index=False)['Marks'].mean()

df = load_data()

# --- 3. NAVIGATION ---
st.sidebar.title("💎 EduInsight Pro")
page = st.sidebar.selectbox("📊 Navigation", [
    "🏠 Executive Overview", "🏫 Class Analysis", "📅 Term Analysis", 
    "📑 Subject x Term", "👥 Comparison Suite", "📄 Reports"
])

# --- 4. EXECUTIVE DASHBOARD ---
if not df.empty:
    if page == "🏠 Executive Overview":
        st.title("🏠 Executive Summary: Class I")
        
        # Metrics
        t_std = df['Student Name'].nunique()
        ov_avg = df['Marks'].mean()
        std_avg = df.groupby('Student Name')['Marks'].mean().reset_index()
        std_avg['Status'] = std_avg['Marks'].apply(lambda x: 'Pass' if x >= 33 else 'Fail')
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Students", t_std)
        k2.metric("Overall Avg", f"{ov_avg:.2f}%")
        k3.metric("Passed", (std_avg['Status'] == 'Pass').sum())
        k4.metric("Failed", (std_avg['Status'] == 'Fail').sum(), delta_color="inverse")
        
        st.divider()
        
        # Charts
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 Subject Averages")
            sub_avg = df.groupby('Subject')['Marks'].mean().reset_index()
            st.plotly_chart(px.bar(sub_avg, x='Subject', y='Marks', text_auto='.2f', color='Subject'))
        with c2:
            st.subheader("🥧 Pass/Fail Ratio")
            st.plotly_chart(px.pie(std_avg, names='Status', color='Status', color_discrete_map={'Pass':'#2ecc71', 'Fail':'#e74c3c'}))
        
        st.subheader("📋 Performance Table")
        st.table(df.groupby('Subject')['Marks'].agg(['max', 'min', 'mean']).style.format("{:.2f}"))

    elif page == "📑 Subject x Term":
        st.title("📑 Term-wise Growth Matrix")
        matrix = df.pivot_table(index='Subject', columns='Term', values='Marks', aggfunc='mean').round(2)
        
        # Add Delta (Growth) column
        if matrix.shape[1] > 1:
            matrix['Delta (UT4 to UT6)'] = (matrix.iloc[:, -1] - matrix.iloc[:, 0]).round(2)
        
        st.plotly_chart(px.bar(df, x='Subject', y='Marks', color='Term', barmode='group', text_auto='.2f'))
        st.table(matrix.style.format("{:.2f}"))

    elif page == "👥 Comparison Suite":
        st.title("👥 Student Comparison")
        selected = st.multiselect("Select Students:", sorted(df['Student Name'].unique()), default=sorted(df['Student Name'].unique())[:2])
        if len(selected) > 1:
            comp_df = df[df['Student Name'].isin(selected)]
            p_comp = comp_df.pivot_table(index='Subject', columns='Student Name', values='Marks', aggfunc='mean').round(2)
            st.plotly_chart(px.bar(comp_df, x='Subject', y='Marks', color='Student Name', barmode='group'))
            st.table(p_comp.style.format("{:.2f}"))
else:
    st.warning("Please ensure 'Student_Marks.xlsx' is uploaded to GitHub.")