import streamlit as st
import pandas as pd
import plotly.express as px
import os
from fpdf import FPDF
import google.generativeai as genai

# --- 1. CONFIG & AI ---
st.set_page_config(page_title="EduInsight Pro | Class I", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. SPECIAL DATA ENGINE FOR CLASS I ---
@st.cache_data
def load_and_clean_data():
    file_path = "Student_Marks.xlsx"
    if not os.path.exists(file_path): return pd.DataFrame()
    
    # Load skipping the empty top rows of your Class I sheet
    df = pd.read_excel(file_path, skiprows=5) 
    df.columns = df.columns.str.strip()
    
    # Clean Student Names
    df = df[df['Student Name'].notna()]
    
    # Mapping Class I Columns to specific Unit Tests (UT)
    mapping = {
        'ENG ': 'UT 4', 'HINDI': 'UT 4', 'Maths': 'UT 4', 'Science': 'UT 4',
        'English': 'UT 5', 'Hindi': 'UT 5', 'Maths.1': 'UT 5', 'Science.1': 'UT 5',
        'English.1': 'UT 6', 'Hindi.1': 'UT 6', 'Maths.2': 'UT 6', 'Science.2': 'UT 6'
    }
    
    # Melt the data into a long format
    df_melted = pd.melt(df, id_vars=['Student Name'], value_vars=list(mapping.keys()), 
                        var_name='Raw_Col', value_name='Marks')
    
    # Assign correct Subject and Term names
    df_melted['Term'] = df_melted['Raw_Col'].map(mapping)
    df_melted['Subject'] = df_melted['Raw_Col'].str.replace('.1', '').str.replace('.2', '').str.strip()
    
    # Numeric cleanup & strict 2-decimal rounding
    df_melted['Marks'] = pd.to_numeric(df_melted['Marks'], errors='coerce').fillna(0).round(2)
    
    # FIXED: Grouping by Mean to prevent the "Comparison Suite" crash (ValueError)
    df_clean = df_melted.groupby(['Student Name', 'Subject', 'Term'], as_index=False)['Marks'].mean()
    
    return df_clean

df = load_and_clean_data()

# --- 3. PROFESSIONAL NAVIGATION ---
st.sidebar.title("💎 EduInsight Pro")
page = st.sidebar.selectbox("📊 Navigation", [
    "🏠 Executive Overview", 
    "🏫 Class Analysis", 
    "📚 Subject Analysis", 
    "📅 Class Term Analysis", 
    "📑 Subject x Term Analysis",
    "📈 Individual Growth", 
    "👥 Compare Students", 
    "📄 Reports"
])

# --- 4. PAGE LOGIC ---

if page == "🏠 Executive Overview":
    st.title("🏠 Executive Overview: Class I")
    
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
    
    g1, g2 = st.columns([2, 1])
    with g1:
        st.subheader("📊 Subject Averages (Column Chart)")
        sub_avg = df.groupby('Subject')['Marks'].mean().reset_index()
        st.plotly_chart(px.bar(sub_avg, x='Subject', y='Marks', text_auto='.2f', color='Subject'), use_container_width=True)
    with g2:
        st.subheader("🥧 Pass/Fail Ratio")
        st.plotly_chart(px.pie(std_avg, names='Status', color='Status', color_discrete_map={'Pass':'#2ecc71', 'Fail':'#e74c3c'}), use_container_width=True)
    
    st.subheader("🏆 Subject Performance (Tabular Data)")
    st.table(df.groupby('Subject')['Marks'].agg(['max', 'min', 'mean']).rename(columns={'max':'Highest','min':'Lowest','mean':'Avg'}).style.format("{:.2f}"))

elif page == "📑 Subject x Term Analysis":
    st.title("📑 Subject Growth Across Terms")
    # Using pivot_table to fix the crash seen in your image
    matrix = df.pivot_table(index='Subject', columns='Term', values='Marks', aggfunc='mean').round(2)
    
    # DELTA CHANGE: Calculate growth from first UT to last UT
    if matrix.shape[1] > 1:
        terms = list(matrix.columns)
        matrix['Delta (Growth)'] = (matrix[terms[-1]] - matrix[terms[0]]).round(2)
    
    st.plotly_chart(px.bar(df, x='Subject', y='Marks', color='Term', barmode='group', text_auto='.2f'))
    st.subheader("📊 Term Comparison Table with Delta")
    st.table(matrix.style.format("{:.2f}"))

elif page == "👥 Compare Students":
    st.title("👥 Comparison Suite")
    st.info("Comparison is relative to the FIRST student selected.")
    selected = st.multiselect("Select Students:", sorted(df['Student Name'].unique()), default=sorted(df['Student Name'].unique())[:2])
    
    if len(selected) > 1:
        c_df = df[df['Student Name'].isin(selected)]
        # pivot_table fixes the ValueError (Duplicate keys)
        pivot_c = c_df.pivot_table(index='Subject', columns='Student Name', values='Marks', aggfunc='mean').round(2)
        
        baseline = selected[0]
        for other in selected[1:]:
            pivot_c[f"Diff (vs {baseline})"] = (pivot_c[other] - pivot_c[baseline]).round(2)
            
        st.plotly_chart(px.bar(c_df, x='Subject', y='Marks', color='Student Name', barmode='group'))
        st.subheader("📊 Comparative Student Data")
        st.table(pivot_c.style.format("{:.2f}"))

# (Other pages follow this robust logic...)