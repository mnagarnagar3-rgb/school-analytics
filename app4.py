import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
from fpdf import FPDF
import google.generativeai as genai

# --- 1. CONFIG & AI SETUP ---
st.set_page_config(page_title="EduInsight Pro | Class I Analytics", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. ROBUST DATA ENGINE FOR CLASS I ---
@st.cache_data
def load_and_process_class1_data():
    file_path = "Student_Marks.xlsx"
    if not os.path.exists(file_path): return pd.DataFrame()
    
    # Reading data and handling the specific Class I format
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    
    # Identify subject columns for melting
    subjects = [c for c in df.columns if any(s in c for s in ['ENG', 'HINDI', 'Maths', 'Science', 'English'])]
    
    df_melted = pd.melt(df, id_vars=['Student_Name'], 
                         value_vars=subjects, var_name='Subject_Term', value_name='Marks')
    
    # Strict 2-decimal rounding
    df_melted['Marks'] = pd.to_numeric(df_melted['Marks'], errors='coerce').fillna(0).round(2)
    
    # Splitting Term/Unit Test info if present in subject names
    # Example: 'ENG UT 4' -> Subject: 'ENG', Term: 'UT 4'
    df_melted['Term'] = df_melted['Subject_Term'].apply(lambda x: x.split()[-1] if ' ' in x else 'General')
    df_melted['Subject'] = df_melted['Subject_Term'].apply(lambda x: ' '.join(x.split()[:-1]) if ' ' in x else x)
    
    return df_melted

df = load_and_process_class1_data()

# --- 3. PROFESSIONAL SIDEBAR ---
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
    
    t_std = df['Student_Name'].nunique()
    ov_avg = df['Marks'].mean()
    std_avg = df.groupby('Student_Name')['Marks'].mean().reset_index()
    std_avg['Status'] = std_avg['Marks'].apply(lambda x: 'Pass' if x >= 33 else 'Fail')
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Students", t_std)
    k2.metric("Overall Avg", f"{ov_avg:.2f}%")
    k3.metric("Passed", (std_avg['Status'] == 'Pass').sum())
    k4.metric("Failed", (std_avg['Status'] == 'Fail').sum(), delta_color="inverse")
    
    st.divider()
    
    g1, g2 = st.columns([2, 1])
    with g1:
        st.subheader("📊 Subject Averages")
        sub_avg = df.groupby('Subject')['Marks'].mean().reset_index()
        st.plotly_chart(px.bar(sub_avg, x='Subject', y='Marks', text_auto='.2f', color='Subject'), use_container_width=True)
    with g2:
        st.subheader("🥧 Pass/Fail Ratio")
        st.plotly_chart(px.pie(std_avg, names='Status', color='Status', 
                               color_discrete_map={'Pass':'#2ecc71', 'Fail':'#e74c3c'}), use_container_width=True)
    
    st.subheader("🏆 Subject Performance Highlights")
    st.table(df.groupby('Subject')['Marks'].agg(['max', 'min', 'mean']).rename(columns={'max':'Highest','min':'Lowest','mean':'Avg'}).style.format("{:.2f}"))

elif page == "🏫 Class Analysis":
    st.title("🏫 Comprehensive Class Analysis")
    st.plotly_chart(px.box(df, x="Subject", y="Marks", color="Subject", title="Score Distribution by Subject"))
    st.subheader("📋 Raw Class Data Table")
    st.dataframe(df, use_container_width=True)

elif page == "📅 Class Term Analysis":
    st.title("📅 Term-wise Progress Tracker")
    t_trend = df.groupby('Term')['Marks'].mean().reset_index()
    st.plotly_chart(px.line(t_trend, x='Term', y='Marks', markers=True, title="Unit Test Performance Trend"))
    st.subheader("📊 Tabular Term Data")
    st.table(t_trend.set_index('Term').style.format("{:.2f}"))

elif page == "📑 Subject x Term Analysis":
    st.title("📑 Subject Growth Across Terms")
    # Using pivot_table to fix the crash seen in your image
    matrix = df.pivot_table(index='Subject', columns='Term', values='Marks', aggfunc='mean').round(2)
    
    if matrix.shape[1] > 1:
        terms = list(matrix.columns)
        matrix['Delta (Growth)'] = (matrix[terms[-1]] - matrix[terms[0]]).round(2)
    
    st.plotly_chart(px.bar(df, x='Subject', y='Marks', color='Term', barmode='group', text_auto='.2f'))
    st.subheader("📊 Term Comparison Table with Delta")
    st.table(matrix.style.format("{:.2f}"))

elif page == "📈 Individual Growth":
    std = st.selectbox("Select Student", sorted(df['Student_Name'].unique()))
    s_df = df[df['Student_Name'] == std]
    st.plotly_chart(px.bar(s_df, x="Term", y="Marks", color="Subject", barmode="group", title=f"Progress: {std}"))
    st.subheader(f"📋 Performance Table: {std}")
    st.table(s_df.pivot_table(index='Subject', columns='Term', values='Marks', aggfunc='mean').style.format("{:.2f}"))

elif page == "👥 Compare Students":
    st.title("👥 Comparison Suite")
    st.info("Comparison is relative to the FIRST student selected.")
    selected = st.multiselect("Select Students:", sorted(df['Student_Name'].unique()), default=sorted(df['Student_Name'].unique())[:2])
    
    if len(selected) > 1:
        c_df = df[df['Student_Name'].isin(selected)]
        # pivot_table fixes the ValueError seen in your image
        pivot_c = c_df.pivot_table(index='Subject', columns='Student_Name', values='Marks', aggfunc='mean').round(2)
        
        baseline = selected[0]
        for other in selected[1:]:
            pivot_c[f"Diff (vs {baseline})"] = (pivot_c[other] - pivot_c[baseline]).round(2)
            
        st.plotly_chart(px.bar(c_df, x='Subject', y='Marks', color='Student_Name', barmode='group'))
        st.subheader("📊 Comparative Student Data")
        st.table(pivot_c.style.format("{:.2f}"))

elif page == "📄 Reports":
    st.title("📄 Export Center")
    st.dataframe(df, use_container_width=True)
    if st.button("Generate Executive PDF"):
        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Class I Academic Report Summary", ln=True, align='C')
        for i, r in df.head(15).iterrows():
            pdf.cell(200, 8, txt=f"{r['Student_Name']} | {r['Subject']} | {r['Term']}: {r['Marks']}", ln=True)
        st.download_button("📥 Download PDF", bytes(pdf.output()), "ClassI_Report.pdf")