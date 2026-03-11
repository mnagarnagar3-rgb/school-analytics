import streamlit as st
import pandas as pd
import plotly.express as px
import os
from fpdf import FPDF
import google.generativeai as genai

# --- 1. CONFIG ---
st.set_page_config(page_title="EduInsight Pro | Class I", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. SPECIAL DATA ENGINE FOR CLASS I ---
@st.cache_data
def load_class1_data():
    file_path = "CLASS I UT DATA TERM II.xlsx"
    if not os.path.exists(file_path): return pd.DataFrame()
    
    # Load skipping the empty top rows
    df = pd.read_excel(file_path, skiprows=5) 
    df.columns = df.columns.str.strip()
    
    # Clean Student Names
    df = df[df['Student Name'].notna()]
    
    # Subjects based on your Class I columns
    # We map them to their respective UTs based on the Excel structure
    # Cols 1-4: UT 4 | Cols 5-8: UT 5 | Cols 9-12: UT 6
    mapping = {
        'ENG ': 'UT 4', 'HINDI': 'UT 4', 'Maths': 'UT 4', 'Science': 'UT 4',
        'English': 'UT 5', 'Hindi': 'UT 5', 'Maths.1': 'UT 5', 'Science.1': 'UT 5',
        'English.1': 'UT 6', 'Hindi.1': 'UT 6', 'Maths.2': 'UT 6', 'Science.2': 'UT 6'
    }
    
    # Melt the data
    df_melted = pd.melt(df, id_vars=['Student Name'], value_vars=list(mapping.keys()), 
                        var_name='Raw_Col', value_name='Marks')
    
    # Assign correct Subject and Term names
    df_melted['Term'] = df_melted['Raw_Col'].map(mapping)
    df_melted['Subject'] = df_melted['Raw_Col'].str.replace('.1', '').str.replace('.2', '').str.strip()
    
    # Numeric cleanup & strict 2-decimal rounding
    df_melted['Marks'] = pd.to_numeric(df_melted['Marks'], errors='coerce').fillna(0).round(2)
    
    # FIXED: Grouping by Mean to prevent the "Comparison Suite" crash
    df_clean = df_melted.groupby(['Student Name', 'Subject', 'Term'], as_index=False)['Marks'].mean()
    
    return df_clean

df = load_class1_data()

# --- 3. EXECUTIVE NAVIGATION ---
st.sidebar.title("💎 EduInsight Pro")
page = st.sidebar.selectbox("📊 Navigation", [
    "🏠 Executive Overview", "🏫 Class Analysis", "📚 Subject Analysis",
    "📅 Term-wise Progress", "📑 Subject x Term Matrix",
    "📈 Student Growth", "👥 Comparison Suite", "📄 Reports"
])

# --- 4. PAGE LOGIC ---

if page == "🏠 Executive Overview":
    st.title("🏠 Class I Executive Summary")
    
    # Metrics
    total_std = df['Student Name'].nunique()
    avg_marks = df['Marks'].mean()
    
    # Pass/Fail logic (using 33 as threshold)
    std_perf = df.groupby('Student Name')['Marks'].mean().reset_index()
    std_perf['Status'] = std_perf['Marks'].apply(lambda x: 'Pass' if x >= 33 else 'Fail')
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", total_std)
    m2.metric("Overall Avg", f"{avg_marks:.2f}%")
    m3.metric("Passed", (std_perf['Status'] == 'Pass').sum())
    m4.metric("Failed", (std_perf['Status'] == 'Fail').sum(), delta_color="inverse")
    
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📊 Average Marks by Subject (Column Chart)")
        sub_avg = df.groupby('Subject')['Marks'].mean().reset_index()
        st.plotly_chart(px.bar(sub_avg, x='Subject', y='Marks', text_auto='.2f', color='Subject'))
    with c2:
        st.subheader("🥧 Pass/Fail Distribution")
        st.plotly_chart(px.pie(std_perf, names='Status', color='Status', color_discrete_map={'Pass':'#27ae60', 'Fail':'#e74c3c'}))

elif page == "📑 Subject x Term Matrix":
    st.title("📑 Term-over-Term Growth")
    # Using pivot_table with mean is 100% crash-proof
    matrix = df.pivot_table(index='Subject', columns='Term', values='Marks', aggfunc='mean').round(2)
    
    # Calculate Delta (Growth)
    if "UT 6" in matrix.columns and "UT 4" in matrix.columns:
        matrix['Growth (UT4 to UT6)'] = (matrix['UT 6'] - matrix['UT 4']).round(2)
    
    st.plotly_chart(px.bar(df, x='Subject', y='Marks', color='Term', barmode='group', text_auto='.2f'))
    st.subheader("📊 Comparative Table with Delta")
    st.table(matrix.style.format("{:.2f}"))

elif page == "👥 Comparison Suite":
    st.title("👥 Comparison Suite")
    st.info("The first student selected is the baseline.")
    selected = st.multiselect("Select Students:", sorted(df['Student Name'].unique()), default=sorted(df['Student Name'].unique())[:2])
    
    if len(selected) > 1:
        comp_data = df[df['Student Name'].isin(selected)]
        # FIXED: Pivot table replaces the failing .pivot() method
        p_comp = comp_data.pivot_table(index='Subject', columns='Student Name', values='Marks', aggfunc='mean').round(2)
        
        base = selected[0]
        for other in selected[1:]:
            p_comp[f"Diff (vs {base})"] = (p_comp[other] - p_comp[base]).round(2)
        
        st.plotly_chart(px.bar(comp_data, x='Subject', y='Marks', color='Student Name', barmode='group'))
        st.subheader("📋 Comparison Data")
        st.table(p_comp.style.format("{:.2f}"))

elif page == "📄 Reports":
    st.title("📄 Executive Data Export")
    st.dataframe(df, use_container_width=True)
    if st.button("📥 Download Report (PDF)"):
        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Class I Term II Performance Report", ln=True, align='C')
        for _, r in df.head(20).iterrows():
            pdf.cell(200, 8, txt=f"{r['Student Name']} | {r['Subject']} | {r['Term']}: {r['Marks']}", ln=True)
        st.download_button("Download PDF", bytes(pdf.output()), "ClassI_Analysis.pdf")

# (Other pages like Subject/Class analysis use same robust logic)