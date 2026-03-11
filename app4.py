import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
from fpdf import FPDF
import google.generativeai as genai

# --- 1. CONFIG & AI SETUP ---
st.set_page_config(page_title="EduInsight Pro | School Analytics", layout="wide")

# Secure API Key handling for live link
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. DATA PROCESSING ---
@st.cache_data
def load_and_process_data():
    file_path = "Student_Marks.xlsx"
    if not os.path.exists(file_path): return pd.DataFrame()
    
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    if 'Sec' in df.columns: df = df.rename(columns={'Sec': 'Section'})
    
    subjects = [c for c in df.columns if c in ['English', 'Hindi', 'Math', 'Maths', 'Science', 'Social']]
    df_melted = pd.melt(df, id_vars=['Student_Name', 'Year', 'Class', 'Section', 'Term', 'Teacher'], 
                         value_vars=subjects, var_name='Subject', value_name='Marks')
    
    # Strictly round to 2 decimals at the source
    df_melted['Marks'] = pd.to_numeric(df_melted['Marks'], errors='coerce').fillna(0).round(2)
    
    # Term Sorting Logic
    term_order = {'H.Y': 1, 'Final': 2, 'ANN': 2, 'Term 1': 1, 'Term 2': 2}
    df_melted['Term_Rank'] = df_melted['Term'].map(term_order).fillna(3)
    return df_melted

df = load_and_process_data()

# --- 3. PROFESSIONAL SIDEBAR NAVIGATION ---
st.sidebar.title("EduInsight Menu")
page = st.sidebar.selectbox("Choose Analysis View:", [
    "🏠 Executive Overview", 
    "📊 Class Analysis", 
    "📚 Subject Analysis", 
    "📅 Class Term Analysis", 
    "📑 Class Subject Term Analysis",
    "📈 Individual Growth", 
    "👥 Compare Students", 
    "👨‍🏫 Teacher Analysis", 
    "📄 Reports & Download"
])

# Global Filters
st.sidebar.divider()
y_sel = st.sidebar.selectbox("Year", sorted(df['Year'].unique(), reverse=True))
cl_sel = st.sidebar.selectbox("Class", sorted(df['Class'].unique()))
sc_sel = st.sidebar.selectbox("Section", sorted(df['Section'].unique()))
base_df = df[(df['Year'] == y_sel) & (df['Class'] == cl_sel) & (df['Section'] == sc_sel)]

# --- 4. PAGE LOGIC ---

if page == "🏠 Executive Overview":
    st.title(f"🏫 Executive Dashboard: {cl_sel}-{sc_sel} ({y_sel})")
    
    # KPI Calculations
    total_students = base_df['Student_Name'].nunique()
    overall_avg = base_df['Marks'].mean()
    
    # Pass/Fail logic (using 33% as threshold)
    std_avg = base_df.groupby('Student_Name')['Marks'].mean()
    passed_count = (std_avg >= 33).sum()
    failed_count = total_students - passed_count
    
    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total_students)
    c2.metric("Class Average", f"{overall_avg:.2f}%")
    c3.metric("Passed", passed_count, delta_color="normal")
    c4.metric("Failed", failed_count, delta="-", delta_color="inverse")
    
    st.divider()
    
    # Subject Highlights Table
    st.subheader("🏆 Subject Performance Highlights")
    sub_stats = base_df.groupby('Subject')['Marks'].agg(['max', 'min', 'mean']).round(2)
    sub_stats.columns = ['Highest', 'Lowest', 'Average']
    st.table(sub_stats.style.format("{:.2f}"))

elif page == "📊 Class Analysis":
    st.title("📊 Detailed Class Analysis")
    fig = px.bar(base_df.groupby('Subject')['Marks'].mean().reset_index(), 
                 x='Subject', y='Marks', color='Subject', text_auto='.2f', title="Average Marks per Subject")
    st.plotly_chart(fig, use_container_width=True)

elif page == "📚 Subject Analysis":
    st.title("📚 Subject Performance Deep-Dive")
    sub_choice = st.selectbox("Select Subject", base_df['Subject'].unique())
    sub_df = base_df[base_df['Subject'] == sub_choice]
    st.plotly_chart(px.histogram(sub_df, x="Marks", title=f"Score Distribution for {sub_choice}"))

elif page == "📅 Class Term Analysis":
    st.title("📅 Progress Across Terms")
    term_trend = base_df.groupby('Term')['Marks'].mean().reset_index()
    st.plotly_chart(px.line(term_trend, x='Term', y='Marks', markers=True, title="Term-wise Class Growth"))

elif page == "📑 Class Subject Term Analysis":
    st.title("📑 Term-wise Subject Analysis")
    term_pivot = base_df.pivot_table(index='Subject', columns='Term', values='Marks', aggfunc='mean').round(2)
    st.table(term_pivot.style.format("{:.2f}"))

elif page == "📈 Individual Growth":
    student = st.selectbox("Select Student", sorted(base_df['Student_Name'].unique()))
    s_hist = df[df['Student_Name'] == student].sort_values('Term_Rank')
    st.plotly_chart(px.line(s_hist, x="Term", y="Marks", color="Subject", markers=True))
    
    if st.button("🤖 AI Analysis"):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"Analyze this student's marks and provide a 3-point summary: {s_hist.to_string()}")
            st.info(response.text)
        except:
            st.error("AI Key not configured or invalid.")

elif page == "👥 Compare Students":
    st.title("👥 Comparison Suite")
    selected = st.multiselect("Select Students (First is Baseline):", sorted(base_df['Student_Name'].unique()), default=sorted(base_df['Student_Name'].unique())[:2])
    if len(selected) > 1:
        comp_df = base_df[base_df['Student_Name'].isin(selected)].pivot(index='Subject', columns='Student_Name', values='Marks')
        for other in selected[1:]:
            comp_df[f"Delta ({other} vs {selected[0]})"] = (comp_df[other] - comp_df[selected[0]]).round(2)
        st.table(comp_df.style.format("{:.2f}"))

elif page == "👨‍🏫 Teacher Analysis":
    st.title("👨‍🏫 Teacher-wise Impact")
    teacher = st.selectbox("Select Teacher", base_df['Teacher'].unique())
    t_data = base_df[base_df['Teacher'] == teacher].groupby('Subject')['Marks'].mean().reset_index()
    st.table(t_data.set_index('Subject').style.format("{:.2f}"))

elif page == "📄 Reports & Download":
    st.title("📄 Report Center")
    st.dataframe(base_df, hide_index=True)
    
    # Export PDF Button
    if st.button("📑 Download PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Academic Report Summary", ln=True, align='C')
        for i, row in base_df.head(20).iterrows():
            pdf.cell(200, 10, txt=f"{row['Student_Name']} - {row['Subject']}: {row['Marks']}", ln=True)
        st.download_button("Download File", bytes(pdf.output()), "School_Report.pdf")