import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
from fpdf import FPDF
import google.generativeai as genai

# --- 1. CONFIG & AI SETUP ---
st.set_page_config(page_title="EduInsight School Analytics", layout="wide")

# Replace with your actual Gemini API Key from Google AI Studio
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. DATA PROCESSING ---
@st.cache_data
def load_and_process_data():
    file_path = "Student_Marks.xlsx"
    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        if 'Sec' in df.columns: df = df.rename(columns={'Sec': 'Section'})
        
        subjects = [c for c in df.columns if c in ['English', 'Hindi', 'Math', 'Maths', 'Science', 'Social']]
        
        df_melted = pd.melt(df, 
                     id_vars=['Student_Name', 'Year', 'Class', 'Section', 'Term', 'Teacher'], 
                     value_vars=subjects, 
                     var_name='Subject', 
                     value_name='Marks')
        
        df_melted['Marks'] = pd.to_numeric(df_melted['Marks'], errors='coerce').fillna(0).round(2)

        # Term Sorting Logic
        term_order = {'H.Y': 1, 'Final': 2, 'ANN': 2, 'Term 1': 1, 'Term 2': 2}
        df_melted['Term_Rank'] = df_melted['Term'].map(term_order).fillna(3)
        df_melted = df_melted.sort_values(by=['Student_Name', 'Subject', 'Year', 'Term_Rank'])
        
        # Internal Growth Delta (For Growth Tracker)
        df_melted['Prev'] = df_melted.groupby(['Student_Name', 'Subject'])['Marks'].shift(1)
        df_melted['Diff'] = (df_melted['Marks'] - df_melted['Prev']).round(2)
        df_melted['Change'] = df_melted['Diff'].apply(lambda x: f"{'+' if x>0 else ''}{x:.2f}%" if pd.notna(x) and x != 0 else "-")
        
        return df_melted
    except Exception as e:
        st.error(f"Data Error: {e}")
        return pd.DataFrame()

df = load_and_process_data()

if df.empty:
    st.error("⚠️ 'Student_Marks.xlsx' not found.")
    st.stop()

# --- 3. SIDEBAR ---
page = st.sidebar.selectbox("Select Page:", ["🏠 Overview", "📈 Growth & AI Feedback", "👥 Compare", "👨‍🏫 Teacher Analysis", "📄 Reports & Download"])

# --- 4. GLOBAL FILTERS ---
st.markdown("### 🔍 Global Filters")
c1, c2, c3 = st.columns(3)
y_sel = c1.selectbox("Year", sorted(df['Year'].unique(), reverse=True))
cl_sel = c2.selectbox("Class", sorted(df['Class'].unique()))
sc_sel = c3.selectbox("Section", sorted(df['Section'].unique()))
base_df = df[(df['Year'] == y_sel) & (df['Class'] == cl_sel) & (df['Section'] == sc_sel)]

# --- 5. PAGE LOGIC ---

if page == "🏠 Overview":
    t_sel = st.selectbox("Select Term:", base_df['Term'].unique() if not base_df.empty else ["No Data"])
    v_df = base_df[base_df['Term'] == t_sel]
    if not v_df.empty:
        avg_data = v_df.groupby("Subject")["Marks"].mean().round(2).reset_index()
        st.plotly_chart(px.bar(avg_data, x="Subject", y="Marks", color="Subject", text_auto='.2f'), use_container_width=True)
        st.table(avg_data.set_index('Subject'))

elif page == "📈 Growth & AI Feedback":
    student = st.selectbox("Select Student:", sorted(base_df['Student_Name'].unique()))
    s_hist = df[df['Student_Name'] == student].sort_values(['Year', 'Term_Rank'])
    st.plotly_chart(px.line(s_hist, x="Term", y="Marks", color="Subject", markers=True), use_container_width=True)
    st.dataframe(s_hist[['Year', 'Term', 'Subject', 'Marks', 'Change']], hide_index=True)
    
    if st.button("🤖 Generate AI Analysis"):
        model = genai.GenerativeModel("gemini-1.5-flash")
        data_str = s_hist[['Term', 'Subject', 'Marks']].to_string()
        response = model.generate_content(f"Analyze this student data and give professional 3-point feedback: {data_str}")
        st.info(response.text)

elif page == "👥 Compare":
    st.title("👥 Comparison Suite")
    mode = st.radio("Mode:", ["Multi-Student Comparison", "Single Student Term-over-Term"])
    
    if mode == "Multi-Student Comparison":
        t_sel = st.selectbox("Term:", base_df['Term'].unique() if not base_df.empty else [])
        selected = st.multiselect("Select Students (First one is Baseline):", sorted(base_df['Student_Name'].unique()), default=sorted(base_df['Student_Name'].unique())[:2])
        
        if selected:
            # Pivot the marks
            c_df = base_df[(base_df['Term'] == t_sel) & (base_df['Student_Name'].isin(selected))]
            pivot_df = c_df.pivot(index='Subject', columns='Student_Name', values='Marks').round(2)
            
            # --- RESTORED MULTI-STUDENT DELTA ---
            if len(selected) > 1:
                baseline_std = selected[0]
                st.info(f"💡 Comparison showing difference relative to **{baseline_std}**")
                for other_std in selected[1:]:
                    pivot_df[f"Delta (Vs {other_std})"] = (pivot_df[other_std] - pivot_df[baseline_std]).apply(
                        lambda x: f"⬆️ +{x:.2f}" if x > 0 else (f"⬇️ {x:.2f}" if x < 0 else "➖ 0.00")
                    )
            st.table(pivot_df)
            st.plotly_chart(px.bar(c_df, x="Subject", y="Marks", color="Student_Name", barmode="group", text_auto='.2f'), use_container_width=True)

    else:
        student = st.selectbox("Student:", sorted(base_df['Student_Name'].unique()))
        s_data = df[df['Student_Name'] == student]
        t1 = st.selectbox("Old Term:", s_data['Term'].unique(), index=0)
        t2 = st.selectbox("New Term:", s_data['Term'].unique(), index=min(1, len(s_data['Term'].unique())-1))
        pivot = s_data[s_data['Term'].isin([t1, t2])].pivot_table(index='Subject', columns='Term', values='Marks').round(2)
        if t1 in pivot.columns and t2 in pivot.columns:
            pivot['Net Change'] = (pivot[t2] - pivot[t1]).round(2)
            pivot['Growth Status'] = pivot['Net Change'].apply(lambda x: f"⬆️ +{x:.2f}%" if x > 0 else (f"⬇️ {x:.2f}%" if x < 0 else "➖ 0.00%"))
            st.table(pivot)

elif page == "👨‍🏫 Teacher Analysis":
    teacher_sel = st.selectbox("Select Teacher:", base_df['Teacher'].unique())
    t_avg = base_df[base_df['Teacher'] == teacher_sel].groupby(["Subject", "Term"])["Marks"].mean().round(2).reset_index()
    
    # --- TEACHER DELTA ---
    st.write(f"### 📉 Class Progress under {teacher_sel}")
    t_pivot = t_avg.pivot(index="Subject", columns="Term", values="Marks")
    if t_pivot.shape[1] > 1:
        terms = list(t_pivot.columns)
        t_pivot['Delta'] = (t_pivot[terms[-1]] - t_pivot[terms[0]]).round(2)
        t_pivot['Status'] = t_pivot['Delta'].apply(lambda x: f"🟢 +{x:.2f}%" if x > 0 else (f"🔴 {x:.2f}%" if x < 0 else "🟡 Stable"))
    st.table(t_pivot)
    st.plotly_chart(px.bar(t_avg, x="Subject", y="Marks", color="Term", barmode="group", text_auto='.2f'), use_container_width=True)

elif page == "📄 Reports & Download":
    st.title("📄 Report Center")
    report_data = base_df if not base_df.empty else df
    st.dataframe(report_data[['Student_Name', 'Term', 'Subject', 'Marks', 'Change', 'Teacher']], use_container_width=True, hide_index=True)
    
    c_ex, c_pdf = st.columns(2)
    # Excel
    out_ex = io.BytesIO()
    with pd.ExcelWriter(out_ex, engine='xlsxwriter') as wr:
        report_data.to_excel(wr, index=False)
    c_ex.download_button("📥 Excel Report", out_ex.getvalue(), "Report.xlsx")

    # PDF
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, "Academic Report", ln=True, align='C')
        pdf.set_font("Helvetica", size=10)
        for _, row in report_data.head(50).iterrows():
            pdf.cell(0, 7, f"{row['Student_Name']} | {row['Subject']} | {row['Term']}: {row['Marks']}%", ln=True)
        c_pdf.download_button("📑 PDF Report", bytes(pdf.output()), "Report.pdf", "application/pdf")
    except Exception as e:
        st.error(f"PDF Error: {e}")