import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
import google.generativeai as genai

# --- 1. CONFIG & SECURE AI SETUP ---
st.set_page_config(page_title="EduInsight School Analytics", layout="wide")

# Fetch key securely from Streamlit Cloud Secrets
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = "DEVELOPMENT_MODE"

genai.configure(api_key=API_KEY)

def get_ai_feedback(data_summary, role="Student"):
    try:
        # Using 'gemini-1.5-flash' stable version to avoid 404 errors
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = f"Role: Academic Analyst. Analyze this {role} data and provide 3 actionable insights: {data_summary}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

# --- 2. DATA PROCESSING ---
@st.cache_data
def load_and_process_data():
    file_path = "Student_Marks.xlsx"
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        subjects = [c for c in df.columns if c in ['English', 'Hindi', 'Math', 'Maths', 'Science', 'Social']]
        df_melted = pd.melt(df, id_vars=['Student_Name', 'Year', 'Class', 'Section', 'Term', 'Teacher'], 
                            value_vars=subjects, var_name='Subject', value_name='Marks')
        df_melted['Marks'] = pd.to_numeric(df_melted['Marks'], errors='coerce').fillna(0).round(2)
        # Sorting logic
        term_order = {'H.Y': 1, 'Final': 2, 'ANN': 2, 'Term 1': 1, 'Term 2': 2}
        df_melted['Term_Rank'] = df_melted['Term'].map(term_order).fillna(3)
        df_melted = df_melted.sort_values(by=['Student_Name', 'Subject', 'Year', 'Term_Rank'])
        df_melted['Prev'] = df_melted.groupby(['Student_Name', 'Subject'])['Marks'].shift(1)
        df_melted['Diff'] = (df_melted['Marks'] - df_melted['Prev']).round(2)
        df_melted['Change'] = df_melted['Diff'].apply(lambda x: f"{'+' if x>0 else ''}{x:.2f}%" if pd.notna(x) and x != 0 else "-")
        return df_melted
    except Exception as e:
        st.error(f"Data Error: {e}")
        return pd.DataFrame()

df = load_and_process_data()
if df.empty:
    st.error("⚠️ 'Student_Marks.xlsx' not found. Ensure it is uploaded to GitHub.")
    st.stop()

# --- 3. SIDEBAR & FILTERS ---
page = st.sidebar.selectbox("Select Page:", ["🏠 Overview", "📈 Growth & AI Feedback", "👥 Compare", "👨‍🏫 Teacher Analysis"])

st.markdown("### 🔍 Global Filters")
c1, c2, c3 = st.columns(3)
y_sel = c1.selectbox("Year", sorted(df['Year'].unique(), reverse=True))
cl_sel = c2.selectbox("Class", sorted(df['Class'].unique()))
sc_sel = c3.selectbox("Section", sorted(df['Section'].unique()))
base_df = df[(df['Year'] == y_sel) & (df['Class'] == cl_sel) & (df['Section'] == sc_sel)]

# --- 4. PAGE LOGIC ---
if page == "📈 Growth & AI Feedback":
    student = st.selectbox("Select Student:", sorted(base_df['Student_Name'].unique()))
    s_hist = df[df['Student_Name'] == student].sort_values(['Year', 'Term_Rank'])
    st.plotly_chart(px.line(s_hist, x="Term", y="Marks", color="Subject", markers=True), use_container_width=True)
    if st.button("🤖 Generate AI Analysis"):
        with st.spinner("AI is analyzing performance..."):
            data_str = s_hist[['Term', 'Subject', 'Marks', 'Change']].to_string()
            feedback = get_ai_feedback(data_str, role="Student")
            st.info(feedback)

elif page == "👨‍🏫 Teacher Analysis":
    teacher_sel = st.selectbox("Select Teacher:", base_df['Teacher'].unique() if not base_df.empty else [])
    if teacher_sel:
        t_avg = base_df[base_df['Teacher'] == teacher_sel].groupby(["Subject", "Term"])["Marks"].mean().round(2).reset_index()
        t_pivot = t_avg.pivot(index="Subject", columns="Term", values="Marks")
        st.write(f"### 📉 Class Progress under {teacher_sel}")
        st.table(t_pivot)
        if st.button("🧠 Generate Faculty Insights"):
            with st.spinner("Analyzing class trends..."):
                feedback = get_ai_feedback(t_pivot.to_string(), role="Teacher")
                st.success(feedback)