import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
from fpdf import FPDF
import google.generativeai as genai

# --- 1. SETTINGS & AI CONFIG ---
st.set_page_config(page_title="EduInsight Global Pro", layout="wide")

# Connect to Gemini AI using Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. MULTI-FILE DYNAMIC ENGINE ---
def load_and_process_data():
    st.sidebar.header("📁 Data Management")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Class Excel Sheets (Drop multiple files here)", 
        type=["xlsx"], 
        accept_multiple_files=True
    )
    
    all_data_list = []
    
    if uploaded_files:
        for file in uploaded_files:
            # Step A: Load and find the header row
            df_temp = pd.read_excel(file)
            header_row = 0
            for i, row in df_temp.iterrows():
                if row.astype(str).str.contains('Student Name', case=False).any():
                    header_row = i + 1
                    break
            
            # Step B: Reload with correct header
            df = pd.read_excel(file, skiprows=header_row)
            df.columns = df.columns.astype(str).str.strip()
            
            class_name = file.name.replace(".xlsx", "").replace("_", " ")
            id_col = df.columns[0]
            
            # Identify Subject columns
            subjects = [c for c in df.columns if c != id_col and "Unnamed" not in c and "Teacher" not in c]
            
            # Step C: Melt data for analysis
            melted = pd.melt(df, id_vars=[id_col], value_vars=subjects, var_name='Raw_Col', value_name='Marks')
            melted = melted.rename(columns={id_col: 'Student Name'})
            melted['Class_ID'] = class_name
            melted['Marks'] = pd.to_numeric(melted['Marks'], errors='coerce').fillna(0).round(2)
            
            # Step D: Map Teachers
            if 'Teacher' in df.columns:
                t_map = df[[id_col, 'Teacher']].set_index(id_col)['Teacher'].to_dict()
                melted['Teacher'] = melted['Student Name'].map(t_map)
            else:
                melted['Teacher'] = f"Lead Teacher ({class_name})"
            
            # Step E: Auto-Split Subject and Term
            melted['Term'] = melted['Raw_Col'].apply(lambda x: x.split()[-1] if ' ' in x else 'General')
            melted['Subject'] = melted['Raw_Col'].apply(lambda x: ' '.join(x.split()[:-1]) if ' ' in x else x)
            
            all_data_list.append(melted)
            
        return pd.concat(all_data_list, ignore_index=True)
    return pd.DataFrame()

df = load_and_process_data()

# --- 3. DASHBOARD NAVIGATION ---
if not df.empty:
    st.sidebar.divider()
    selected_class = st.sidebar.selectbox("🎯 Focus on Class", df['Class_ID'].unique())
    view_df = df[df['Class_ID'] == selected_class]

    menu = [
        "🏠 Executive Overview", 
        "📈 Student Growth & AI", 
        "👨‍🏫 Teacher Subject Analysis", 
        "🏫 Class Comparison", 
        "📄 Reports & Downloads"
    ]
    page = st.sidebar.selectbox("📊 Navigation", menu)

    # --- EXECUTIVE OVERVIEW ---
    if page == "🏠 Executive Overview":
        st.title(f"🏠 {selected_class} Executive Summary")
        k1, k2, k3 = st.columns(3)
        k1.metric("Overall Average", f"{view_df['Marks'].mean():.2f}%")
        k2.metric("Total Students", view_df['Student Name'].nunique())
        k3.metric("Faculty Count", view_df['Teacher'].nunique())
        
        st.divider()
        st.subheader("📊 Subject-Wise Performance")
        sub_avg = view_df.groupby('Subject')['Marks'].mean().reset_index()
        # FIXED LINE BELOW
        st.plotly_chart(px.bar(sub_avg, x='Subject', y='Marks', text_auto='.2f', color='Subject'))

    # --- STUDENT GROWTH & AI ---
    elif page == "📈 Student Growth & AI":
        st.title("📈 Individual Student Analytics")
        student = st.selectbox("Select Student", sorted(view_df['Student Name'].unique()))
        s_df = view_df[view_df['Student Name'] == student]
        st.plotly_chart(px.line(s_df, x="Term", y="Marks", color="Subject", markers=True))
        
        if st.button("🤖 Generate AI Insights"):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"Analyze student {student}: {s_df.to_string()}"
                response = model.generate_content(prompt)
                st.info(response.text)
            except:
                st.warning("Please configure your Gemini API Key.")

    # --- TEACHER ANALYSIS ---
    elif page == "👨‍🏫 Teacher Subject Analysis":
        st.title("👨‍🏫 Faculty Performance")
        t_perf = view_df.groupby(['Teacher', 'Subject'])['Marks'].mean().reset_index()
        st.plotly_chart(px.bar(t_perf, x='Subject', y='Marks', color='Teacher', barmode='group', text_auto='.2f'))
        matrix = t_perf.pivot(index='Teacher', columns='Subject', values='Marks').round(2)
        st.dataframe(matrix.style.format("{:.2f}"), use_container_width=True)

    # --- CLASS COMPARISON ---
    elif page == "🏫 Class Comparison":
        st.title("🏫 School-Wide Comparison")
        if df['Class_ID'].nunique() < 2:
            st.warning("Upload multiple files to compare classes.")
        else:
            agg_comp = df.groupby(['Class_ID', 'Subject'])['Marks'].mean().reset_index()
            st.plotly_chart(px.bar(agg_comp, x='Subject', y='Marks', color='Class_ID', barmode='group', text_auto='.2f'))

    # --- REPORTS ---
    elif page == "📄 Reports & Downloads":
        st.title("📄 Academic Reports")
        st.dataframe(view_df, use_container_width=True)
        if st.button("📥 Download PDF"):
            pdf = FPDF()
            pdf.add_page(); pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Report: {selected_class}", ln=True, align='C')
            for _, r in view_df.head(50).iterrows():
                pdf.cell(200, 8, txt=f"{r['Student Name']} | {r['Subject']}: {r['Marks']}", ln=True)
            st.download_button("Save PDF", data=bytes(pdf.output()), file_name=f"{selected_class}.pdf")

else:
    st.title("👋 Welcome to EduInsight Pro")
    st.info("Please upload your Class Excel file(s) in the sidebar to begin.")