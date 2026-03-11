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
            # Step A: Find the header row (looking for 'Student Name')
            df_temp = pd.read_excel(file)
            header_row = 0
            for i, row in df_temp.iterrows():
                if row.astype(str).str.contains('Student Name', case=False).any():
                    header_row = i + 1
                    break
            
            # Step B: Load with correct header and clean names
            df = pd.read_excel(file, skiprows=header_row)
            df.columns = df.columns.astype(str).str.strip()
            
            class_name = file.name.replace(".xlsx", "").replace("_", " ")
            id_col = df.columns[0]
            
            # Identify Subject columns (Exclude Student Name, Teacher, and empty columns)
            subjects = [c for c in df.columns if c != id_col and "Unnamed" not in c and "Teacher" not in c]
            
            # Step C: "Melt" data into long format for analysis
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
            
            # Step E: Auto-Split Subject and Term (Assumes "Subject Term" format)
            melted['Term'] = melted['Raw_Col'].apply(lambda x: x.split()[-1] if ' ' in x else 'General')
            melted['Subject'] = melted['Raw_Col'].apply(lambda x: ' '.join(x.split()[:-1]) if ' ' in x else x)
            
            all_data_list.append(melted)
            
        return pd.concat(all_data_list, ignore_index=True)
    return pd.DataFrame()

# Global Data Object
df = load_and_process_data()

# --- 3. DASHBOARD NAVIGATION ---
if not df.empty:
    st.sidebar.divider()
    # Let user pick which class they want to focus on
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

    # --- PAGE 1: EXECUTIVE OVERVIEW ---
    if page == "🏠 Executive Overview":
        st.title(f"🏠 {selected_class} Executive Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Average", f"{view_df['Marks'].mean():.2f}%")
        k2.metric("Total Students", view_df['Student Name'].nunique())
        k3.metric("Subjects Count", view_df['Subject'].nunique())
        k4.metric("Highest Avg Subject", view_df.groupby('Subject')['Marks'].mean().idxmax())
        
        st.divider()
        st.subheader("📊 Subject-Wise Performance (Mean)")
        sub_avg = view_df.groupby('Subject')['Marks'].mean().reset_index()
        st.plotly_chart(px.bar(sub_avg, x='Subject', y