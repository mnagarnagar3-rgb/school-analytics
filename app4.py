import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
from fpdf import FPDF
import google.generativeai as genai

# --- 1. SETTINGS & AI CONFIG ---
st.set_page_config(page_title="EduInsight Master Pro", layout="wide")

# Configure AI (using Streamlit Secrets)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. THE PERMANENT AUTO-SYNC ENGINE ---
# This looks for ALL .xlsx files in your GitHub folder automatically
def load_all_permanent_data():
    all_data_list = []
    
    # Finds every excel file regardless of the name
    files = glob.glob("*.xlsx")
    
    if files:
        for f in files:
            try:
                # Basic cleaning to find headers (looks for 'Student Name')
                df_temp = pd.read_excel(f)
                header_row = 0
                for i, row in df_temp.iterrows():
                    if row.astype(str).str.contains('Student Name', case=False).any():
                        header_row = i + 1
                        break
                
                df = pd.read_excel(f, skiprows=header_row)
                df.columns = df.columns.astype(str).str.strip()
                
                # Class name derived from the filename
                class_name = f.replace(".xlsx", "").replace("_", " ")
                id_col = df.columns[0]
                
                # Filter out metadata columns to find pure subjects
                subjects = [c for c in df.columns if c != id_col and "Unnamed" not in c and "Teacher" not in c]
                
                # Convert to Long Format (Melt)
                melted = pd.melt(df, id_vars=[id_col], value_vars=subjects, var_name='Raw_Col', value_name='Marks')
                melted = melted.rename(columns={id_col: 'Student Name'})
                melted['Class_ID'] = class_name
                melted['Marks'] = pd.to_numeric(melted['Marks'], errors='coerce').fillna(0).round(2)
                
                # Teacher Mapping Logic
                if 'Teacher' in df.columns:
                    t_map = df[[id_col, 'Teacher']].set_index(id_col)['Teacher'].to_dict()
                    melted['Teacher'] = melted['Student Name'].map(t_map)
                else:
                    melted['Teacher'] = f"Lead Teacher ({class_name})"
                
                # Split Subject and Term (Assumes format: "Subject Term")
                melted['Term'] = melted['Raw_Col'].apply(lambda x: x.split()[-1] if ' ' in x else 'General')
                melted['Subject'] = melted['Raw_Col'].apply(lambda x: ' '.join(x.split()[:-1]) if ' ' in x else x)
                
                all_data_list.append(melted)
            except Exception as e:
                continue 
                
    if all_data_list:
        return pd.concat(all_data_list, ignore_index=True)
    return pd.DataFrame()

# Load data automatically
df = load_all_permanent_data()

# --- 3. DASHBOARD NAVIGATION ---
if not df.empty:
    st.sidebar.title("💎 EduInsight Pro")
    # Dropdown to switch between any files you uploaded to GitHub
    selected_class = st.sidebar.selectbox("🎯 Select Class", sorted(df['Class_ID'].unique()))
    view_df = df[df['Class_ID'] == selected_class]

    menu = ["🏠 Executive Overview", "📈 Individual Growth & AI", "👨‍🏫 Teacher Analysis", "🏫 Class Comparison", "📄 Reports"]
    page = st.sidebar.selectbox("📊 Navigation", menu)

    # --- 🏠 EXECUTIVE OVERVIEW (With Pass/Fail) ---
    if page == "🏠 Executive Overview":
        st.title(f"🏠 {selected_class} Summary")
        
        # Calculate Pass/Fail metrics (Threshold: 33)
        std_summary = view_df.groupby('Student Name')['Marks'].mean().reset_index()
        std_summary['Status'] = std_summary['Marks'].apply(lambda x: 'Pass' if x >= 33 else 'Fail')
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Avg", f"{view_df['Marks'].mean():.2f}%")
        k2.metric("Total Students", view_df['Student Name'].nunique())
        k3.metric("✅ Passed", (std_summary['Status'] == 'Pass').sum())
        k4.metric("❌ Failed", (std_summary['Status'] == 'Fail').sum(), delta_color="inverse")
        
        st.divider()
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(px.bar(view_df.groupby('Subject')['Marks'].mean().reset_index(), 
                                   x='Subject', y='Marks', text_auto='.2f', color='Subject', 
                                   title="Subject-Wise Average Performance"), use_container_width=True)
        with c2:
            st.plotly_chart(px.pie(std_summary, names='Status', color='Status', 
                                   color_discrete_map={'Pass':'#2ecc71', 'Fail':'#e74c3c'}, 
                                   title="Pass vs Fail Ratio"), use_container_width=True)

    # --- 📈 INDIVIDUAL GROWTH & AI (With Delta Change) ---
    elif page == "📈 Individual Growth & AI":
        st.title("📈 Student Progress & AI Feedback")
        student = st.selectbox("Select Student", sorted(view_df['Student Name'].unique()))
        s_df = view_df[view_df['Student Name'] == student]
        
        # Delta Growth Table
        pivot_s = s_df.pivot_table(index='Subject', columns='Term', values='Marks').round(2)
        if pivot_s.shape[1] > 1:
            pivot_s['Delta Growth'] = (pivot_s.iloc[:, -1] - pivot_s.iloc[:, 0]).round(2)

        st.plotly_chart(px.line(s_df, x="Term", y="Marks", color="Subject", markers=True, title=f"Progress Trend: {student}"), use_container_width=True)
        
        st.subheader("📊 Improvement Tracker (Delta)")
        st.table(pivot_s.style.format("{:.2f}").background_gradient(subset=['Delta Growth'], cmap='RdYlGn'))
        
        st.divider()
        st.subheader("🤖 AI Principal Observations")
        if st.button("Generate AI Feedback"):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                res = model.generate_content(f"Analyze the performance of {student} in {selected_class} based on: {s_df.to_string()}")
                st.success(res.text)
            except:
                st.error("AI Configuration Error. Check API Key in Secrets.")

    # --- 👨‍🏫 TEACHER ANALYSIS ---
    elif page == "👨‍🏫 Teacher Analysis":
        st.title("👨‍🏫 Teacher Subject Proficiency")
        t_perf = view_df.groupby(['Teacher', 'Subject'])['Marks'].mean().reset_index()
        st.plotly_chart(px.bar(t_perf, x='Subject', y='Marks', color='Teacher', barmode='group', 
                               text_auto='.2f', title="Performance by Subject and Teacher"), use_container_width=True)
        st.subheader("📋 Faculty Performance Matrix")
        st.dataframe(t_perf.pivot(index='Teacher', columns='Subject', values='Marks').style.format("{:.2f}"), use_container_width=True)

    # --- 🏫 CLASS COMPARISON ---
    elif page == "🏫 Class Comparison":
        st.title("🏫 School-Wide Comparative Analytics")
        if df['Class_ID'].nunique() < 2:
            st.warning("Please upload more than one class file to GitHub for comparison.")
        else:
            agg_comp = df.groupby(['Class_ID', 'Subject'])['Marks'].mean().reset_index()
            st.plotly_chart(px.bar(agg_comp, x='Subject', y='Marks', color='Class_ID', 
                                   barmode='group', text_auto='.2f', title="Inter-Class Subject Averages"), use_container_width=True)

    # --- 📄 REPORTS ---
    elif page == "📄 Reports":
        st.title("📄 Official Data Report")
        st.dataframe(view_df, use_container_width=True)
        if st.button("📥 Export PDF"):
            pdf = FPDF()
            pdf.add_page(); pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Academic Summary: {selected_class}", ln=True, align='C')
            for _, r in view_df.head(50).iterrows():
                pdf.cell(200, 8, txt=f"{r['Student Name']} | {r['Subject']} | {r['Marks']}", ln=True)
            st.download_button("Download PDF", data=bytes(pdf.output()), file_name=f"{selected_class}.pdf")

else:
    st.title("💎 EduInsight Dashboard")
    st.warning("No data found! Please upload your class Excel files (.xlsx) to your GitHub repository.")
    st.info("The system will automatically find any Excel file you upload and build the dashboard for you.")