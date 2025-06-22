"""
Streamlit UI for Autonomous Resume Screening Agent
================================================

Interactive web interface for HR teams to use the autonomous resume screening agent.
Provides real-time agent execution, results visualization, and report generation.
"""

import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import zipfile

# Import our agent
from resume_agent_main import AutonomousResumeAgent, JobRequirement

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .agent-status {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .agent-thinking {
        background-color: #FFF3CD;
        border-left: 4px solid #FFC107;
    }
    .agent-success {
        background-color: #D4EDDA;
        border-left: 4px solid #28A745;
    }
    .candidate-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def get_score_color(score):
    """Get color class based on score"""
    if score >= 70:
        return "score-high"
    elif score >= 50:
        return "score-medium"
    else:
        return "score-low"

def display_agent_status(status, message):
    """Display agent status with styling"""
    css_class = "agent-thinking" if status == "thinking" else "agent-success"
    st.markdown(f'<div class="agent-status {css_class}">🤖 <strong>Agent Status:</strong> {message}</div>', 
                unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    st.title("🤖 Autonomous Resume Screening Agent")
    st.markdown("**Powered by Advanced AI • Built for Onelogica**")
    
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key to enable the AI agent")
        
        if api_key:
            if st.session_state.agent is None:
                try:
                    st.session_state.agent = AutonomousResumeAgent(api_key)
                    st.success("✅ Agent initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize agent: {e}")
        
        st.markdown("---")
        
        # File upload for resumes
        st.header("📄 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF/TXT)",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple resume files for screening"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files uploaded")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Job Requirements")
        
        # Job requirements form
        with st.form("job_requirements_form"):
            job_title = st.text_input("Job Title", value="Senior AI Engineer")
            department = st.text_input("Department", value="Engineering")
            
            col_a, col_b = st.columns(2)
            with col_a:
                required_skills = st.text_area(
                    "Required Skills (one per line)",
                    value="Python\nMachine Learning\nTensorFlow\nAWS\nSQL"
                )
            with col_b:
                preferred_skills = st.text_area(
                    "Preferred Skills (one per line)",
                    value="PyTorch\nKubernetes\nReact\nLeadership"
                )
            
            experience_years = st.slider("Required Experience (years)", 0, 20, 5)
            education_level = st.selectbox(
                "Education Level",
                ["Bachelor's Degree", "Master's Degree", "PhD", "Any"]
            )
            
            job_description = st.text_area(
                "Job Description",
                value="We're looking for a Senior AI Engineer to lead our ML initiatives and build scalable AI solutions.",
                height=100
            )
            
            location = st.text_input("Location", value="Remote/San Francisco")
            
            # Submit button
            submit_job = st.form_submit_button("🚀 Start Autonomous Screening", 
                                               disabled=not (api_key and uploaded_files))
    
    with col2:
        st.header("🎯 Quick Stats")
        
        if st.session_state.screening_results:
            results = st.session_state.screening_results
            
            # Display key metrics
            total_resumes = len(results)
            top_candidates = len([r for r in results if r.overall_score > 0.7])
            avg_score = sum(r.overall_score for r in results) / len(results) if results else 0
            
            st.metric("📊 Total Resumes", total_resumes)
            st.metric("⭐ Top Candidates", top_candidates)
            st.metric("📈 Average Score", f"{avg_score*100:.1f}%")
            
            # Score distribution chart
            scores = [r.overall_score * 100 for r in results]
            fig = px.histogram(
                x=scores, 
                nbins=10,
                title="Score Distribution",
                labels={'x': 'Score (%)', 'y': 'Count'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload resumes and submit job requirements to see analytics")
    
    # Process the screening when form is submitted
    if submit_job and st.session_state.agent and uploaded_files:
        st.session_state.processing = True
        
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Prepare job requirements
            job_data = {
                "title": job_title,
                "department": department,
                "required_skills": [skill.strip() for skill in required_skills.split('\n') if skill.strip()],
                "preferred_skills": [skill.strip() for skill in preferred_skills.split('\n') if skill.strip()],
                "experience_years": experience_years,
                "education_level": education_level,
                "description": job_description,
                "location": location
            }
            
            # Show progress
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            try:
                # Step 1: Initialize screening
                with status_placeholder.container():
                    display_agent_status("thinking", "Analyzing job requirements...")
                progress_bar.progress(20)
                
                # Step 2: Process resumes
                with status_placeholder.container():
                    display_agent_status("thinking", "Scanning and parsing resume pool...")
                progress_bar.progress(40)
                
                # Step 3: Run autonomous screening
                with status_placeholder.container():
                    display_agent_status("thinking", "Agent reasoning and scoring candidates...")
                progress_bar.progress(70)
                
                results = st.session_state.agent.autonomous_screening(job_data, temp_dir)
                
                # Step 4: Complete
                with status_placeholder.container():
                    display_agent_status("success", "Autonomous screening completed successfully!")
                progress_bar.progress(100)
                
                st.session_state.screening_results = results
                st.session_state.processing = False
                
                # Auto-scroll to results
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"❌ Screening failed: {e}")
                st.session_state.processing = False
    
    # Display results
    if st.session_state.screening_results and not st.session_state.processing:
        st.markdown("---")
        st.header("🏆 Screening Results")
        
        results = st.session_state.screening_results
        
        # Generate and display report
        report = st.session_state.agent.generate_report(results, top_n=3)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Resumes Processed", report["screening_summary"]["total_resumes_processed"])
        with col2:
            st.metric("🎯 Top Candidates", report["screening_summary"]["top_candidates_selected"])
        with col3:
            st.metric("📊 Average Score", f"{report['screening_summary']['average_score']*100:.1f}%")
        with col4:
            st.metric("⏱️ Processing Time", "< 2 minutes")
        
        # Top candidates
        st.subheader("🥇 Top 3 Candidates")
        
        for i, candidate in enumerate(report["top_candidates"]):
            with st.expander(f"#{candidate['rank']} {candidate['candidate_name']} - {candidate['overall_score']}% Match", expanded=i==0):
                
                # Candidate overview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📧 Email:** {candidate['contact_email']}")
                    st.markdown(f"**📞 Phone:** {candidate['contact_phone']}")
                    st.markdown(f"**💼 Experience:** {candidate['experience_years']} years")
                    st.markdown(f"**📄 Resume:** {candidate['resume_filename']}")
                
                with col2:
                    # Score visualization
                    scores_data = {
                        'Metric': ['Overall', 'Skills', 'Experience'],
                        'Score': [candidate['overall_score'], candidate['skill_match_score'], candidate['experience_score']]
                    }
                    
                    fig = px.bar(
                        scores_data, 
                        x='Score', 
                        y='Metric',
                        orientation='h',
                        title="Candidate Scores",
                        color='Score',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=200, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Skills and strengths
                st.markdown("**🎯 Key Skills:**")
                skills_text = ", ".join(candidate['key_skills'][:8])
                st.markdown(f"<small>{skills_text}</small>", unsafe_allow_html=True)
                
                # Strengths and concerns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Key Strengths:**")
                    for strength in candidate['key_strengths']:
                        st.markdown(f"• {strength}")
                
                with col2:
                    st.markdown("**⚠️ Potential Concerns:**")
                    if candidate['potential_concerns']:
                        for concern in candidate['potential_concerns']:
                            st.markdown(f"• {concern}")
                    else:
                        st.markdown("• No major concerns identified")
                
                # AI reasoning
                st.markdown("**🤖 AI Agent Analysis:**")
                st.markdown(f"*{candidate['ai_reasoning']}*")
        
        # Download options
        st.markdown("---")
        st.subheader("📥 Download Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON report
            json_report = json.dumps(report, indent=2)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_report,
                file_name=f"screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export
            candidates_df = pd.DataFrame(report["top_candidates"])
            csv_data = candidates_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data,
                file_name=f"top_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Summary report
            summary_text = f"""
AUTONOMOUS RESUME SCREENING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Total Resumes Processed: {report['screening_summary']['total_resumes_processed']}
- Top Candidates Selected: {report['screening_summary']['top_candidates_selected']}
- Average Score: {report['screening_summary']['average_score']*100:.1f}%

TOP 3 CANDIDATES:
"""
            for candidate in report["top_candidates"]:
                summary_text += f"""
{candidate['rank']}. {candidate['candidate_name']} ({candidate['overall_score']}%)
   Email: {candidate['contact_email']}
   Key Strengths: {'; '.join(candidate['key_strengths'][:2])}
"""
            
            st.download_button(
                label="📝 Download Summary",
                data=summary_text,
                file_name=f"screening_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()