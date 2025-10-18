"""
AI Resume-Job Matcher - Beginner Friendly Version
Complete working application for VS Code
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime

# Page configuration - this sets up the browser tab
st.set_page_config(
    page_title="AI Resume-Job Matcher",
    page_icon="🎯",
    layout="wide"
)

# Custom styling to make it look professional
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .skill-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.2rem;
        background-color: #e8f4fd;
        border-radius: 15px;
        font-size: 0.9rem;
        color: #1e3d59;
    }
    .success-badge {
        background-color: #d4f4dd;
        color: #0d7a2e;
    }
    .warning-badge {
        background-color: #ffe4e1;
        color: #d32f2f;
    }
    .job-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (this stores data between reruns)
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'resume_skills' not in st.session_state:
    st.session_state.resume_skills = []
if 'page' not in st.session_state:
    st.session_state.page = "Upload Resume"

def extract_skills(text):
    """
    Extract skills from resume text
    This is a simple version using keyword matching
    """
    text_lower = text.lower()
    
    # Define skills to look for (you can add more!)
    all_skills = {
        # Programming Languages
        'python': ['python', 'py'],
        'r': ['r programming', 'r language', ' r ', 'r,', 'r.'],
        'sql': ['sql', 'mysql', 'postgresql', 'sqlite'],
        'java': ['java', 'jvm'],
        'javascript': ['javascript', 'js', 'node.js', 'nodejs'],
        'c++': ['c++', 'cpp'],
        
        # Data Science & ML
        'machine learning': ['machine learning', 'ml ', 'ml,', 'ml.'],
        'deep learning': ['deep learning', 'neural network', 'dl '],
        'data analysis': ['data analysis', 'data analytics', 'analytical'],
        'statistics': ['statistics', 'statistical analysis', 'statistical'],
        'data visualization': ['visualization', 'data viz', 'visualizing'],
        'nlp': ['nlp', 'natural language processing', 'text mining'],
        'computer vision': ['computer vision', 'cv ', 'image processing'],
        
        # Tools & Frameworks
        'tensorflow': ['tensorflow', 'tf '],
        'pytorch': ['pytorch', 'torch'],
        'scikit-learn': ['scikit-learn', 'sklearn', 'sci-kit'],
        'pandas': ['pandas'],
        'numpy': ['numpy'],
        'matplotlib': ['matplotlib'],
        'tableau': ['tableau'],
        'power bi': ['power bi', 'powerbi'],
        'excel': ['excel', 'spreadsheet'],
        
        # Cloud & DevOps
        'aws': ['aws', 'amazon web services', 'ec2', 's3'],
        'azure': ['azure', 'microsoft azure'],
        'gcp': ['gcp', 'google cloud', 'google cloud platform'],
        'docker': ['docker', 'containerization'],
        'kubernetes': ['kubernetes', 'k8s'],
        'git': ['git', 'github', 'gitlab', 'version control'],
        
        # Databases
        'mongodb': ['mongodb', 'mongo'],
        'redis': ['redis'],
        'elasticsearch': ['elasticsearch', 'elastic'],
        
        # Big Data
        'spark': ['spark', 'pyspark', 'apache spark'],
        'hadoop': ['hadoop', 'hdfs', 'mapreduce'],
        'kafka': ['kafka', 'streaming'],
        'airflow': ['airflow', 'apache airflow'],
        
        # Soft Skills
        'agile': ['agile', 'scrum', 'sprint'],
        'communication': ['communication', 'communicate', 'presenting'],
        'teamwork': ['team', 'collaboration', 'collaborative'],
        'problem solving': ['problem solving', 'analytical thinking', 'troubleshoot'],
    }
    
    found_skills = []
    for skill, keywords in all_skills.items():
        for keyword in keywords:
            if keyword in text_lower and skill not in found_skills:
                found_skills.append(skill)
                break
    
    return found_skills

def get_sample_jobs():
    """
    Returns sample job postings
    In a real app, this would fetch from a database or API
    """
    return [
        {
            'id': 1,
            'title': '🎯 Data Scientist - Entry Level',
            'company': 'TechCorp Boston',
            'location': 'Boston, MA',
            'type': 'Full-time',
            'experience': '0-2 years',
            'salary': '$75,000 - $95,000',
            'description': 'Join our data science team to build ML models and derive insights from data.',
            'required_skills': ['python', 'sql', 'machine learning', 'statistics', 'data analysis'],
            'nice_to_have': ['tensorflow', 'aws', 'docker'],
            'posted': '2 days ago'
        },
        {
            'id': 2,
            'title': '📊 Data Analyst Co-op',
            'company': 'Analytics Pro',
            'location': 'Cambridge, MA',
            'type': 'Co-op',
            'experience': 'Students welcome',
            'salary': '$25-30/hour',
            'description': 'Perfect co-op opportunity for students interested in data analytics.',
            'required_skills': ['sql', 'excel', 'python', 'data visualization'],
            'nice_to_have': ['tableau', 'power bi', 'statistics'],
            'posted': '1 day ago'
        },
        {
            'id': 3,
            'title': '🔧 Data Engineer',
            'company': 'DataFlow Inc',
            'location': 'Boston, MA (Remote)',
            'type': 'Full-time',
            'experience': '2-4 years',
            'salary': '$95,000 - $120,000',
            'description': 'Build and maintain data pipelines for our growing platform.',
            'required_skills': ['python', 'sql', 'spark', 'airflow', 'docker'],
            'nice_to_have': ['aws', 'kubernetes', 'kafka'],
            'posted': '3 days ago'
        },
        {
            'id': 4,
            'title': '🤖 Machine Learning Engineer',
            'company': 'AI Innovations',
            'location': 'Remote',
            'type': 'Full-time',
            'experience': '3-5 years',
            'salary': '$110,000 - $140,000',
            'description': 'Deploy ML models to production and build scalable AI systems.',
            'required_skills': ['python', 'machine learning', 'deep learning', 'docker', 'git'],
            'nice_to_have': ['kubernetes', 'pytorch', 'tensorflow', 'aws'],
            'posted': '1 week ago'
        },
        {
            'id': 5,
            'title': '📈 Business Intelligence Analyst',
            'company': 'Finance Corp',
            'location': 'Boston, MA',
            'type': 'Full-time',
            'experience': '1-3 years',
            'salary': '$70,000 - $90,000',
            'description': 'Transform business data into actionable insights.',
            'required_skills': ['sql', 'excel', 'tableau', 'data analysis'],
            'nice_to_have': ['python', 'power bi', 'statistics'],
            'posted': '4 days ago'
        },
        {
            'id': 6,
            'title': '🎓 Data Science Intern',
            'company': 'StartupXYZ',
            'location': 'Boston, MA',
            'type': 'Internship',
            'experience': 'Students',
            'salary': '$20-25/hour',
            'description': 'Learn and grow with our data science team.',
            'required_skills': ['python', 'statistics', 'sql'],
            'nice_to_have': ['machine learning', 'pandas', 'numpy'],
            'posted': 'Today'
        }
    ]

def calculate_match_score(resume_skills, job):
    """
    Calculate how well a resume matches a job
    Returns a score from 0-100
    """
    if not resume_skills:
        return 0, [], []
    
    required = set(job['required_skills'])
    nice = set(job.get('nice_to_have', []))
    user_skills = set(resume_skills)
    
    # Find matches
    required_matches = required.intersection(user_skills)
    nice_matches = nice.intersection(user_skills)
    
    # Calculate score
    if len(required) > 0:
        required_score = len(required_matches) / len(required) * 70
    else:
        required_score = 0
    
    if len(nice) > 0:
        nice_score = len(nice_matches) / len(nice) * 30
    else:
        nice_score = 0
    
    total_score = required_score + nice_score
    
    # Find missing skills
    missing_skills = list(required - user_skills)
    
    return min(100, total_score), list(required_matches), missing_skills

# MAIN APPLICATION
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Resume-Job Matcher</h1>', unsafe_allow_html=True)
    st.markdown("### Find your perfect job match in seconds!")
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 📱 Navigation")
        
        # Page selection
        pages = ["Upload Resume", "Find Jobs", "My Matches", "Skills Analysis"]
        selected_page = st.radio("Go to:", pages, index=pages.index(st.session_state.page))
        st.session_state.page = selected_page
        
        st.markdown("---")
        
        # Quick Stats
        if st.session_state.resume_skills:
            st.markdown("### 📊 Your Stats")
            st.success(f"✅ Resume Uploaded")
            st.info(f"🛠️ {len(st.session_state.resume_skills)} skills found")
            
            # Show skills
            st.markdown("**Your Skills:**")
            for skill in st.session_state.resume_skills[:5]:
                st.write(f"• {skill}")
            if len(st.session_state.resume_skills) > 5:
                st.write(f"... and {len(st.session_state.resume_skills)-5} more")
        else:
            st.warning("📄 No resume uploaded yet")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info(
            "**Pro Tip:** Include technical skills, "
            "programming languages, and tools you've "
            "used in your resume for better matches!"
        )
    
    # Main Content Area
    if st.session_state.page == "Upload Resume":
        upload_resume_page()
    elif st.session_state.page == "Find Jobs":
        find_jobs_page()
    elif st.session_state.page == "My Matches":
        my_matches_page()
    elif st.session_state.page == "Skills Analysis":
        skills_analysis_page()

def upload_resume_page():
    """Page for uploading resume"""
    st.header("📄 Upload Your Resume")
    
    # Instructions
    st.markdown("""
    Upload your resume by either:
    1. Pasting the text below, or
    2. Using our sample resume to test the app
    """)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input area
        st.markdown("### 📝 Paste Your Resume Text")
        resume_input = st.text_area(
            "Copy and paste your resume here:",
            height=400,
            placeholder="""Paste your resume here...

Example format:
John Doe
Data Scientist

Skills: Python, SQL, Machine Learning, TensorFlow...

Experience:
- Data Scientist at Company X
- Used Python and SQL for data analysis...

Education:
- BS Computer Science..."""
        )
        
        # Analyze button
        if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
            if resume_input:
                # Extract skills
                skills = extract_skills(resume_input)
                
                # Save to session state
                st.session_state.resume_text = resume_input
                st.session_state.resume_skills = skills
                
                # Show success message
                st.success(f"✅ Resume analyzed successfully! Found {len(skills)} skills.")
                
                # Display found skills
                if skills:
                    st.markdown("### 🛠️ Skills We Found:")
                    skills_html = ""
                    for skill in skills:
                        skills_html += f'<span class="skill-badge success-badge">{skill}</span>'
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    # Suggest next step
                    st.info("👉 Go to **'Find Jobs'** page to see matching positions!")
                else:
                    st.warning("No technical skills detected. Make sure to include programming languages, tools, and technologies.")
            else:
                st.error("Please paste your resume text first!")
    
    with col2:
        # Sample resume option
        st.markdown("### 🎯 Quick Start")
        st.markdown("Don't have your resume ready? Try our sample!")
        
        if st.button("📋 Use Sample Resume", use_container_width=True):
            sample_resume = """
John Doe
Data Scientist | 3 Years Experience

SKILLS:
• Programming: Python, R, SQL, Java
• Machine Learning: TensorFlow, PyTorch, Scikit-learn
• Data Analysis: Pandas, NumPy, Statistical Analysis
• Visualization: Tableau, Matplotlib, Power BI
• Cloud: AWS, Docker, Kubernetes
• Databases: MongoDB, PostgreSQL, Redis
• Big Data: Spark, Hadoop, Kafka
• Tools: Git, Jupyter, Agile, Excel

EXPERIENCE:
Data Scientist - Tech Company (2021-Present)
• Developed machine learning models using Python and TensorFlow
• Performed data analysis and created visualizations with Tableau
• Worked with AWS for model deployment
• Used SQL for database queries and data extraction

Junior Data Analyst - StartupXYZ (2020-2021)  
• Data analysis using Python, Pandas, and NumPy
• Created dashboards in Power BI
• Statistical analysis and A/B testing
• Collaborated using Agile methodology

EDUCATION:
Bachelor of Science in Computer Science
Focus on Machine Learning and Data Science
Courses: Statistics, Deep Learning, NLP, Computer Vision

PROJECTS:
• Customer Churn Prediction using Machine Learning
• Real-time Data Pipeline with Kafka and Spark
• Docker containerization for ML model deployment
"""
            # Set the sample resume
            st.session_state.resume_text = sample_resume
            st.session_state.resume_skills = extract_skills(sample_resume)
            
            st.success("✅ Sample resume loaded! Check your skills in the sidebar.")
            st.balloons()  # Fun animation!
            
        st.markdown("---")
        
        # Tips section
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        Include these in your resume:
        - **Languages:** Python, R, SQL, Java
        - **ML/AI:** TensorFlow, PyTorch, Scikit-learn
        - **Tools:** Docker, Git, Jupyter
        - **Cloud:** AWS, Azure, GCP
        - **Databases:** MongoDB, PostgreSQL
        """)

def find_jobs_page():
    """Page for finding jobs"""
    st.header("🔍 Find Jobs")
    
    if not st.session_state.resume_skills:
        st.warning("⚠️ Please upload your resume first to get personalized matches!")
        if st.button("Go to Upload Resume"):
            st.session_state.page = "Upload Resume"
            st.rerun()
        return
    
    # Get all jobs
    jobs = get_sample_jobs()
    
    # Filter options
    st.markdown("### 🎯 Filter Jobs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        job_type = st.selectbox("Job Type", ["All", "Full-time", "Co-op", "Internship", "Remote"])
    
    with col2:
        experience = st.selectbox("Experience Level", ["All", "Entry Level", "Mid Level", "Senior"])
    
    with col3:
        sort_by = st.selectbox("Sort By", ["Best Match", "Most Recent", "Salary"])
    
    # Calculate matches for all jobs
    job_matches = []
    for job in jobs:
        score, matched_skills, missing_skills = calculate_match_score(
            st.session_state.resume_skills,
            job
        )
        job['match_score'] = score
        job['matched_skills'] = matched_skills
        job['missing_skills'] = missing_skills
        job_matches.append(job)
    
    # Sort by match score
    job_matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Display jobs
    st.markdown("### 💼 Available Positions")
    
    for job in job_matches:
        # Create an expander for each job
        match_emoji = "🟢" if job['match_score'] >= 70 else "🟡" if job['match_score'] >= 40 else "🔴"
        
        with st.expander(f"{match_emoji} {job['title']} - {job['company']} ({job['match_score']:.0f}% match)"):
            # Job header info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Match Score", f"{job['match_score']:.0f}%")
            with col2:
                st.metric("Salary", job['salary'])
            with col3:
                st.metric("Experience", job['experience'])
            with col4:
                st.metric("Posted", job['posted'])
            
            # Job details
            st.markdown(f"**📍 Location:** {job['location']}")
            st.markdown(f"**💼 Type:** {job['type']}")
            st.markdown(f"**📝 Description:** {job['description']}")
            
            # Skills section
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**✅ Your Matching Skills:**")
                if job['matched_skills']:
                    for skill in job['matched_skills']:
                        st.write(f"• {skill}")
                else:
                    st.write("No direct matches")
            
            with col_b:
                st.markdown("**📚 Skills to Learn:**")
                if job['missing_skills']:
                    for skill in job['missing_skills']:
                        st.write(f"• {skill}")
                else:
                    st.write("You have all required skills! 🎉")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                st.button(f"💾 Save Job", key=f"save_{job['id']}")
            with col2:
                st.button(f"📧 Apply Now", key=f"apply_{job['id']}")
            with col3:
                st.button(f"📊 See Details", key=f"details_{job['id']}")

def my_matches_page():
    """Page showing best matches"""
    st.header("🎯 My Best Matches")
    
    if not st.session_state.resume_skills:
        st.warning("⚠️ Please upload your resume first!")
        return
    
    # Get jobs and calculate matches
    jobs = get_sample_jobs()
    job_matches = []
    
    for job in jobs:
        score, matched_skills, missing_skills = calculate_match_score(
            st.session_state.resume_skills,
            job
        )
        if score >= 50:  # Only show good matches
            job['match_score'] = score
            job['matched_skills'] = matched_skills
            job['missing_skills'] = missing_skills
            job_matches.append(job)
    
    # Sort by match score
    job_matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    if job_matches:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Match", f"{job_matches[0]['match_score']:.0f}%")
        with col2:
            st.metric("Good Matches (>70%)", sum(1 for j in job_matches if j['match_score'] >= 70))
        with col3:
            st.metric("Total Matches", len(job_matches))
        
        # Top 3 recommendations
        st.markdown("### 🏆 Top 3 Recommendations")
        
        for i, job in enumerate(job_matches[:3], 1):
            # Create card-like display
            st.markdown(f"""
            <div class="job-card">
                <h4>#{i} {job['title']}</h4>
                <p><strong>{job['company']}</strong> | {job['location']}</p>
                <p>Match Score: {job['match_score']:.0f}% | {job['salary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick apply button
            st.button(f"Quick Apply to {job['company']}", key=f"quick_{job['id']}")
    else:
        st.info("No strong matches found. Try adding more skills to your resume!")

def skills_analysis_page():
    """Page for skills analysis"""
    st.header("📊 Skills Analysis")
    
    # Market demand data
    in_demand_skills = {
        'Python': 95,
        'SQL': 88,
        'Machine Learning': 82,
        'AWS': 78,
        'Docker': 72,
        'Tableau': 68,
        'Statistics': 65,
        'Git': 60,
        'Agile': 55,
        'Excel': 50
    }
    
    # Your skills vs market
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Most In-Demand Skills")
        
        # Create DataFrame for chart
        df = pd.DataFrame(
            list(in_demand_skills.items()),
            columns=['Skill', 'Demand Score']
        )
        
        # Bar chart
        st.bar_chart(df.set_index('Skill'))
        
    with col2:
        st.markdown("### 🎯 Your Skills Match")
        
        if st.session_state.resume_skills:
            your_skills = st.session_state.resume_skills
            
            # Check which in-demand skills you have
            have = []
            dont_have = []
            
            for skill in in_demand_skills.keys():
                if skill.lower() in [s.lower() for s in your_skills]:
                    have.append(skill)
                else:
                    dont_have.append(skill)
            
            st.markdown("**✅ Skills You Have:**")
            for skill in have:
                st.write(f"• {skill}")
            
            if dont_have:
                st.markdown("**📚 Skills to Learn:**")
                for skill in dont_have[:5]:  # Show top 5
                    st.write(f"• {skill}")
        else:
            st.info("Upload your resume to see personalized recommendations!")
    
    # Learning resources
    st.markdown("### 📚 Learning Resources")
    
    resources = {
        "Python": "https://www.python.org/about/gettingstarted/",
        "SQL": "https://www.w3schools.com/sql/",
        "Machine Learning": "https://www.coursera.org/learn/machine-learning",
        "AWS": "https://aws.amazon.com/training/",
        "Docker": "https://docs.docker.com/get-started/"
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Online Courses**")
        st.write("• Coursera")
        st.write("• Udemy")
        st.write("• edX")
    
    with col2:
        st.markdown("**Practice Platforms**")
        st.write("• Kaggle")
        st.write("• LeetCode")
        st.write("• HackerRank")
    
    with col3:
        st.markdown("**Documentation**")
        for skill, link in list(resources.items())[:3]:
            st.write(f"• [{skill}]({link})")

# Run the app
if __name__ == "__main__":
    main()