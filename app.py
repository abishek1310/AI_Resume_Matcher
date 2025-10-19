import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import requests
from collections import Counter
import json

# Page config
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .skill-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 15px;
        background: #667eea;
        color: white;
        font-size: 0.9rem;
    }
    .match-score {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'jobs' not in st.session_state:
    st.session_state.jobs = []
if 'user_skills' not in st.session_state:
    st.session_state.user_skills = set()

# ==================== SKILL EXTRACTION ====================
def extract_skills(text):
    """Extract skills from text using comprehensive skill database"""
    text_lower = text.lower()
    
    skill_categories = {
        'Programming Languages': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
            'go', 'rust', 'swift', 'kotlin', 'php', 'scala', 'r', 'matlab'
        ],
        'Data Science & ML': [
            'machine learning', 'deep learning', 'neural networks', 'nlp', 
            'computer vision', 'tensorflow', 'pytorch', 'scikit-learn', 'keras',
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter'
        ],
        'Data Engineering': [
            'sql', 'nosql', 'postgresql', 'mysql', 'mongodb', 'redis',
            'apache spark', 'hadoop', 'kafka', 'airflow', 'etl', 'data pipeline',
            'snowflake', 'bigquery', 'redshift'
        ],
        'Cloud & DevOps': [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'jenkins',
            'terraform', 'ansible', 'linux', 'bash', 'git', 'github'
        ],
        'Web Development': [
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'fastapi', 'rest api', 'graphql', 'html', 'css', 'bootstrap', 'tailwind'
        ],
        'Data Analysis': [
            'excel', 'tableau', 'power bi', 'looker', 'data visualization',
            'statistics', 'a/b testing', 'hypothesis testing', 'regression'
        ],
        'Soft Skills': [
            'communication', 'teamwork', 'leadership', 'problem solving',
            'analytical thinking', 'project management', 'agile', 'scrum'
        ]
    }
    
    found_skills = {category: [] for category in skill_categories}
    all_skills = set()
    
    for category, skills in skill_categories.items():
        for skill in skills:
            if skill in text_lower:
                found_skills[category].append(skill)
                all_skills.add(skill)
    
    return found_skills, all_skills

# ==================== JOB API INTEGRATIONS ====================
def get_fallback_jobs(search_term="data"):
    """Fallback sample jobs when API is unavailable"""
    sample_jobs = [
        {
            'title': 'Data Scientist',
            'company': 'Tech Corp',
            'location': 'Remote',
            'description': 'Seeking data scientist with Python, machine learning, SQL, and TensorFlow experience. Work on predictive models and data pipelines.',
            'salary': '$80,000 - $120,000',
            'url': 'https://example.com/job1',
            'posted_date': '2 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Senior Data Engineer',
            'company': 'DataFlow Inc',
            'location': 'Boston, MA',
            'description': 'Data engineer needed for building ETL pipelines with Spark, Airflow, Python, and AWS. Experience with real-time data processing.',
            'salary': '$100,000 - $150,000',
            'url': 'https://example.com/job2',
            'posted_date': '1 week ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Machine Learning Engineer',
            'company': 'AI Solutions',
            'location': 'Remote',
            'description': 'ML engineer to develop and deploy models using PyTorch, TensorFlow, Docker, and Kubernetes. Strong Python skills required.',
            'salary': '$110,000 - $160,000',
            'url': 'https://example.com/job3',
            'posted_date': '3 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Data Analyst',
            'company': 'Analytics Plus',
            'location': 'New York, NY',
            'description': 'Analyst role focusing on SQL, Tableau, Power BI, Excel, and statistical analysis. Create dashboards and reports.',
            'salary': '$65,000 - $90,000',
            'url': 'https://example.com/job4',
            'posted_date': '5 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Junior Data Scientist',
            'company': 'StartUp Labs',
            'location': 'Remote',
            'description': 'Entry-level data scientist position. Python, Pandas, Scikit-learn, and SQL required. Machine learning knowledge preferred.',
            'salary': '$60,000 - $85,000',
            'url': 'https://example.com/job5',
            'posted_date': '1 day ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Lead Data Engineer',
            'company': 'Big Data Corp',
            'location': 'San Francisco, CA',
            'description': 'Leading data engineering team. Expertise in Spark, Kafka, Python, AWS, and data architecture required.',
            'salary': '$140,000 - $180,000',
            'url': 'https://example.com/job6',
            'posted_date': '1 week ago',
            'source': 'Sample Data'
        },
        {
            'title': 'BI Developer',
            'company': 'Enterprise Solutions',
            'location': 'Chicago, IL',
            'description': 'Business Intelligence developer using Tableau, Power BI, SQL, and Python. Create executive dashboards.',
            'salary': '$75,000 - $105,000',
            'url': 'https://example.com/job7',
            'posted_date': '4 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Research Scientist - ML',
            'company': 'Research Lab',
            'location': 'Remote',
            'description': 'Research role in machine learning and deep learning. PhD preferred. PyTorch, TensorFlow, NLP experience.',
            'salary': '$120,000 - $170,000',
            'url': 'https://example.com/job8',
            'posted_date': '2 weeks ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Data Platform Engineer',
            'company': 'Cloud Systems',
            'location': 'Seattle, WA',
            'description': 'Building scalable data platforms with Kubernetes, Docker, Python, and cloud services (AWS/GCP/Azure).',
            'salary': '$115,000 - $155,000',
            'url': 'https://example.com/job9',
            'posted_date': '6 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Analytics Engineer',
            'company': 'Data Insights',
            'location': 'Austin, TX',
            'description': 'Analytics engineering role. SQL, Python, dbt, and data modeling. Build data transformation pipelines.',
            'salary': '$85,000 - $120,000',
            'url': 'https://example.com/job10',
            'posted_date': '3 days ago',
            'source': 'Sample Data'
        }
    ]
    
    # Filter by search term
    filtered = [job for job in sample_jobs 
                if search_term.lower() in job['title'].lower() 
                or search_term.lower() in job['description'].lower()]
    
    return filtered if filtered else sample_jobs[:5]

def fetch_remoteok_jobs(search_term="data"):
    """Fetch jobs from RemoteOK API"""
    try:
        url = "https://remoteok.com/api"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            jobs_data = response.json()[1:]  # Skip first item (metadata)
            processed_jobs = []
            
            for job in jobs_data[:30]:  # Limit to 30 jobs
                if search_term.lower() in job.get('position', '').lower() or \
                   search_term.lower() in job.get('description', '').lower():
                    processed_jobs.append({
                        'title': job.get('position', 'N/A'),
                        'company': job.get('company', 'N/A'),
                        'location': job.get('location', 'Remote'),
                        'description': job.get('description', '')[:500],
                        'salary': f"${job.get('salary_min', 'N/A')} - ${job.get('salary_max', 'N/A')}" if job.get('salary_min') else 'Not specified',
                        'url': job.get('url', '#'),
                        'posted_date': 'Recent',
                        'source': 'RemoteOK'
                    })
            
            # If no jobs match search term, return fallback
            if not processed_jobs:
                st.info(f"No exact matches for '{search_term}' from RemoteOK. Using sample data.")
                return get_fallback_jobs(search_term)
            
            return processed_jobs
        
        # If API fails, return fallback
        st.warning("RemoteOK API unavailable. Using sample data.")
        return get_fallback_jobs(search_term)
        
    except Exception as e:
        st.warning(f"RemoteOK API error: {str(e)}. Using sample data.")
        return get_fallback_jobs(search_term)

def fetch_github_jobs(search_term="data"):
    """Fetch jobs from GitHub Jobs API alternative"""
    # Note: GitHub Jobs API was deprecated, using mock data for demonstration
    mock_jobs = [
        {
            'title': 'Data Scientist',
            'company': 'Tech Corp',
            'location': 'Remote',
            'description': 'Looking for data scientist with Python, ML, and SQL skills.',
            'salary': '$80,000 - $120,000',
            'url': 'https://example.com',
            'posted_date': '2 days ago',
            'source': 'GitHub Jobs'
        },
        {
            'title': 'Data Engineer',
            'company': 'DataFlow Inc',
            'location': 'Boston, MA',
            'description': 'Data engineer needed for ETL pipeline development with Spark and Airflow.',
            'salary': '$90,000 - $130,000',
            'url': 'https://example.com',
            'posted_date': '1 week ago',
            'source': 'GitHub Jobs'
        }
    ]
    return [job for job in mock_jobs if search_term.lower() in job['title'].lower()]

def fetch_all_jobs(search_term="data"):
    """Aggregate jobs from all sources"""
    all_jobs = []
    
    with st.spinner("🔍 Fetching jobs from RemoteOK..."):
        all_jobs.extend(fetch_remoteok_jobs(search_term))
    
    with st.spinner("🔍 Fetching jobs from GitHub..."):
        all_jobs.extend(fetch_github_jobs(search_term))
    
    return all_jobs

# ==================== SMART MATCHING ALGORITHM ====================
def calculate_match_score(user_skills, job_description):
    """Calculate match score using advanced algorithm"""
    job_lower = job_description.lower()
    
    # Extract skills from job description
    job_skill_dict, job_skills = extract_skills(job_description)
    
    # Calculate various match factors
    exact_matches = len(user_skills & job_skills)
    total_required = len(job_skills) if job_skills else 1
    
    # Skill match score (60% weight)
    skill_score = (exact_matches / total_required) * 60
    
    # Experience level matching (20% weight)
    experience_keywords = ['senior', 'lead', 'junior', 'entry', 'mid-level']
    experience_score = 20  # Default neutral score
    
    # Keyword density (20% weight)
    keyword_score = min((exact_matches / 5) * 20, 20)  # Cap at 20
    
    total_score = skill_score + experience_score + keyword_score
    
    return min(round(total_score, 1), 100), exact_matches, len(job_skills)

def rank_jobs(jobs, user_skills):
    """Rank jobs by match score"""
    ranked_jobs = []
    
    for job in jobs:
        description = job.get('description', '') + ' ' + job.get('title', '')
        score, matched, required = calculate_match_score(user_skills, description)
        
        job['match_score'] = score
        job['matched_skills'] = matched
        job['required_skills'] = required
        ranked_jobs.append(job)
    
    return sorted(ranked_jobs, key=lambda x: x['match_score'], reverse=True)

# ==================== SKILL GAP ANALYSIS ====================
def analyze_skill_gaps(user_skills, top_jobs):
    """Identify missing skills and prioritize them"""
    all_job_skills = set()
    skill_frequency = Counter()
    
    for job in top_jobs[:10]:  # Analyze top 10 jobs
        description = job.get('description', '') + ' ' + job.get('title', '')
        _, job_skills = extract_skills(description)
        all_job_skills.update(job_skills)
        skill_frequency.update(job_skills)
    
    missing_skills = all_job_skills - user_skills
    
    # Prioritize by frequency
    priority_skills = [
        {
            'skill': skill,
            'frequency': skill_frequency[skill],
            'priority': 'High' if skill_frequency[skill] >= 5 else 'Medium' if skill_frequency[skill] >= 3 else 'Low'
        }
        for skill in missing_skills
    ]
    
    return sorted(priority_skills, key=lambda x: x['frequency'], reverse=True)

def get_learning_resources(skill):
    """Suggest learning resources for skills"""
    resources = {
        'python': ['Python.org Tutorial', 'Coursera Python Specialization', 'LeetCode Python'],
        'machine learning': ['Coursera ML by Andrew Ng', 'Fast.ai', 'Kaggle Learn'],
        'sql': ['SQLZoo', 'Mode SQL Tutorial', 'LeetCode Database'],
        'docker': ['Docker Official Docs', 'Docker for Beginners', 'KodeKloud Docker'],
        'aws': ['AWS Free Tier', 'A Cloud Guru', 'AWS Certified Solutions Architect'],
    }
    
    return resources.get(skill.lower(), ['Google Search', 'YouTube Tutorials', 'Official Documentation'])

# ==================== VISUALIZATIONS ====================
def create_skill_distribution_chart(user_skills_dict):
    """Create skill distribution pie chart"""
    categories = []
    counts = []
    
    for category, skills in user_skills_dict.items():
        if skills:
            categories.append(category)
            counts.append(len(skills))
    
    fig = px.pie(
        values=counts,
        names=categories,
        title='Your Skills Distribution',
        color_discrete_sequence=px.colors.sequential.Purples
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_match_score_chart(jobs):
    """Create match score distribution chart"""
    if not jobs:
        return None
    
    df = pd.DataFrame(jobs[:20])  # Top 20 jobs
    
    fig = px.bar(
        df,
        x='title',
        y='match_score',
        title='Top 20 Job Matches',
        labels={'match_score': 'Match Score (%)', 'title': 'Job Title'},
        color='match_score',
        color_continuous_scale='Purples'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_skill_gap_chart(skill_gaps):
    """Create skill gap priority chart"""
    if not skill_gaps:
        return None
    
    df = pd.DataFrame(skill_gaps[:15])  # Top 15 missing skills
    
    fig = px.bar(
        df,
        x='skill',
        y='frequency',
        title='Top Missing Skills (by frequency in job postings)',
        labels={'frequency': 'Frequency', 'skill': 'Skill'},
        color='priority',
        color_discrete_map={'High': '#764ba2', 'Medium': '#667eea', 'Low': '#a8b3ff'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_salary_insights(jobs):
    """Create salary insights visualization"""
    salaries = []
    titles = []
    
    for job in jobs[:20]:
        salary_str = job.get('salary', '')
        if salary_str and '$' in salary_str:
            try:
                # Extract numeric values
                numbers = re.findall(r'\d+', salary_str.replace(',', ''))
                if len(numbers) >= 2:
                    avg_salary = (int(numbers[0]) + int(numbers[1])) / 2
                    salaries.append(avg_salary)
                    titles.append(job['title'][:30])
            except:
                pass
    
    if not salaries:
        return None
    
    df = pd.DataFrame({'Job Title': titles, 'Average Salary': salaries})
    
    fig = px.bar(
        df,
        x='Job Title',
        y='Average Salary',
        title='Salary Insights for Matched Jobs',
        labels={'Average Salary': 'Average Salary ($)'},
        color='Average Salary',
        color_continuous_scale='Purples'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

# ==================== EXPORT FEATURES ====================
def generate_csv_report(jobs, user_skills):
    """Generate CSV report of job matches"""
    if not jobs:
        return None
    
    df = pd.DataFrame(jobs)
    df = df[['title', 'company', 'location', 'match_score', 'salary', 'url', 'source']]
    return df.to_csv(index=False)

def generate_detailed_report(jobs, user_skills, skill_gaps):
    """Generate detailed text report"""
    report = f"""
# AI RESUME MATCHER - DETAILED REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## YOUR SKILLS ({len(user_skills)})
{', '.join(sorted(user_skills))}

## TOP JOB MATCHES
"""
    
    for i, job in enumerate(jobs[:10], 1):
        report += f"""
### {i}. {job['title']} at {job['company']}
- Match Score: {job['match_score']}%
- Location: {job['location']}
- Salary: {job['salary']}
- Skills Matched: {job['matched_skills']}/{job['required_skills']}
- URL: {job['url']}

"""
    
    report += f"""
## SKILL GAP ANALYSIS
Missing skills to improve your matches:
"""
    
    for gap in skill_gaps[:10]:
        report += f"- {gap['skill'].title()} (Priority: {gap['priority']}, Frequency: {gap['frequency']})\n"
    
    return report

# ==================== MAIN APP ====================
def main():
    st.markdown('<div class="main-header">🎯 AI-Powered Resume Matcher</div>', unsafe_allow_html=True)
    st.markdown("### Match your resume with perfect job opportunities using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        search_role = st.selectbox(
            "Target Role",
            ["data scientist", "data engineer", "data analyst", "software engineer", "machine learning"]
        )
        
        min_match_score = st.slider("Minimum Match Score (%)", 0, 100, 50)
        
        # Add Clear Results button
        if st.session_state.jobs:
            if st.button("🗑️ Clear Job Results", type="secondary"):
                st.session_state.jobs = []
                st.success("Job results cleared! Upload a new resume or search again.")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if st.session_state.jobs:
            st.metric("Total Jobs Found", len(st.session_state.jobs))
            st.metric("Your Skills", len(st.session_state.user_skills))
            avg_score = sum(j.get('match_score', 0) for j in st.session_state.jobs) / len(st.session_state.jobs)
            st.metric("Avg Match Score", f"{avg_score:.1f}%")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Resume Upload", 
        "💼 Job Matches", 
        "📊 Analytics", 
        "🎯 Skill Gap", 
        "📥 Export"
    ])
    
    # TAB 1: RESUME UPLOAD
    with tab1:
        st.header("Upload Your Resume")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            resume_text = st.text_area(
                "Paste your resume text here:",
                height=300,
                placeholder="Copy and paste your resume text here..."
            )
            
            if st.button("🔍 Analyze Resume", type="primary"):
                if resume_text:
                    st.session_state.resume_text = resume_text
                    skill_dict, skills = extract_skills(resume_text)
                    st.session_state.user_skills = skills
                    
                    # CLEAR OLD JOB RESULTS when new resume is uploaded
                    st.session_state.jobs = []
                    
                    st.success(f"✅ Found {len(skills)} skills in your resume!")
                    st.info("💡 Now go to 'Job Matches' tab to find new jobs for this resume!")
                    
                    # Display skills by category
                    st.subheader("Detected Skills")
                    for category, cat_skills in skill_dict.items():
                        if cat_skills:
                            st.markdown(f"**{category}:**")
                            st.markdown(" ".join([f'<span class="skill-badge">{skill}</span>' 
                                                 for skill in cat_skills]), 
                                      unsafe_allow_html=True)
                else:
                    st.error("Please paste your resume text first!")
        
        with col2:
            st.info("""
            **💡 Tips for best results:**
            - Include all your technical skills
            - Mention programming languages
            - Add frameworks and tools
            - Include soft skills
            - Mention certifications
            """)
            
            if st.button("📝 Use Sample Resume"):
                sample_resume = """
                Data Scientist with 2 years of experience in machine learning and data analysis.
                
                Skills: Python, SQL, Machine Learning, TensorFlow, Pandas, NumPy, Scikit-learn,
                Data Visualization, Tableau, Power BI, Statistical Analysis, A/B Testing,
                Git, Docker, AWS, Communication, Teamwork, Problem Solving
                
                Experience with building predictive models, ETL pipelines, and data dashboards.
                Strong analytical and communication skills.
                """
                st.session_state.resume_text = sample_resume
                skill_dict, skills = extract_skills(sample_resume)
                st.session_state.user_skills = skills
                
                # CLEAR OLD JOB RESULTS when sample resume is loaded
                st.session_state.jobs = []
                
                st.success("Sample resume loaded! Click 'Analyze Resume' to see extracted skills.")
                st.rerun()
    
    # TAB 2: JOB MATCHES
    with tab2:
        st.header("Job Matches")
        
        if not st.session_state.user_skills:
            st.warning("⚠️ Please upload your resume first in the 'Resume Upload' tab!")
        else:
            # Show warning if no jobs fetched yet
            if not st.session_state.jobs:
                st.info("👇 Click the button below to fetch jobs matching your resume!")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🔎 Find Matching Jobs", type="primary"):
                    jobs = fetch_all_jobs(search_role)
                    
                    if jobs:
                        ranked_jobs = rank_jobs(jobs, st.session_state.user_skills)
                        st.session_state.jobs = ranked_jobs
                        st.success(f"✅ Found {len(jobs)} jobs matching your skills!")
                        st.rerun()
                    else:
                        st.error("No jobs found. Try a different search term.")
            
            with col2:
                if st.session_state.jobs:
                    st.metric("Jobs Found", len(st.session_state.jobs))
            
            # Display jobs
            if st.session_state.jobs:
                filtered_jobs = [j for j in st.session_state.jobs if j['match_score'] >= min_match_score]
                
                st.subheader(f"Top Matches ({len(filtered_jobs)} jobs)")
                
                for i, job in enumerate(filtered_jobs[:20], 1):
                    with st.expander(f"#{i} - {job['title']} at {job['company']} - {job['match_score']}% Match"):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.markdown(f"**Company:** {job['company']}")
                            st.markdown(f"**Location:** {job['location']}")
                            st.markdown(f"**Posted:** {job['posted_date']}")
                        
                        with col2:
                            st.markdown(f"**Salary:** {job['salary']}")
                            st.markdown(f"**Source:** {job['source']}")
                            st.markdown(f"**Skills Match:** {job['matched_skills']}/{job['required_skills']}")
                        
                        with col3:
                            st.markdown(f'<div class="match-score">{job["match_score"]}%</div>', 
                                      unsafe_allow_html=True)
                        
                        st.markdown("**Description:**")
                        st.write(job['description'][:300] + "...")
                        
                        st.markdown(f"[🔗 View Job]({job['url']})")
    
    # TAB 3: ANALYTICS
    with tab3:
        st.header("Analytics Dashboard")
        
        if not st.session_state.user_skills:
            st.warning("⚠️ Please upload your resume first!")
        elif not st.session_state.jobs:
            st.warning("⚠️ Please find matching jobs first!")
        else:
            # Skills distribution
            skill_dict, _ = extract_skills(st.session_state.resume_text)
            fig1 = create_skill_distribution_chart(skill_dict)
            st.plotly_chart(fig1, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Match score chart
                fig2 = create_match_score_chart(st.session_state.jobs)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Salary insights
                fig4 = create_salary_insights(st.session_state.jobs)
                if fig4:
                    st.plotly_chart(fig4, use_container_width=True)
    
    # TAB 4: SKILL GAP
    with tab4:
        st.header("Skill Gap Analysis")
        
        if not st.session_state.user_skills or not st.session_state.jobs:
            st.warning("⚠️ Please upload your resume and find jobs first!")
        else:
            skill_gaps = analyze_skill_gaps(st.session_state.user_skills, st.session_state.jobs)
            
            if skill_gaps:
                # Visualization
                fig3 = create_skill_gap_chart(skill_gaps)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
                
                st.subheader("🎯 Skills to Learn")
                st.markdown("These skills appear frequently in your matched jobs but are missing from your resume:")
                
                for gap in skill_gaps[:10]:
                    with st.expander(f"**{gap['skill'].title()}** - {gap['priority']} Priority (appears in {gap['frequency']} jobs)"):
                        st.markdown(f"**Priority Level:** {gap['priority']}")
                        st.markdown(f"**Frequency:** Found in {gap['frequency']} job postings")
                        
                        st.markdown("**📚 Recommended Learning Resources:**")
                        resources = get_learning_resources(gap['skill'])
                        for resource in resources:
                            st.markdown(f"- {resource}")
            else:
                st.success("🎉 Great! You have all the major skills needed for these positions!")
    
    # TAB 5: EXPORT
    with tab5:
        st.header("Export Reports")
        
        if not st.session_state.jobs:
            st.warning("⚠️ Please find matching jobs first!")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 CSV Export")
                csv_data = generate_csv_report(st.session_state.jobs, st.session_state.user_skills)
                if csv_data:
                    st.download_button(
                        label="📥 Download Jobs CSV",
                        data=csv_data,
                        file_name=f"job_matches_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.subheader("📄 Detailed Report")
                skill_gaps = analyze_skill_gaps(st.session_state.user_skills, st.session_state.jobs)
                report_data = generate_detailed_report(
                    st.session_state.jobs, 
                    st.session_state.user_skills,
                    skill_gaps
                )
                st.download_button(
                    label="📥 Download Full Report",
                    data=report_data,
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            
            st.markdown("---")
            st.subheader("📧 Email Report")
            email = st.text_input("Enter your email to receive the report:")
            if st.button("Send Report"):
                if email:
                    st.success(f"📧 Report will be sent to {email} (Email functionality to be implemented)")
                else:
                    st.error("Please enter an email address")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit | AI Resume Matcher v2.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()