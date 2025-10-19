import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import requests
from collections import Counter
import json

# ML/NLP imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False
    import numpy as np

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    """Download NLTK data with caching"""
    if NLTK_AVAILABLE:
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except:
                pass

download_nltk_data()

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

# ==================== ML/NLP HELPER FUNCTIONS ====================
def calculate_tfidf_similarity(resume_text, job_description):
    """Calculate semantic similarity using TF-IDF vectorization"""
    if not SKLEARN_AVAILABLE:
        return 0
    try:
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)
    except:
        return 0

def preprocess_text(text):
    """NLP preprocessing: tokenization, lowercasing, stopword removal"""
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            return filtered_tokens
        except:
            pass
    
    # Fallback
    common_stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 
                       'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
    words = text.lower().split()
    filtered = [w for w in words if w.isalnum() and w not in common_stopwords]
    return filtered

def extract_keywords_tfidf(text, top_n=10):
    """Extract most important keywords using TF-IDF"""
    if not SKLEARN_AVAILABLE:
        return []
    try:
        vectorizer = TfidfVectorizer(
            max_features=top_n,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        keywords = [(feature_names[i], scores[i]) for i in scores.argsort()[-top_n:][::-1]]
        return keywords
    except:
        return []

def extract_topics_lda(texts, n_topics=3, n_words=5):
    """Extract topics using LDA topic modeling"""
    if not SKLEARN_AVAILABLE or len(texts) < 5:
        return []
    try:
        vectorizer = CountVectorizer(max_features=100, stop_words='english', min_df=2)
        doc_term_matrix = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        lda.fit(doc_term_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({'topic_num': topic_idx + 1, 'keywords': top_words})
        return topics
    except:
        return []

# ==================== SKILL EXTRACTION ====================
def extract_skills(text):
    """Extract skills from text using comprehensive skill database"""
    text_lower = text.lower()
    
    skill_categories = {
        'Programming Languages': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'ruby', 
            'go', 'rust', 'swift', 'kotlin', 'php', 'scala', 'r programming', 'r language', 
            'matlab', 'hdl'
        ],
        'Electronics & Hardware': [
            'digital circuit design', 'analog electronics', 'embedded systems', 
            'vlsi', 'microprocessors', 'microcontrollers', '8051', 'arm', 'avr', 'pic',
            'fpga', 'verilog', 'vhdl', 'pcb design', 'kicad', 'altium', 'eagle',
            'power electronics', 'signal conditioning', 'hardware debugging', 
            'circuit design', 'schematic design', 'embedded c', 'rtos',
            'iot', 'raspberry pi', 'arduino', 'nodemcu', 'pcb layout', 'sensors'
        ],
        'Data Science & ML': [
            'machine learning', 'deep learning', 'neural networks', 'nlp', 'cnn',
            'computer vision', 'tensorflow', 'pytorch', 'scikit-learn', 'keras',
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'transfer learning'
        ],
        'Data Engineering': [
            'sql', 'nosql', 'postgresql', 'mysql', 'mongodb', 'redis',
            'apache spark', 'hadoop', 'kafka', 'airflow', 'etl', 'data pipeline',
            'snowflake', 'bigquery', 'redshift'
        ],
        'Cloud & DevOps': [
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'ci/cd', 
            'jenkins', 'terraform', 'ansible', 'linux', 'bash', 'git version control', 
            'github', 'gitlab', 'devops'
        ],
        'Web Development': [
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'fastapi', 'rest api', 'graphql', 'html', 'css', 'bootstrap', 'tailwind'
        ],
        'Data Analysis': [
            'excel', 'tableau', 'power bi', 'looker', 'data visualization',
            'statistics', 'a/b testing', 'hypothesis testing', 'regression', 'ggplot2',
            'plotly', 'statistical analysis'
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
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills[category].append(skill)
                all_skills.add(skill)
    
    return found_skills, all_skills

# ==================== JOB DATA ====================
def get_fallback_jobs(search_role="data scientist"):
    """Get sample jobs filtered by role type"""
    
    # Complete job database
    all_jobs = [
        # DATA SCIENTIST ROLES
        {'title': 'Data Scientist', 'company': 'Tech Innovations Inc', 'location': 'Remote',
         'description': 'Seeking data scientist with Python, machine learning, SQL, and TensorFlow. Work on predictive models and data pipelines.',
         'salary': '$85,000 - $125,000', 'url': 'https://example.com/ds1', 'posted_date': '2 days ago',
         'source': 'Sample', 'role_type': 'data scientist'},
        
        {'title': 'Senior Data Scientist', 'company': 'FinTech Solutions', 'location': 'New York, NY',
         'description': 'Lead data science projects using Python, R, machine learning. Financial data, risk modeling, PyTorch, TensorFlow.',
         'salary': '$120,000 - $160,000', 'url': 'https://example.com/ds2', 'posted_date': '1 week ago',
         'source': 'Sample', 'role_type': 'data scientist'},
        
        {'title': 'Junior Data Scientist', 'company': 'E-commerce Analytics', 'location': 'Remote',
         'description': 'Entry-level data scientist. Python, SQL, Pandas, Scikit-learn. Customer segmentation, recommendation systems.',
         'salary': '$65,000 - $85,000', 'url': 'https://example.com/ds3', 'posted_date': '3 days ago',
         'source': 'Sample', 'role_type': 'data scientist'},
        
        {'title': 'Research Scientist - AI', 'company': 'Research Lab', 'location': 'Boston, MA',
         'description': 'Research in machine learning and deep learning. PyTorch, TensorFlow, publications. Cutting-edge AI research.',
         'salary': '$90,000 - $140,000', 'url': 'https://example.com/ds4', 'posted_date': '2 weeks ago',
         'source': 'Sample', 'role_type': 'data scientist'},
        
        # DATA ANALYST ROLES
        {'title': 'Data Analyst', 'company': 'Retail Analytics', 'location': 'Chicago, IL',
         'description': 'SQL, Excel, Tableau, Power BI. Dashboards, statistical analysis, business insights.',
         'salary': '$60,000 - $85,000', 'url': 'https://example.com/da1', 'posted_date': '2 days ago',
         'source': 'Sample', 'role_type': 'data analyst'},
        
        {'title': 'Business Intelligence Analyst', 'company': 'Healthcare Data', 'location': 'Remote',
         'description': 'BI analyst. SQL, Tableau, Excel. Executive dashboards, automate reports, healthcare metrics.',
         'salary': '$70,000 - $95,000', 'url': 'https://example.com/da2', 'posted_date': '1 week ago',
         'source': 'Sample', 'role_type': 'data analyst'},
        
        {'title': 'Senior Data Analyst', 'company': 'Marketing Analytics', 'location': 'Austin, TX',
         'description': 'Lead analyst. SQL, Python, Tableau, Power BI. Marketing attribution, customer analytics.',
         'salary': '$85,000 - $115,000', 'url': 'https://example.com/da3', 'posted_date': '3 days ago',
         'source': 'Sample', 'role_type': 'data analyst'},
        
        # DATA ENGINEER ROLES
        {'title': 'Data Engineer', 'company': 'Cloud Data Systems', 'location': 'Boston, MA',
         'description': 'ETL pipelines with Spark, Airflow, Python, AWS. Real-time data, Kafka, data warehousing.',
         'salary': '$95,000 - $140,000', 'url': 'https://example.com/de1', 'posted_date': '5 days ago',
         'source': 'Sample', 'role_type': 'data engineer'},
        
        {'title': 'Senior Data Engineer', 'company': 'Big Data Corp', 'location': 'San Francisco, CA',
         'description': 'Lead engineering. Spark, Hadoop, Kafka, Python, AWS. Scalable pipelines, millions of records.',
         'salary': '$130,000 - $175,000', 'url': 'https://example.com/de2', 'posted_date': '1 week ago',
         'source': 'Sample', 'role_type': 'data engineer'},
        
        {'title': 'Analytics Engineer', 'company': 'Data Insights', 'location': 'Remote',
         'description': 'SQL, Python, dbt, data modeling. Transformation pipelines, Snowflake, BigQuery.',
         'salary': '$85,000 - $120,000', 'url': 'https://example.com/de3', 'posted_date': '4 days ago',
         'source': 'Sample', 'role_type': 'data engineer'},
        
        # MACHINE LEARNING ROLES
        {'title': 'Machine Learning Engineer', 'company': 'AI Innovations', 'location': 'Remote',
         'description': 'Develop and deploy ML models. PyTorch, TensorFlow, Docker, Kubernetes. MLOps, cloud deployment.',
         'salary': '$110,000 - $160,000', 'url': 'https://example.com/ml1', 'posted_date': '4 days ago',
         'source': 'Sample', 'role_type': 'machine learning'},
        
        {'title': 'ML Operations Engineer', 'company': 'Model Deployment Inc', 'location': 'Seattle, WA',
         'description': 'MLOps. Deploy and monitor ML models. Docker, Kubernetes, CI/CD, Python, MLflow.',
         'salary': '$100,000 - $145,000', 'url': 'https://example.com/ml2', 'posted_date': '6 days ago',
         'source': 'Sample', 'role_type': 'machine learning'},
        
        {'title': 'NLP Engineer', 'company': 'Language AI', 'location': 'Remote',
         'description': 'NLP engineer. Python, transformers, BERT, spaCy, NLTK. Text classification, sentiment analysis.',
         'salary': '$105,000 - $150,000', 'url': 'https://example.com/ml3', 'posted_date': '5 days ago',
         'source': 'Sample', 'role_type': 'machine learning'},
        
        {'title': 'Computer Vision Engineer', 'company': 'Robotics AI', 'location': 'San Jose, CA',
         'description': 'Computer vision. Python, OpenCV, TensorFlow, PyTorch. Object detection, image segmentation, CNN.',
         'salary': '$115,000 - $165,000', 'url': 'https://example.com/ml4', 'posted_date': '5 days ago',
         'source': 'Sample', 'role_type': 'machine learning'},
        
        # SOFTWARE ENGINEER ROLES
        {'title': 'Software Engineer - Backend', 'company': 'Tech Startup', 'location': 'Remote',
         'description': 'Backend engineer. Python, Django, PostgreSQL, Redis, Docker. Scalable microservices architecture.',
         'salary': '$95,000 - $135,000', 'url': 'https://example.com/se1', 'posted_date': '3 days ago',
         'source': 'Sample', 'role_type': 'software engineer'},
        
        {'title': 'Full Stack Developer', 'company': 'Web Solutions', 'location': 'Austin, TX',
         'description': 'Full stack. React, Node.js, Python, MongoDB, REST APIs. Modern web applications.',
         'salary': '$85,000 - $125,000', 'url': 'https://example.com/se2', 'posted_date': '5 days ago',
         'source': 'Sample', 'role_type': 'software engineer'},
        
        {'title': 'Backend Developer', 'company': 'Cloud Services', 'location': 'Seattle, WA',
         'description': 'Backend for cloud services. Python, FastAPI, PostgreSQL, Docker, Kubernetes, AWS.',
         'salary': '$90,000 - $130,000', 'url': 'https://example.com/se3', 'posted_date': '4 days ago',
         'source': 'Sample', 'role_type': 'software engineer'},
        
        {'title': 'Software Engineer - Data Platform', 'company': 'Platform Engineering', 'location': 'San Francisco, CA',
         'description': 'Build data platforms. Python, Java, Spark, Kafka, cloud. Scalable data processing.',
         'salary': '$110,000 - $155,000', 'url': 'https://example.com/se4', 'posted_date': '1 week ago',
         'source': 'Sample', 'role_type': 'software engineer'},
        
        # IoT/EMBEDDED ROLES
        {'title': 'IoT Data Engineer', 'company': 'Smart Devices Inc', 'location': 'Boston, MA',
         'description': 'IoT data engineer. Sensor data, embedded systems, Python, Arduino, Kafka, time-series databases.',
         'salary': '$95,000 - $135,000', 'url': 'https://example.com/iot1', 'posted_date': '6 days ago',
         'source': 'Sample', 'role_type': 'data engineer'},
        
        {'title': 'Embedded Systems Engineer', 'company': 'Hardware AI', 'location': 'San Jose, CA',
         'description': 'Embedded engineer. C, Python, microcontrollers, FPGA, signal processing. IoT devices with ML.',
         'salary': '$100,000 - $145,000', 'url': 'https://example.com/iot2', 'posted_date': '4 days ago',
         'source': 'Sample', 'role_type': 'software engineer'},
        
        {'title': 'FPGA Engineer', 'company': 'Hardware Design', 'location': 'Austin, TX',
         'description': 'FPGA development. Verilog/VHDL, digital circuit design, embedded systems, signal processing.',
         'salary': '$95,000 - $140,000', 'url': 'https://example.com/fpga1', 'posted_date': '1 week ago',
         'source': 'Sample', 'role_type': 'software engineer'},
    ]
    
    # Filter by role type
    filtered = []
    for job in all_jobs:
        # Match by role_type or title/description
        if (search_role.lower() in job.get('role_type', '').lower() or
            search_role.lower() in job['title'].lower() or
            search_role.lower() in job['description'].lower()):
            filtered.append(job)
    
    return filtered if filtered else all_jobs[:10]

def fetch_remoteok_jobs(search_term="data scientist"):
    """Fetch jobs from RemoteOK API"""
    try:
        url = "https://remoteok.com/api"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            jobs_data = response.json()[1:]
            processed_jobs = []
            
            for job in jobs_data[:30]:
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
                        'source': 'RemoteOK',
                        'role_type': search_term.lower()
                    })
            
            return processed_jobs if processed_jobs else []
        return []
    except:
        return []

def fetch_all_jobs(search_role="data scientist"):
    """Fetch jobs filtered by selected role"""
    all_jobs = []
    
    # Fetch from RemoteOK
    with st.spinner(f"🔍 Searching for '{search_role}' jobs..."):
        remoteok_jobs = fetch_remoteok_jobs(search_role)
        if remoteok_jobs:
            all_jobs.extend(remoteok_jobs)
            st.info(f"Found {len(remoteok_jobs)} jobs from RemoteOK")
    
    # Get sample jobs filtered by role
    sample_jobs = get_fallback_jobs(search_role)
    all_jobs.extend(sample_jobs)
    
    # Remove duplicates
    seen = set()
    unique_jobs = []
    
    for job in all_jobs:
        job_key = (job['title'].lower().strip(), job['company'].lower().strip())
        if job_key not in seen:
            seen.add(job_key)
            unique_jobs.append(job)
    
    return unique_jobs

# ==================== MATCHING ALGORITHM ====================
def calculate_match_score(user_skills, job_description, resume_text=""):
    """ML-Enhanced matching algorithm"""
    job_skill_dict, job_skills = extract_skills(job_description)
    
    # 1. Skill Match (40%)
    exact_matches = len(user_skills & job_skills)
    total_required = len(job_skills) if job_skills else 1
    skill_score = (exact_matches / total_required) * 40
    
    # 2. TF-IDF Semantic Similarity (30%)
    if resume_text and len(resume_text) > 50 and SKLEARN_AVAILABLE:
        tfidf_similarity = calculate_tfidf_similarity(resume_text, job_description)
        semantic_score = (tfidf_similarity / 100) * 30
    else:
        semantic_score = 0
    
    # 3. Keyword Overlap (20%)
    if resume_text:
        resume_keywords = set(preprocess_text(resume_text))
        job_keywords = set(preprocess_text(job_description))
        if resume_keywords and job_keywords:
            keyword_overlap = len(resume_keywords & job_keywords) / len(job_keywords)
            keyword_score = keyword_overlap * 20
        else:
            keyword_score = 0
    else:
        keyword_score = 0
    
    # 4. Experience (10%)
    experience_score = 10
    
    total_score = skill_score + semantic_score + keyword_score + experience_score
    return min(round(total_score, 1), 100), exact_matches, len(job_skills)

def rank_jobs(jobs, user_skills, resume_text=""):
    """Rank jobs by ML-enhanced match score"""
    ranked_jobs = []
    
    for job in jobs:
        description = job.get('description', '') + ' ' + job.get('title', '')
        score, matched, required = calculate_match_score(user_skills, description, resume_text)
        
        job['match_score'] = score
        job['matched_skills'] = matched
        job['required_skills'] = required
        ranked_jobs.append(job)
    
    return sorted(ranked_jobs, key=lambda x: x['match_score'], reverse=True)

# ==================== SKILL GAP ANALYSIS ====================
def analyze_skill_gaps(user_skills, top_jobs):
    """Identify missing skills"""
    all_job_skills = set()
    skill_frequency = Counter()
    
    for job in top_jobs[:10]:
        description = job.get('description', '') + ' ' + job.get('title', '')
        _, job_skills = extract_skills(description)
        all_job_skills.update(job_skills)
        skill_frequency.update(job_skills)
    
    missing_skills = all_job_skills - user_skills
    
    priority_skills = [
        {'skill': skill, 'frequency': skill_frequency[skill],
         'priority': 'High' if skill_frequency[skill] >= 5 else 'Medium' if skill_frequency[skill] >= 3 else 'Low'}
        for skill in missing_skills
    ]
    
    return sorted(priority_skills, key=lambda x: x['frequency'], reverse=True)

def get_learning_resources(skill):
    """Learning resources"""
    resources = {
        'python': ['Python.org', 'Coursera Python', 'LeetCode'],
        'machine learning': ['Coursera ML', 'Fast.ai', 'Kaggle'],
        'sql': ['SQLZoo', 'Mode SQL', 'LeetCode DB'],
    }
    return resources.get(skill.lower(), ['Google', 'YouTube', 'Docs'])

# ==================== VISUALIZATIONS ====================
def create_skill_distribution_chart(user_skills_dict):
    """Skills pie chart"""
    categories = [cat for cat, skills in user_skills_dict.items() if skills]
    counts = [len(skills) for cat, skills in user_skills_dict.items() if skills]
    
    fig = px.pie(values=counts, names=categories, title='Your Skills Distribution',
                 color_discrete_sequence=px.colors.sequential.Purples)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_match_score_chart(jobs):
    """Match score bar chart"""
    if not jobs:
        return None
    df = pd.DataFrame(jobs[:20])
    fig = px.bar(df, x='title', y='match_score', title='Top 20 Job Matches',
                 labels={'match_score': 'Match Score (%)', 'title': 'Job Title'},
                 color='match_score', color_continuous_scale='Purples')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_skill_gap_chart(skill_gaps):
    """Skill gap chart"""
    if not skill_gaps:
        return None
    df = pd.DataFrame(skill_gaps[:15])
    fig = px.bar(df, x='skill', y='frequency', title='Top Missing Skills',
                 labels={'frequency': 'Frequency', 'skill': 'Skill'}, color='priority',
                 color_discrete_map={'High': '#764ba2', 'Medium': '#667eea', 'Low': '#a8b3ff'})
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_salary_insights(jobs):
    """Salary chart"""
    salaries, titles = [], []
    for job in jobs[:20]:
        salary_str = job.get('salary', '')
        if salary_str and '$' in salary_str:
            try:
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
    fig = px.bar(df, x='Job Title', y='Average Salary', title='Salary Insights',
                 color='Average Salary', color_continuous_scale='Purples')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

# ==================== EXPORT ====================
def generate_csv_report(jobs):
    """CSV export"""
    if not jobs:
        return None
    df = pd.DataFrame(jobs)
    df = df[['title', 'company', 'location', 'match_score', 'salary', 'url', 'source']]
    return df.to_csv(index=False)

def generate_detailed_report(jobs, user_skills, skill_gaps):
    """Text report"""
    report = f"""# AI RESUME MATCHER - REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## YOUR SKILLS ({len(user_skills)})
{', '.join(sorted(user_skills))}

## TOP JOB MATCHES
"""
    for i, job in enumerate(jobs[:10], 1):
        report += f"""
### {i}. {job['title']} at {job['company']}
- Match: {job['match_score']}%
- Location: {job['location']}
- Salary: {job['salary']}
- URL: {job['url']}
"""
    
    report += f"\n## SKILL GAPS\n"
    for gap in skill_gaps[:10]:
        report += f"- {gap['skill'].title()} (Priority: {gap['priority']}, Freq: {gap['frequency']})\n"
    
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
        
        if st.session_state.jobs:
            if st.button("🗑️ Clear Job Results"):
                st.session_state.jobs = []
                st.success("Cleared!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if st.session_state.jobs:
            st.metric("Total Jobs", len(st.session_state.jobs))
            st.metric("Your Skills", len(st.session_state.user_skills))
            avg_score = sum(j.get('match_score', 0) for j in st.session_state.jobs) / len(st.session_state.jobs)
            st.metric("Avg Match", f"{avg_score:.1f}%")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Resume Upload", "💼 Job Matches", "📊 Analytics", 
        "🎯 Skill Gap", "📥 Export", "🤖 ML Insights"
    ])
    
    # TAB 1: RESUME UPLOAD
    with tab1:
        st.header("Upload Your Resume")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            resume_text = st.text_area(
                "Paste your resume text here:",
                value=st.session_state.resume_text,
                height=300,
                placeholder="Copy and paste your resume text here..."
            )
            
            if st.button("🔍 Analyze Resume", type="primary"):
                if resume_text:
                    st.session_state.resume_text = resume_text
                    skill_dict, skills = extract_skills(resume_text)
                    st.session_state.user_skills = skills
                    st.session_state.jobs = []
                    
                    st.success(f"✅ Found {len(skills)} skills!")
                    st.info("💡 Go to 'Job Matches' tab to find jobs!")
                    
                    st.subheader("Detected Skills")
                    for category, cat_skills in skill_dict.items():
                        if cat_skills:
                            st.markdown(f"**{category}:**")
                            st.markdown(" ".join([f'<span class="skill-badge">{skill}</span>' 
                                                 for skill in cat_skills]), unsafe_allow_html=True)
                else:
                    st.error("Please paste your resume!")
        
        with col2:
            st.info("""**💡 Tips:**
            - Include technical skills
            - Mention programming languages
            - Add frameworks/tools
            - Include soft skills
            """)
            
            if st.button("📝 Use Sample Resume"):
                sample_resume = """Data Scientist with 2 years experience in machine learning.
                
Skills: Python, SQL, Machine Learning, TensorFlow, Pandas, NumPy, Scikit-learn,
Data Visualization, Tableau, Power BI, Statistical Analysis, A/B Testing,
Git, Docker, AWS, Communication, Teamwork, Problem Solving

Experience with predictive models, ETL pipelines, and data dashboards."""
                
                st.session_state.resume_text = sample_resume
                skill_dict, skills = extract_skills(sample_resume)
                st.session_state.user_skills = skills
                st.session_state.jobs = []
                
                st.success(f"✅ Sample loaded! Found {len(skills)} skills.")
                
                st.subheader("Sample Resume Skills")
                for category, cat_skills in skill_dict.items():
                    if cat_skills:
                        st.markdown(f"**{category}:**")
                        st.markdown(" ".join([f'<span class="skill-badge">{skill}</span>' 
                                             for skill in cat_skills]), unsafe_allow_html=True)
    
    # TAB 2: JOB MATCHES
    with tab2:
        st.header("Job Matches")
        
        if not st.session_state.user_skills:
            st.warning("⚠️ Upload your resume first!")
        else:
            if not st.session_state.jobs:
                st.info(f"👇 Click to find '{search_role}' jobs matching your resume!")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🔎 Find Matching Jobs", type="primary"):
                    jobs = fetch_all_jobs(search_role)
                    
                    if jobs:
                        ranked_jobs = rank_jobs(jobs, st.session_state.user_skills, st.session_state.resume_text)
                        st.session_state.jobs = ranked_jobs
                        st.success(f"✅ Found {len(jobs)} '{search_role}' jobs!")
                        if SKLEARN_AVAILABLE:
                            st.info("💡 Using ML-enhanced matching!")
                        st.rerun()
                    else:
                        st.error("No jobs found. Try different role.")
            
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
        
        if not st.session_state.user_skills or not st.session_state.jobs:
            st.warning("⚠️ Upload resume and find jobs first!")
        else:
            skill_dict, _ = extract_skills(st.session_state.resume_text)
            fig1 = create_skill_distribution_chart(skill_dict)
            st.plotly_chart(fig1, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig2 = create_match_score_chart(st.session_state.jobs)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig4 = create_salary_insights(st.session_state.jobs)
                if fig4:
                    st.plotly_chart(fig4, use_container_width=True)
    
    # TAB 4: SKILL GAP
    with tab4:
        st.header("Skill Gap Analysis")
        
        if not st.session_state.user_skills or not st.session_state.jobs:
            st.warning("⚠️ Upload resume and find jobs first!")
        else:
            skill_gaps = analyze_skill_gaps(st.session_state.user_skills, st.session_state.jobs)
            
            if skill_gaps:
                fig3 = create_skill_gap_chart(skill_gaps)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
                
                st.subheader("🎯 Skills to Learn")
                
                for gap in skill_gaps[:10]:
                    with st.expander(f"**{gap['skill'].title()}** - {gap['priority']} Priority (appears in {gap['frequency']} jobs)"):
                        st.markdown(f"**Priority:** {gap['priority']}")
                        st.markdown(f"**Frequency:** {gap['frequency']} jobs")
                        st.markdown("**📚 Resources:**")
                        for resource in get_learning_resources(gap['skill']):
                            st.markdown(f"- {resource}")
            else:
                st.success("🎉 You have all major skills!")
    
    # TAB 5: EXPORT
    with tab5:
        st.header("Export Reports")
        
        if not st.session_state.jobs:
            st.warning("⚠️ Find jobs first!")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 CSV Export")
                csv_data = generate_csv_report(st.session_state.jobs)
                if csv_data:
                    st.download_button(
                        "📥 Download CSV",
                        data=csv_data,
                        file_name=f"jobs_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.subheader("📄 Text Report")
                skill_gaps = analyze_skill_gaps(st.session_state.user_skills, st.session_state.jobs)
                report_data = generate_detailed_report(st.session_state.jobs, st.session_state.user_skills, skill_gaps)
                st.download_button(
                    "📥 Download Report",
                    data=report_data,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
    
    # TAB 6: ML INSIGHTS
    with tab6:
        st.header("🤖 Machine Learning Insights")
        
        with st.expander("🔍 Debug Info"):
            st.write(f"Resume: {len(st.session_state.resume_text)} chars")
            st.write(f"Jobs: {len(st.session_state.jobs)}")
            st.write(f"Skills: {len(st.session_state.user_skills)}")
            st.write(f"Sklearn: {SKLEARN_AVAILABLE}")
            st.write(f"NLTK: {NLTK_AVAILABLE}")
        
        if not st.session_state.resume_text:
            st.warning("⚠️ Upload resume first!")
        else:
            # Text Stats
            st.subheader("📝 Resume Analysis")
            tokens = preprocess_text(st.session_state.resume_text)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Words", len(st.session_state.resume_text.split()))
            with col2:
                st.metric("Unique Words", len(set(tokens)))
            with col3:
                st.metric("Lexical Diversity", round(len(set(tokens))/len(tokens) if tokens else 0, 2))
            with col4:
                st.metric("Avg Word Len", round(np.mean([len(w) for w in tokens]) if tokens else 0, 2))
            
            # Keywords
            if SKLEARN_AVAILABLE:
                st.markdown("---")
                st.subheader("🔑 Top Keywords (TF-IDF)")
                keywords = extract_keywords_tfidf(st.session_state.resume_text, 15)
                
                if keywords:
                    kw_df = pd.DataFrame(keywords, columns=['Keyword', 'Importance'])
                    fig = px.bar(kw_df, x='Importance', y='Keyword', orientation='h',
                                title='Most Important Keywords', color='Importance',
                                color_continuous_scale='Purples')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Similarity
            if st.session_state.jobs and SKLEARN_AVAILABLE:
                st.markdown("---")
                st.subheader("🎯 Semantic Similarity")
                
                similarity_scores = []
                for job in st.session_state.jobs[:10]:
                    job_text = job.get('description', '') + ' ' + job.get('title', '')
                    sim = calculate_tfidf_similarity(st.session_state.resume_text, job_text)
                    similarity_scores.append({'Job': job['title'][:40], 'Similarity': sim})
                
                if similarity_scores:
                    sim_df = pd.DataFrame(similarity_scores)
                    fig = px.bar(sim_df, x='Job', y='Similarity', 
                                title='Resume-Job Semantic Similarity',
                                color='Similarity', color_continuous_scale='Purples')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Topics
            if st.session_state.jobs and len(st.session_state.jobs) >= 5 and SKLEARN_AVAILABLE:
                st.markdown("---")
                st.subheader("📚 Job Market Topics (LDA)")
                
                job_texts = [j.get('description', '') + ' ' + j.get('title', '') 
                           for j in st.session_state.jobs[:50]]
                job_texts = [t for t in job_texts if len(t) > 50]
                
                if len(job_texts) >= 5:
                    topics = extract_topics_lda(job_texts, 3, 5)
                    if topics:
                        for topic in topics:
                            with st.expander(f"📌 Topic {topic['topic_num']}: {', '.join(topic['keywords'][:3])}"):
                                for word in topic['keywords']:
                                    st.markdown(f"- {word}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit | AI Resume Matcher v2.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()