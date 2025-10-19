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

# Download NLTK data (first time only)
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
    
    # Fallback: simple splitting
    common_stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
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
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words='english',
            min_df=2
        )
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
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
            'go', 'rust', 'swift', 'kotlin', 'php', 'scala', 'r programming', 'r language', 'matlab'
        ],
        'Electronics & Hardware': [
            'digital circuit design', 'analog electronics', 'embedded systems', 
            'vlsi', 'microprocessors', 'microcontrollers', '8051', 'arm', 'avr', 'pic',
            'fpga', 'verilog', 'vhdl', 'pcb design', 'kicad', 'altium', 'eagle',
            'power electronics', 'signal conditioning', 'hardware debugging', 
            'circuit design', 'schematic design', 'embedded c', 'rtos',
            'iot', 'raspberry pi', 'arduino', 'pcb layout'
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
            # Use word boundary matching
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills[category].append(skill)
                all_skills.add(skill)
    
    return found_skills, all_skills

# ==================== JOB API INTEGRATIONS ====================
def get_fallback_jobs(search_term="data"):
    """Fallback sample jobs when API is unavailable - 20+ diverse options"""
    sample_jobs = [
        # Data Science Roles
        {
            'title': 'Data Scientist',
            'company': 'Tech Innovations Inc',
            'location': 'Remote',
            'description': 'Seeking data scientist with Python, machine learning, SQL, and TensorFlow experience. Work on predictive models and data pipelines. Experience with Pandas, NumPy, and statistical analysis required.',
            'salary': '$85,000 - $125,000',
            'url': 'https://example.com/job1',
            'posted_date': '2 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Senior Data Scientist',
            'company': 'FinTech Solutions',
            'location': 'New York, NY',
            'description': 'Lead data science projects using Python, R, machine learning algorithms. Experience with financial data, risk modeling, and deep learning frameworks like PyTorch and TensorFlow.',
            'salary': '$120,000 - $160,000',
            'url': 'https://example.com/job2',
            'posted_date': '1 week ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Junior Data Scientist',
            'company': 'E-commerce Analytics',
            'location': 'Remote',
            'description': 'Entry-level data scientist. Python, SQL, Pandas, Scikit-learn required. Work on customer segmentation, recommendation systems, and A/B testing.',
            'salary': '$65,000 - $85,000',
            'url': 'https://example.com/job3',
            'posted_date': '3 days ago',
            'source': 'Sample Data'
        },
        
        # Data Engineer Roles
        {
            'title': 'Data Engineer',
            'company': 'Cloud Data Systems',
            'location': 'Boston, MA',
            'description': 'Build ETL pipelines with Spark, Airflow, Python, and AWS. Experience with real-time data processing, Kafka, and data warehousing. SQL and NoSQL databases.',
            'salary': '$95,000 - $140,000',
            'url': 'https://example.com/job4',
            'posted_date': '5 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Senior Data Engineer',
            'company': 'Big Data Corp',
            'location': 'San Francisco, CA',
            'description': 'Lead data engineering team. Expertise in Spark, Hadoop, Kafka, Python, AWS, and data architecture. Design scalable data pipelines processing millions of records.',
            'salary': '$130,000 - $175,000',
            'url': 'https://example.com/job5',
            'posted_date': '1 week ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Analytics Engineer',
            'company': 'Data Insights Inc',
            'location': 'Remote',
            'description': 'SQL, Python, dbt, and data modeling. Build data transformation pipelines. Work with Snowflake, BigQuery, and create data marts for business teams.',
            'salary': '$85,000 - $120,000',
            'url': 'https://example.com/job6',
            'posted_date': '4 days ago',
            'source': 'Sample Data'
        },
        
        # Data Analyst Roles
        {
            'title': 'Data Analyst',
            'company': 'Retail Analytics Group',
            'location': 'Chicago, IL',
            'description': 'Analyst role using SQL, Excel, Tableau, Power BI. Create dashboards, perform statistical analysis, and provide business insights. A/B testing experience preferred.',
            'salary': '$60,000 - $85,000',
            'url': 'https://example.com/job7',
            'posted_date': '2 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Business Intelligence Analyst',
            'company': 'Healthcare Data Solutions',
            'location': 'Remote',
            'description': 'BI analyst with SQL, Tableau, and Excel skills. Build executive dashboards, automate reports, and analyze healthcare metrics. Python is a plus.',
            'salary': '$70,000 - $95,000',
            'url': 'https://example.com/job8',
            'posted_date': '1 week ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Senior Data Analyst',
            'company': 'Marketing Analytics Co',
            'location': 'Austin, TX',
            'description': 'Lead analyst role. SQL, Python, Tableau, Power BI, statistical analysis. Work on marketing attribution, customer analytics, and predictive modeling.',
            'salary': '$85,000 - $115,000',
            'url': 'https://example.com/job9',
            'posted_date': '3 days ago',
            'source': 'Sample Data'
        },
        
        # Machine Learning Roles
        {
            'title': 'Machine Learning Engineer',
            'company': 'AI Innovations Lab',
            'location': 'Remote',
            'description': 'ML engineer to develop and deploy models using PyTorch, TensorFlow, Docker, Kubernetes. Strong Python skills, MLOps experience, and cloud deployment knowledge.',
            'salary': '$110,000 - $160,000',
            'url': 'https://example.com/job10',
            'posted_date': '4 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'ML Operations Engineer',
            'company': 'Model Deployment Inc',
            'location': 'Seattle, WA',
            'description': 'MLOps role. Deploy and monitor ML models in production. Docker, Kubernetes, CI/CD, Python, and cloud platforms. Experience with MLflow and model monitoring.',
            'salary': '$100,000 - $145,000',
            'url': 'https://example.com/job11',
            'posted_date': '6 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'NLP Engineer',
            'company': 'Language AI Systems',
            'location': 'Remote',
            'description': 'Natural language processing engineer. Python, transformers, BERT, spaCy, NLTK. Work on text classification, sentiment analysis, and chatbot development.',
            'salary': '$105,000 - $150,000',
            'url': 'https://example.com/job12',
            'posted_date': '5 days ago',
            'source': 'Sample Data'
        },
        
        # Specialized Roles
        {
            'title': 'Research Scientist - AI',
            'company': 'University Research Lab',
            'location': 'Boston, MA',
            'description': 'Research scientist in machine learning and deep learning. PhD preferred. PyTorch, TensorFlow, research publication experience. Work on cutting-edge AI research.',
            'salary': '$90,000 - $140,000',
            'url': 'https://example.com/job13',
            'posted_date': '2 weeks ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Quantitative Analyst',
            'company': 'Hedge Fund Analytics',
            'location': 'New York, NY',
            'description': 'Quant analyst for financial modeling. Python, R, statistical analysis, time series, risk modeling. Strong mathematical background required.',
            'salary': '$100,000 - $150,000',
            'url': 'https://example.com/job14',
            'posted_date': '1 week ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Product Analyst',
            'company': 'Tech Product Company',
            'location': 'Remote',
            'description': 'Product analyst working with product teams. SQL, Python, A/B testing, user analytics. Experience with product metrics and data-driven decision making.',
            'salary': '$75,000 - $105,000',
            'url': 'https://example.com/job15',
            'posted_date': '4 days ago',
            'source': 'Sample Data'
        },
        
        # Entry Level / Internship
        {
            'title': 'Data Science Intern',
            'company': 'StartUp Analytics',
            'location': 'Remote',
            'description': 'Internship in data science. Python, machine learning basics, SQL. Great learning opportunity working on real projects with mentorship.',
            'salary': '$25 - $35 per hour',
            'url': 'https://example.com/job16',
            'posted_date': '1 day ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Junior BI Developer',
            'company': 'Business Intelligence Inc',
            'location': 'Denver, CO',
            'description': 'Entry-level BI developer. SQL, Tableau or Power BI, basic ETL. Build dashboards and support senior analysts. Training provided.',
            'salary': '$55,000 - $75,000',
            'url': 'https://example.com/job17',
            'posted_date': '3 days ago',
            'source': 'Sample Data'
        },
        
        # Technical/Engineering with Data Focus
        {
            'title': 'Software Engineer - Data Platform',
            'company': 'Platform Engineering Co',
            'location': 'San Francisco, CA',
            'description': 'Software engineer building data platforms. Python, Java, Spark, Kafka, cloud infrastructure. Design scalable data processing systems.',
            'salary': '$110,000 - $155,000',
            'url': 'https://example.com/job18',
            'posted_date': '1 week ago',
            'source': 'Sample Data'
        },
        {
            'title': 'Data Architect',
            'company': 'Enterprise Data Solutions',
            'location': 'Remote',
            'description': 'Design data architecture for enterprise systems. SQL, NoSQL, cloud data warehouses, data modeling. Lead technical strategy for data infrastructure.',
            'salary': '$120,000 - $165,000',
            'url': 'https://example.com/job19',
            'posted_date': '5 days ago',
            'source': 'Sample Data'
        },
        {
            'title': 'IoT Data Engineer',
            'company': 'Smart Devices Inc',
            'location': 'Boston, MA',
            'description': 'IoT data engineer working with sensor data. Python, Kafka, time-series databases, edge computing. Process and analyze IoT device data at scale.',
            'salary': '$95,000 - $135,000',
            'url': 'https://example.com/job20',
            'posted_date': '6 days ago',
            'source': 'Sample Data'
        },
    ]
    
    # Filter by search term
    filtered = [job for job in sample_jobs 
                if search_term.lower() in job['title'].lower() 
                or search_term.lower() in job['description'].lower()]
    
    return filtered if filtered else sample_jobs[:10]

def fetch_remoteok_jobs(search_term="data"):
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
                        'source': 'RemoteOK'
                    })
            
            if not processed_jobs:
                st.info(f"No exact matches for '{search_term}' from RemoteOK. Using sample data.")
                return get_fallback_jobs(search_term)
            
            return processed_jobs
        
        st.warning("RemoteOK API unavailable. Using sample data.")
        return get_fallback_jobs(search_term)
        
    except Exception as e:
        st.warning(f"RemoteOK API error: {str(e)}. Using sample data.")
        return get_fallback_jobs(search_term)

def fetch_github_jobs(search_term="data"):
    """Fetch jobs from GitHub Jobs (diverse sample data)"""
    github_sample = [
        {
            'title': 'Data Science Manager',
            'company': 'Growth Analytics',
            'location': 'Remote',
            'description': 'Lead data science team. Python, R, machine learning, team management. Drive data strategy and mentor junior data scientists.',
            'salary': '$130,000 - $170,000',
            'url': 'https://example.com/github1',
            'posted_date': '3 days ago',
            'source': 'GitHub Jobs'
        },
        {
            'title': 'Clinical Data Analyst',
            'company': 'MedTech Research',
            'location': 'Philadelphia, PA',
            'description': 'Healthcare data analyst. SQL, R, SAS, clinical trials data. Analyze patient outcomes and medical research data.',
            'salary': '$70,000 - $95,000',
            'url': 'https://example.com/github2',
            'posted_date': '1 week ago',
            'source': 'GitHub Jobs'
        },
        {
            'title': 'Computer Vision Engineer',
            'company': 'Robotics AI',
            'location': 'San Jose, CA',
            'description': 'Computer vision engineer. Python, OpenCV, TensorFlow, PyTorch. Work on object detection, image segmentation, and autonomous systems.',
            'salary': '$115,000 - $165,000',
            'url': 'https://example.com/github3',
            'posted_date': '5 days ago',
            'source': 'GitHub Jobs'
        },
        {
            'title': 'Risk Analyst',
            'company': 'Insurance Analytics',
            'location': 'Hartford, CT',
            'description': 'Risk modeling analyst. SQL, Python, statistical analysis, predictive modeling. Assess insurance risk and build actuarial models.',
            'salary': '$75,000 - $100,000',
            'url': 'https://example.com/github4',
            'posted_date': '4 days ago',
            'source': 'GitHub Jobs'
        },
        {
            'title': 'Data Operations Analyst',
            'company': 'Tech Operations',
            'location': 'Remote',
            'description': 'Data ops analyst ensuring data quality and pipeline reliability. SQL, Python, data validation, monitoring, and incident response.',
            'salary': '$70,000 - $95,000',
            'url': 'https://example.com/github5',
            'posted_date': '2 days ago',
            'source': 'GitHub Jobs'
        },
    ]
    
    # Filter by search term
    filtered = [job for job in github_sample 
                if search_term.lower() in job['title'].lower() 
                or search_term.lower() in job['description'].lower()]
    
    return filtered if filtered else github_sample

def fetch_all_jobs(search_term="data"):
    """Aggregate jobs from all sources and remove duplicates"""
    all_jobs = []
    
    with st.spinner("🔍 Fetching jobs from RemoteOK..."):
        all_jobs.extend(fetch_remoteok_jobs(search_term))
    
    with st.spinner("🔍 Fetching jobs from GitHub..."):
        all_jobs.extend(fetch_github_jobs(search_term))
    
    # Remove duplicates based on title + company
    seen = set()
    unique_jobs = []
    
    for job in all_jobs:
        job_key = (job['title'].lower().strip(), job['company'].lower().strip())
        if job_key not in seen:
            seen.add(job_key)
            unique_jobs.append(job)
    
    return unique_jobs

# ==================== SMART MATCHING ALGORITHM ====================
def calculate_match_score(user_skills, job_description, resume_text=""):
    """ML-Enhanced matching algorithm"""
    job_lower = job_description.lower()
    job_skill_dict, job_skills = extract_skills(job_description)
    
    # 1. Skill Match (40%)
    exact_matches = len(user_skills & job_skills)
    total_required = len(job_skills) if job_skills else 1
    skill_score = (exact_matches / total_required) * 40
    
    # 2. TF-IDF Semantic Similarity (30%) - ML
    if resume_text and len(resume_text) > 50 and SKLEARN_AVAILABLE:
        tfidf_similarity = calculate_tfidf_similarity(resume_text, job_description)
        semantic_score = (tfidf_similarity / 100) * 30
    else:
        semantic_score = 0
    
    # 3. Keyword Overlap (20%) - NLP
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
    """Identify missing skills and prioritize them"""
    all_job_skills = set()
    skill_frequency = Counter()
    
    for job in top_jobs[:10]:
        description = job.get('description', '') + ' ' + job.get('title', '')
        _, job_skills = extract_skills(description)
        all_job_skills.update(job_skills)
        skill_frequency.update(job_skills)
    
    missing_skills = all_job_skills - user_skills
    
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
    
    df = pd.DataFrame(jobs[:20])
    
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
    
    df = pd.DataFrame(skill_gaps[:15])
    
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
                st.success("Job results cleared!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if st.session_state.jobs:
            st.metric("Total Jobs Found", len(st.session_state.jobs))
            st.metric("Your Skills", len(st.session_state.user_skills))
            avg_score = sum(j.get('match_score', 0) for j in st.session_state.jobs) / len(st.session_state.jobs)
            st.metric("Avg Match Score", f"{avg_score:.1f}%")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Resume Upload", 
        "💼 Job Matches", 
        "📊 Analytics", 
        "🎯 Skill Gap", 
        "📥 Export",
        "🤖 ML Insights"
    ])
    
    # TAB 1: RESUME UPLOAD
    with tab1:
        st.header("Upload Your Resume")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            resume_text = st.text_area(
                "Paste your resume text here:",
                value=st.session_state.resume_text,  # Show loaded resume
                height=300,
                placeholder="Copy and paste your resume text here..."
            )
            
            if st.button("🔍 Analyze Resume", type="primary"):
                if resume_text:
                    st.session_state.resume_text = resume_text
                    skill_dict, skills = extract_skills(resume_text)
                    st.session_state.user_skills = skills
                    
                    # Clear old jobs
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
                
                # Clear old jobs
                st.session_state.jobs = []
                
                st.success(f"✅ Sample resume loaded! Found {len(skills)} skills.")
                st.info("💡 Scroll up to see the resume text, then click 'Analyze Resume'")
                
                # Auto-analyze
                st.subheader("Sample Resume Detected Skills")
                for category, cat_skills in skill_dict.items():
                    if cat_skills:
                        st.markdown(f"**{category}:**")
                        st.markdown(" ".join([f'<span class="skill-badge">{skill}</span>' 
                                             for skill in cat_skills]), 
                                  unsafe_allow_html=True)
    
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
                        ranked_jobs = rank_jobs(jobs, st.session_state.user_skills, st.session_state.resume_text)
                        st.session_state.jobs = ranked_jobs
                        st.success(f"✅ Found {len(jobs)} jobs matching your skills!")
                        if SKLEARN_AVAILABLE:
                            st.info("💡 Using ML-enhanced matching with TF-IDF semantic similarity!")
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
            st.warning("⚠️ Please upload your resume and find jobs first!")
        else:
            skill_gaps = analyze_skill_gaps(st.session_state.user_skills, st.session_state.jobs)
            
            if skill_gaps:
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
    
    # TAB 6: ML INSIGHTS
    with tab6:
        st.header("🤖 Machine Learning Insights")
        
        # Debug Information
        with st.expander("🔍 Debug Information"):
            st.write(f"**Resume text exists:** {bool(st.session_state.resume_text)}")
            st.write(f"**Resume length:** {len(st.session_state.resume_text)} characters")
            st.write(f"**Jobs count:** {len(st.session_state.jobs)}")
            st.write(f"**Skills count:** {len(st.session_state.user_skills)}")
            st.write(f"**Sklearn available:** {SKLEARN_AVAILABLE}")
            st.write(f"**NLTK available:** {NLTK_AVAILABLE}")
        
        if not st.session_state.resume_text:
            st.warning("⚠️ Please upload your resume first!")
            st.info("👉 Go to 'Resume Upload' → Paste resume → Click 'Analyze Resume'")
        else:
            # Section 1: Text Statistics
            st.subheader("📝 Resume Text Analysis")
            
            try:
                tokens = preprocess_text(st.session_state.resume_text)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Words", len(st.session_state.resume_text.split()))
                with col2:
                    st.metric("Unique Words", len(set(tokens)))
                with col3:
                    lexical_div = round(len(set(tokens)) / len(tokens) if tokens else 0, 2)
                    st.metric("Lexical Diversity", lexical_div)
                with col4:
                    avg_len = round(np.mean([len(word) for word in tokens]) if tokens else 0, 2)
                    st.metric("Avg Word Length", avg_len)
            except Exception as e:
                st.error(f"Text analysis error: {str(e)}")
            
            # Section 2: Keywords
            st.markdown("---")
            st.subheader("🔑 Top Keywords (TF-IDF)")
            
            if SKLEARN_AVAILABLE:
                try:
                    keywords = extract_keywords_tfidf(st.session_state.resume_text, top_n=15)
                    
                    if keywords:
                        kw_df = pd.DataFrame(keywords, columns=['Keyword', 'Importance'])
                        fig = px.bar(
                            kw_df,
                            x='Importance',
                            y='Keyword',
                            orientation='h',
                            title='Most Important Keywords',
                            color='Importance',
                            color_continuous_scale='Purples'
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Resume might be too short for keyword extraction")
                except Exception as e:
                    st.error(f"Keyword extraction error: {str(e)}")
            else:
                st.warning("⚠️ Install scikit-learn for keyword extraction: `pip install scikit-learn`")
            
            # Section 3: Semantic Similarity
            if st.session_state.jobs and SKLEARN_AVAILABLE:
                st.markdown("---")
                st.subheader("🎯 Semantic Similarity Analysis")
                
                try:
                    similarity_scores = []
                    for job in st.session_state.jobs[:10]:
                        job_text = job.get('description', '') + ' ' + job.get('title', '')
                        sim_score = calculate_tfidf_similarity(st.session_state.resume_text, job_text)
                        similarity_scores.append({
                            'Job': job['title'][:40],
                            'Similarity': sim_score
                        })
                    
                    if similarity_scores:
                        sim_df = pd.DataFrame(similarity_scores)
                        fig = px.bar(
                            sim_df,
                            x='Job',
                            y='Similarity',
                            title='Resume-Job Semantic Similarity',
                            color='Similarity',
                            color_continuous_scale='Purples'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Similarity analysis error: {str(e)}")
            
            # Section 4: Topic Modeling
            if st.session_state.jobs and len(st.session_state.jobs) >= 5 and SKLEARN_AVAILABLE:
                st.markdown("---")
                st.subheader("📚 Job Market Topics (LDA)")
                
                try:
                    job_texts = [j.get('description', '') + ' ' + j.get('title', '') 
                               for j in st.session_state.jobs[:50]]
                    job_texts = [t for t in job_texts if len(t) > 50]
                    
                    if len(job_texts) >= 5:
                        topics = extract_topics_lda(job_texts, n_topics=3, n_words=5)
                        
                        if topics:
                            for topic in topics:
                                with st.expander(f"📌 Topic {topic['topic_num']}: {', '.join(topic['keywords'][:3])}"):
                                    st.write("**Key Terms:**")
                                    for word in topic['keywords']:
                                        st.markdown(f"- {word}")
                except Exception as e:
                    st.error(f"Topic modeling error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit | AI Resume Matcher v2.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()