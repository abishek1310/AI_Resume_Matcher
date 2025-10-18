"""
Machine Learning Based Job Matcher
Save this as: ml_matcher.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class MLJobMatcher:
    """Advanced ML-based job matching"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        
        # Skill importance weights (learned from market data)
        self.skill_weights = {
            'python': 1.0,
            'machine learning': 0.95,
            'sql': 0.9,
            'deep learning': 0.88,
            'tensorflow': 0.85,
            'pytorch': 0.85,
            'data analysis': 0.82,
            'aws': 0.8,
            'docker': 0.75,
            'kubernetes': 0.72,
            'statistics': 0.7,
            'excel': 0.6,
            'tableau': 0.65,
            'git': 0.6
        }
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        text = text.lower()
        # Expand abbreviations
        text = text.replace('ml', 'machine learning')
        text = text.replace('dl', 'deep learning')
        text = text.replace('nlp', 'natural language processing')
        text = text.replace('cv', 'computer vision')
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_features(self, resume_text, job_description):
        """Extract features for ML matching"""
        features = {}
        
        # Text similarity features
        resume_clean = self.preprocess_text(resume_text)
        job_clean = self.preprocess_text(job_description)
        
        # Create TF-IDF vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform([resume_clean, job_clean])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            features['text_similarity'] = similarity
        except:
            features['text_similarity'] = 0
        
        # Keyword matching features
        important_keywords = ['python', 'machine learning', 'sql', 'data', 'analysis', 
                            'statistics', 'model', 'algorithm', 'deployment']
        
        resume_words = set(resume_clean.split())
        job_words = set(job_clean.split())
        
        keyword_matches = 0
        for keyword in important_keywords:
            if keyword in resume_words and keyword in job_words:
                keyword_matches += 1
        
        features['keyword_match_ratio'] = keyword_matches / len(important_keywords)
        
        return features
    
    def calculate_smart_match(self, resume_text, job, user_skills):
        """Calculate intelligent match score"""
        
        # 1. Content similarity (40%)
        features = self.extract_features(
            resume_text,
            job.get('description', '') + ' ' + job.get('title', '')
        )
        content_score = features['text_similarity'] * 40
        
        # 2. Skill matching (40%)
        job_skills = self.extract_skills_from_text(job.get('description', ''))
        if 'required_skills' in job:
            job_skills.extend(job['required_skills'])
        
        skill_score = self.calculate_skill_match(user_skills, job_skills) * 40
        
        # 3. Keyword bonus (20%)
        keyword_score = features['keyword_match_ratio'] * 20
        
        # 4. Special bonuses
        bonus = 0
        
        # Location bonus
        if 'remote' in str(job.get('location', '')).lower():
            bonus += 5
        
        # Fresh posting bonus
        if 'today' in str(job.get('posted', '')).lower():
            bonus += 3
        
        # Entry-level bonus (for co-op seeker)
        if any(word in str(job.get('title', '')).lower() 
               for word in ['junior', 'entry', 'intern', 'co-op']):
            bonus += 10
        
        total_score = content_score + skill_score + keyword_score + bonus
        
        # Cap at 100
        total_score = min(100, total_score)
        
        # Prepare insights
        insights = self.generate_insights(user_skills, job_skills, total_score)
        
        return {
            'score': round(total_score, 1),
            'content_match': round(content_score, 1),
            'skill_match': round(skill_score, 1),
            'keyword_match': round(keyword_score, 1),
            'bonus_points': bonus,
            'insights': insights,
            'confidence': self.calculate_confidence(total_score, len(user_skills))
        }
    
    def extract_skills_from_text(self, text):
        """Extract skills from job description"""
        text_lower = text.lower()
        found_skills = []
        
        # Common skills to look for
        skill_patterns = {
            'python', 'java', 'javascript', 'sql', 'r',
            'machine learning', 'deep learning', 'neural network',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'pandas', 'numpy', 'matplotlib', 'seaborn',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'git', 'jenkins', 'ci/cd', 'agile', 'scrum',
            'tableau', 'power bi', 'excel', 'spark', 'hadoop',
            'mongodb', 'postgresql', 'mysql', 'redis',
            'statistics', 'mathematics', 'algorithm',
            'nlp', 'computer vision', 'data analysis','sql','r'
        }
        
        for skill in skill_patterns:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def calculate_skill_match(self, user_skills, job_skills):
        """Calculate weighted skill match"""
        if not job_skills:
            return 1.0  # No skills required
        
        user_skills_set = set([s.lower() for s in user_skills])
        job_skills_set = set([s.lower() for s in job_skills])
        
        matched_weight = 0
        total_weight = 0
        
        for skill in job_skills_set:
            weight = self.skill_weights.get(skill, 0.5)
            total_weight += weight
            if skill in user_skills_set:
                matched_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return matched_weight / total_weight
    
    def generate_insights(self, user_skills, job_skills, score):
        """Generate actionable insights"""
        insights = []
        
        if score >= 80:
            insights.append("🎯 Excellent match! You're a strong candidate.")
        elif score >= 60:
            insights.append("👍 Good match! Consider highlighting relevant projects.")
        else:
            insights.append("📚 Moderate match. Focus on gaining more relevant skills.")
        
        # Skill gap analysis
        user_set = set([s.lower() for s in user_skills])
        job_set = set([s.lower() for s in job_skills])
        missing = job_set - user_set
        
        if missing:
            top_missing = sorted(missing, 
                               key=lambda x: self.skill_weights.get(x, 0.5), 
                               reverse=True)[:3]
            insights.append(f"💡 Key skills to add: {', '.join(top_missing)}")
        
        return insights
    
    def calculate_confidence(self, score, num_skills):
        """Calculate confidence in the match"""
        if score > 80 and num_skills > 5:
            return 0.9
        elif score > 60 and num_skills > 3:
            return 0.7
        elif score > 40:
            return 0.5
        else:
            return 0.3
    
    def rank_jobs(self, resume_text, user_skills, jobs):
        """Rank all jobs by match score"""
        ranked_jobs = []
        
        for job in jobs:
            match_result = self.calculate_smart_match(resume_text, job, user_skills)
            job['match_details'] = match_result
            ranked_jobs.append(job)
        
        # Sort by score
        ranked_jobs.sort(key=lambda x: x['match_details']['score'], reverse=True)
        
        return ranked_jobs

# Example usage
if __name__ == "__main__":
    matcher = MLJobMatcher()
    
    sample_resume = "Python developer with machine learning experience using TensorFlow"
    sample_skills = ['python', 'machine learning', 'tensorflow','sql']
    
    sample_job = {
        'title': 'Data Scientist',
        'description': 'Looking for Python expert with ML experience',
        'required_skills': ['python', 'sql', 'machine learning']
    }
    
    result = matcher.calculate_smart_match(sample_resume, sample_job, sample_skills)
    print(f"Match Score: {result['score']}%")
    print(f"Insights: {result['insights']}")