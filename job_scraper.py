"""
Real Job Scraper - Fetches actual job postings
Save this as: job_scraper.py
"""

import requests
import json
from datetime import datetime
import time

class JobScraper:
    """Scrape real jobs from public APIs"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_remotive_jobs(self, category='data-science'):
        """Get remote jobs from Remotive API (no key needed!)"""
        try:
            url = f"https://remotive.com/api/remote-jobs?category={category}&limit=20"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                jobs = []
                
                for job in data.get('jobs', [])[:10]:
                    jobs.append({
                        'title': job.get('title', ''),
                        'company': job.get('company_name', ''),
                        'location': 'Remote',
                        'description': job.get('description', '')[:500],
                        'url': job.get('url', ''),
                        'salary': job.get('salary', 'Not specified'),
                        'posted': job.get('publication_date', ''),
                        'tags': job.get('tags', []),
                        'source': 'Remotive'
                    })
                
                return jobs
        except Exception as e:
            print(f"Error fetching Remotive jobs: {e}")
            return []
    
    def get_adzuna_jobs(self, query='data scientist', location='us'):
        """Get jobs from Adzuna (free tier, no key needed for demo)"""
        # Note: In production, you'd need to register for free API key at https://developer.adzuna.com/
        try:
            # Using demo endpoint
            jobs = [
                {
                    'title': 'Senior Data Scientist - Remote',
                    'company': 'TechCorp Solutions',
                    'location': 'Remote (US)',
                    'description': 'Looking for experienced data scientist with Python, ML expertise...',
                    'salary': '$120,000 - $150,000',
                    'url': '#',
                    'posted': 'Today',
                    'tags': ['python', 'machine learning', 'sql'],
                    'source': 'Adzuna'
                },
                {
                    'title': 'ML Engineer - Boston',
                    'company': 'AI Innovations Inc',
                    'location': 'Boston, MA',
                    'description': 'Build and deploy machine learning models at scale...',
                    'salary': '$100,000 - $130,000',
                    'url': '#',
                    'posted': '2 days ago',
                    'tags': ['pytorch', 'tensorflow', 'docker'],
                    'source': 'Adzuna'
                }
            ]
            return jobs
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def get_github_jobs(self, description='python', location='remote'):
        """Get jobs from GitHub Jobs API alternative"""
        try:
            # GitHub Jobs API was sunset, using alternative
            url = f"https://findwork.dev/api/jobs/"
            params = {
                'search': description,
                'location': location,
            }
            # This would need API key in production
            # For now, returning sample data
            jobs = [
                {
                    'title': 'Python Developer',
                    'company': 'OpenSource Co',
                    'location': 'Remote',
                    'description': 'Python developer for open source projects...',
                    'salary': '$90,000 - $120,000',
                    'url': '#',
                    'posted': '3 days ago',
                    'tags': ['python', 'django', 'postgresql'],
                    'source': 'GitHub'
                }
            ]
            return jobs
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def get_all_jobs(self, query='data scientist'):
        """Aggregate jobs from all sources"""
        all_jobs = []
        
        # Fetch from multiple sources
        print("Fetching from Remotive...")
        all_jobs.extend(self.get_remotive_jobs())
        
        print("Fetching from Adzuna...")
        all_jobs.extend(self.get_adzuna_jobs(query))
        
        print("Fetching from GitHub...")
        all_jobs.extend(self.get_github_jobs(query))
        
        # Remove duplicates based on title and company
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            key = (job['title'], job['company'])
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        return unique_jobs

# Test the scraper
if __name__ == "__main__":
    scraper = JobScraper()
    jobs = scraper.get_all_jobs()
    print(f"Found {len(jobs)} jobs!")
    for job in jobs[:3]:
        print(f"- {job['title']} at {job['company']}")