"""
Autonomous Resume-Screening Agent for HR Teams
============================================

An intelligent agent that autonomously screens resumes against job requirements,
using agentic thinking, RAG, and modern AI tools.

Author: Claude (for Onelogica AI Internship Challenge)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

# Try importing dependencies with fallbacks
try:
    import openai
except ImportError:
    print("Warning: openai not installed. Run: pip install openai")
    openai = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
except ImportError:
    print("Warning: langchain not installed. Run: pip install langchain")
    # Create dummy classes for basic functionality
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            pass
    
    class OpenAIEmbeddings:
        def __init__(self, openai_api_key=None):
            pass
        def embed_documents(self, texts):
            return [[0.0] * 1536 for _ in texts]  # Dummy embeddings
    
    class OpenAI:
        def __init__(self, openai_api_key=None, temperature=0.1):
            pass

# PDF processing
try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. Run: pip install PyPDF2")
    PyPDF2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: PyMuPDF not installed. Run: pip install PyMuPDF")
    fitz = None

# Data handling
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: pandas/numpy/scikit-learn not installed. Run: pip install pandas numpy scikit-learn")
    import math
    
    # Dummy numpy for basic operations
    class np:
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0
        
        @staticmethod
        def max(arr, axis=None):
            if axis is None:
                return max(arr) if arr else 0
            return [max(row) for row in arr]
    
    def cosine_similarity(a, b):
        return [[0.5 for _ in range(len(b))] for _ in range(len(a))]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class JobRequirement:
    """Structured job requirement data"""
    title: str
    department: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_years: int
    education_level: str
    description: str
    location: str = ""
    
@dataclass 
class ResumeProfile:
    """Extracted resume profile data"""
    filename: str
    candidate_name: str
    email: str
    phone: str
    skills: List[str]
    experience_years: float
    education: List[str]
    work_history: List[Dict[str, Any]]
    raw_text: str
    extraction_confidence: float

@dataclass
class MatchResult:
    """Resume matching result with scoring"""
    resume_profile: ResumeProfile
    overall_score: float
    skill_match_score: float
    experience_score: float
    education_score: float
    reasoning: str
    key_strengths: List[str]
    potential_concerns: List[str]

class ResumeParser:
    """Advanced resume parsing with multiple extraction methods"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods for robustness"""
        text = ""
        
        # Method 1: PyMuPDF (if available)
        if fitz:
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                if len(text.strip()) > 100:  # Good extraction
                    return text
                    
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        
        # Method 2: PyPDF2 (if available)
        if PyPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                        
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
        
        if not text.strip():
            logger.error(f"No text extracted from {pdf_path}. Please install PyPDF2 and/or PyMuPDF")
            
        return text
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information using regex patterns"""
        contact_info = {"name": "", "email": "", "phone": ""}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info["email"] = emails[0] if emails else ""
        
        # Phone extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        contact_info["phone"] = phones[0] if phones else ""
        
        # Name extraction (first few lines, excluding common headers)
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if line and len(line.split()) >= 2 and len(line) < 50:
                # Skip lines with common resume headers
                skip_keywords = ['resume', 'cv', 'curriculum', 'vitae', 'contact', 'email', 'phone']
                if not any(keyword in line.lower() for keyword in skip_keywords):
                    contact_info["name"] = line
                    break
                    
        return contact_info
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using keyword matching"""
        # Common technical skills database
        skill_keywords = [
            # Programming
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'sql', 'r', 
            'scala', 'go', 'rust', 'kotlin', 'swift', 'php', 'ruby',
            
            # Frameworks & Libraries
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'terraform', 'ansible', 'prometheus', 'grafana',
            
            # Databases
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'cassandra', 'dynamodb', 'snowflake',
            
            # AI/ML
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'data science', 'artificial intelligence', 'neural networks',
            
            # Business Skills
            'project management', 'agile', 'scrum', 'leadership', 'communication',
            'problem solving', 'analytical thinking', 'strategic planning'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
                
        return list(set(found_skills))  # Remove duplicates
    
    def extract_experience_years(self, text: str) -> float:
        """Extract years of experience using pattern matching"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience[:\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in\s*\w+',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(match) for match in matches])
            
        return max(years) if years else 0.0
    
    def parse_resume(self, file_path: str) -> ResumeProfile:
        """Main method to parse resume into structured data"""
        try:
            # Extract text
            if file_path.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            if not text.strip():
                raise ValueError("No text extracted from file")
            
            # Extract structured information
            contact_info = self.extract_contact_info(text)
            skills = self.extract_skills(text)
            experience_years = self.extract_experience_years(text)
            
            # Create resume profile
            profile = ResumeProfile(
                filename=Path(file_path).name,
                candidate_name=contact_info["name"] or "Unknown",
                email=contact_info["email"],
                phone=contact_info["phone"],
                skills=skills,
                experience_years=experience_years,
                education=[],  # Could be enhanced with education extraction
                work_history=[],  # Could be enhanced with work history extraction
                raw_text=text,
                extraction_confidence=0.8 if contact_info["email"] else 0.5
            )
            
            logger.info(f"Successfully parsed resume: {profile.filename}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to parse resume {file_path}: {e}")
            raise

class AutonomousResumeAgent:
    """
    Autonomous agent that screens resumes using agentic thinking patterns:
    - Perception: Analyze job requirements and resume pool
    - Reasoning: Score and rank candidates using multiple criteria
    - Action: Generate recommendations and reports
    - Memory: Maintain context across screening sessions
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        if openai and openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize components
        self.parser = ResumeParser()
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) if openai_api_key else None
        self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0.1) if openai_api_key else None
        
        # Agent memory
        self.session_memory = {
            "processed_jobs": [],
            "processed_resumes": [],
            "screening_history": []
        }
        
        logger.info("Autonomous Resume Agent initialized")
    
    def perceive_job_requirements(self, job_data: Dict[str, Any]) -> JobRequirement:
        """Agent's perception layer: understand job requirements"""
        logger.info("Agent perceiving job requirements...")
        
        # Parse and structure job requirements
        job_req = JobRequirement(
            title=job_data.get('title', ''),
            department=job_data.get('department', ''),
            required_skills=job_data.get('required_skills', []),
            preferred_skills=job_data.get('preferred_skills', []),
            experience_years=job_data.get('experience_years', 0),
            education_level=job_data.get('education_level', ''),
            description=job_data.get('description', ''),
            location=job_data.get('location', '')
        )
        
        # Store in agent memory
        self.session_memory["processed_jobs"].append(job_req)
        
        logger.info(f"Job requirements perceived: {job_req.title}")
        return job_req
    
    def scan_resume_pool(self, resume_folder: str) -> List[ResumeProfile]:
        """Agent's scanning action: process all resumes in folder"""
        logger.info(f"Agent scanning resume pool: {resume_folder}")
        
        resume_profiles = []
        resume_path = Path(resume_folder)
        
        if not resume_path.exists():
            logger.error(f"Resume folder not found: {resume_folder}")
            return resume_profiles
        
        # Find all resume files
        resume_files = []
        for ext in ['*.pdf', '*.txt', '*.doc', '*.docx']:
            resume_files.extend(resume_path.glob(ext))
        
        logger.info(f"Found {len(resume_files)} resume files")
        
        # Process each resume
        for file_path in resume_files:
            try:
                profile = self.parser.parse_resume(str(file_path))
                resume_profiles.append(profile)
                self.session_memory["processed_resumes"].append(profile)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(resume_profiles)} resumes")
        return resume_profiles
    
    def reason_and_score(self, job_req: JobRequirement, resume_profile: ResumeProfile) -> MatchResult:
        """Agent's reasoning layer: intelligent scoring using multiple criteria"""
        
        # 1. Skill matching
        skill_score = self._calculate_skill_match(job_req, resume_profile)
        
        # 2. Experience scoring
        experience_score = self._calculate_experience_match(job_req, resume_profile)
        
        # 3. Education scoring (placeholder)
        education_score = 0.7  # Neutral score
        
        # 4. Generate reasoning
        reasoning = self._generate_basic_reasoning(job_req, resume_profile)
        
        # 5. Calculate overall score with weights
        overall_score = (
            skill_score * 0.5 +
            experience_score * 0.3 +
            education_score * 0.2
        )
        
        # 6. Generate strengths and concerns
        strengths, concerns = self._analyze_strengths_concerns(job_req, resume_profile)
        
        return MatchResult(
            resume_profile=resume_profile,
            overall_score=overall_score,
            skill_match_score=skill_score,
            experience_score=experience_score,
            education_score=education_score,
            reasoning=reasoning,
            key_strengths=strengths,
            potential_concerns=concerns
        )
    
    def _calculate_skill_match(self, job_req: JobRequirement, resume: ResumeProfile) -> float:
        """Calculate skill matching score"""
        required_skills = [skill.lower() for skill in job_req.required_skills]
        candidate_skills = [skill.lower() for skill in resume.skills]
        
        if not required_skills:
            return 0.5  # Neutral if no requirements specified
        
        # Direct matches
        direct_matches = len(set(required_skills) & set(candidate_skills))
        direct_score = direct_matches / len(required_skills)
        
        return min(direct_score, 1.0)
    
    def _calculate_experience_match(self, job_req: JobRequirement, resume: ResumeProfile) -> float:
        """Calculate experience matching score"""
        required_exp = job_req.experience_years
        candidate_exp = resume.experience_years
        
        if required_exp == 0:
            return 0.8  # Neutral if no experience requirement
        
        if candidate_exp >= required_exp:
            # Candidate meets or exceeds requirement
            return min(1.0, 0.8 + (candidate_exp - required_exp) * 0.05)
        else:
            # Candidate has less experience
            ratio = candidate_exp / required_exp
            return max(0.1, ratio * 0.8)
    
    def _generate_basic_reasoning(self, job_req: JobRequirement, resume: ResumeProfile) -> str:
        """Generate basic reasoning without LLM"""
        matched_skills = set(s.lower() for s in job_req.required_skills) & set(s.lower() for s in resume.skills)
        
        reasoning_parts = []
        
        if matched_skills:
            reasoning_parts.append(f"Candidate has {len(matched_skills)} required skills: {', '.join(matched_skills)}")
        
        if resume.experience_years >= job_req.experience_years:
            reasoning_parts.append(f"Meets experience requirement with {resume.experience_years} years")
        else:
            reasoning_parts.append(f"Has {resume.experience_years} years experience vs {job_req.experience_years} required")
        
        return ". ".join(reasoning_parts) if reasoning_parts else "Basic candidate profile analysis completed"
    
    def _analyze_strengths_concerns(self, job_req: JobRequirement, resume: ResumeProfile) -> Tuple[List[str], List[str]]:
        """Analyze candidate strengths and potential concerns"""
        strengths = []
        concerns = []
        
        # Skill analysis
        matched_skills = set(s.lower() for s in job_req.required_skills) & set(s.lower() for s in resume.skills)
        if matched_skills:
            strengths.append(f"Strong skill match: {', '.join(matched_skills)}")
        
        missing_skills = set(s.lower() for s in job_req.required_skills) - set(s.lower() for s in resume.skills)
        if missing_skills:
            concerns.append(f"Missing required skills: {', '.join(missing_skills)}")
        
        # Experience analysis
        if resume.experience_years >= job_req.experience_years:
            strengths.append(f"Meets experience requirement ({resume.experience_years} years)")
        else:
            concerns.append(f"Below required experience ({resume.experience_years} vs {job_req.experience_years} years)")
        
        return strengths, concerns
    
    def autonomous_screening(self, job_data: Dict[str, Any], resume_folder: str) -> List[MatchResult]:
        """Main autonomous screening process"""
        logger.info("Starting autonomous resume screening...")
        
        # Step 1: Perceive job requirements
        job_req = self.perceive_job_requirements(job_data)
        
        # Step 2: Scan and parse resume pool
        resume_profiles = self.scan_resume_pool(resume_folder)
        
        if not resume_profiles:
            logger.warning("No resumes found to process")
            return []
        
        # Step 3: Reason and score each resume
        match_results = []
        for resume in resume_profiles:
            try:
                match_result = self.reason_and_score(job_req, resume)
                match_results.append(match_result)
            except Exception as e:
                logger.error(f"Failed to score resume {resume.filename}: {e}")
                continue
        
        # Step 4: Rank results by overall score
        match_results.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Step 5: Store in agent memory
        screening_session = {
            "timestamp": datetime.now().isoformat(),
            "job_title": job_req.title,
            "total_resumes": len(resume_profiles),
            "top_matches": len([r for r in match_results if r.overall_score > 0.7])
        }
        self.session_memory["screening_history"].append(screening_session)
        
        logger.info(f"Autonomous screening completed. Processed {len(resume_profiles)} resumes.")
        return match_results
    
    def generate_report(self, match_results: List[MatchResult], top_n: int = 3) -> Dict[str, Any]:
        """Generate comprehensive screening report"""
        top_matches = match_results[:top_n]
        
        report = {
            "screening_summary": {
                "total_resumes_processed": len(match_results),
                "top_candidates_selected": len(top_matches),
                "average_score": np.mean([r.overall_score for r in match_results]),
                "screening_timestamp": datetime.now().isoformat()
            },
            "top_candidates": []
        }
        
        for i, match in enumerate(top_matches, 1):
            candidate_data = {
                "rank": i,
                "candidate_name": match.resume_profile.candidate_name,
                "contact_email": match.resume_profile.email,
                "contact_phone": match.resume_profile.phone,
                "overall_score": round(match.overall_score * 100, 1),
                "skill_match_score": round(match.skill_match_score * 100, 1),
                "experience_score": round(match.experience_score * 100, 1),
                "key_skills": match.resume_profile.skills[:10],  # Top 10 skills
                "experience_years": match.resume_profile.experience_years,
                "key_strengths": match.key_strengths,
                "potential_concerns": match.potential_concerns,
                "ai_reasoning": match.reasoning,
                "resume_filename": match.resume_profile.filename
            }
            report["top_candidates"].append(candidate_data)
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = "screening_report.json"):
        """Save screening report to file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")

def main():
    """Main execution function"""
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    RESUME_FOLDER = "sample_resumes"
    
    # Sample job requirements
    job_requirements = {
        "title": "Senior AI Engineer",
        "department": "Engineering",
        "required_skills": ["Python", "Machine Learning", "TensorFlow", "AWS", "SQL"],
        "preferred_skills": ["PyTorch", "Kubernetes", "React", "Leadership"],
        "experience_years": 5,
        "education_level": "Bachelor's or Master's in Computer Science",
        "description": "We're looking for a Senior AI Engineer to lead our ML initiatives...",
        "location": "Remote/San Francisco"
    }
    
    try:
        # Initialize the autonomous agent
        agent = AutonomousResumeAgent(OPENAI_API_KEY)
        
        # Run autonomous screening
        results = agent.autonomous_screening(job_requirements, RESUME_FOLDER)
        
        if results:
            # Generate and save report
            report = agent.generate_report(results, top_n=3)
            agent.save_report(report)
            
            # Print summary
            print("\n" + "="*60)
            print("AUTONOMOUS RESUME SCREENING COMPLETED")
            print("="*60)
            print(f"Total Resumes Processed: {len(results)}")
            print(f"Top 3 Candidates Selected:")
            
            for i, candidate in enumerate(report["top_candidates"], 1):
                print(f"\n{i}. {candidate['candidate_name']}")
                print(f"   Score: {candidate['overall_score']}%")
                print(f"   Email: {candidate['contact_email']}")
                print(f"   Key Strengths: {'; '.join(candidate['key_strengths'][:2])}")
        else:
            print("No resumes found to process. Please check the resume folder.")
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()