"""
Text Preparation for CV and Job Announcement Content
Converts structured data into canonical text format for embedding
"""

import json
from typing import Dict, Any, List, Optional


def prepare_job_announcement_text(job_data: Dict[str, Any]) -> str:
    """
    Convert structured job announcement data into canonical text format.
    
    Args:
        job_data: Dictionary containing job announcement fields
    
    Returns:
        Formatted text string ready for embedding
    """
    sections = []
    
    # Job title / position (if available in description)
    if job_data.get('description'):
        sections.append(f"Position: {job_data['description'][:200]}")
    
    # Contract and work details
    contract_info = []
    if job_data.get('contract_type'):
        contract_info.append(f"Contract: {job_data['contract_type']}")
    if job_data.get('work_location'):
        contract_info.append(f"Location: {job_data['work_location']}")
    if contract_info:
        sections.append(', '.join(contract_info))
    
    # Requirements
    requirements = []
    if job_data.get('required_experience'):
        requirements.append(f"Experience: {job_data['required_experience']}")
    if job_data.get('education_level'):
        requirements.append(f"Education: {job_data['education_level']}")
    if requirements:
        sections.append("Requirements: " + ', '.join(requirements))
    
    # Skills and languages
    if job_data.get('languages'):
        languages = job_data['languages']
        if isinstance(languages, list):
            sections.append(f"Languages: {', '.join(languages)}")
        else:
            sections.append(f"Languages: {languages}")
    
    # Compensation and benefits
    comp_info = []
    if job_data.get('proposed_salary'):
        comp_info.append(f"Salary: {job_data['proposed_salary']}")
    if job_data.get('availability'):
        comp_info.append(f"Availability: {job_data['availability']}")
    if comp_info:
        sections.append(', '.join(comp_info))
    
    # Company information
    company_info = []
    if job_data.get('entreprise_name'):
        company_info.append(f"Company: {job_data['entreprise_name']}")
    if job_data.get('entreprise_sector'):
        company_info.append(f"Sector: {job_data['entreprise_sector']}")
    if job_data.get('entreprise_size'):
        company_info.append(f"Size: {job_data['entreprise_size']}")
    if company_info:
        sections.append(', '.join(company_info))
    
    # Mobility
    if job_data.get('mobility'):
        sections.append(f"Mobility: {job_data['mobility']}")
    
    return '\n'.join(sections)


def prepare_cv_text(cv_data: Dict[str, Any]) -> str:
    """
    Convert structured CV data into canonical text format.
    
    Args:
        cv_data: Dictionary containing user CV fields
    
    Returns:
        Formatted text string ready for embedding
    """
    sections = []
    
    # Professional identity
    identity = []
    if cv_data.get('full_name'):
        identity.append(cv_data['full_name'])
    if cv_data.get('professional_summary'):
        identity.append(f"Summary: {cv_data['professional_summary']}")
    if identity:
        sections.append(' - '.join(identity))
    
    # Contact and location
    contact_info = []
    if cv_data.get('location'):
        contact_info.append(f"Location: {cv_data['location']}")
    if cv_data.get('email'):
        contact_info.append(f"Email: {cv_data['email']}")
    if contact_info:
        sections.append(', '.join(contact_info))
    
    # Experience summary
    experience_parts = []
    if cv_data.get('total_years_experience'):
        experience_parts.append(f"Total Experience: {cv_data['total_years_experience']} years")
    
    # Parse work experience from JSONB
    if cv_data.get('work_experience'):
        work_exp = cv_data['work_experience']
        if isinstance(work_exp, str):
            work_exp = json.loads(work_exp)
        
        if isinstance(work_exp, list):
            for exp in work_exp[:3]:  # Top 3 most recent
                exp_parts = []
                if exp.get('title'):
                    exp_parts.append(exp['title'])
                if exp.get('company'):
                    exp_parts.append(f"at {exp['company']}")
                if exp.get('duration'):
                    exp_parts.append(f"({exp['duration']})")
                if exp_parts:
                    experience_parts.append(' '.join(exp_parts))
    
    if experience_parts:
        sections.append("Experience: " + ', '.join(experience_parts))
    
    # Technical skills
    skills = []
    if cv_data.get('technical_skills'):
        tech_skills = cv_data['technical_skills']
        if isinstance(tech_skills, list):
            skills.extend(tech_skills)
        else:
            skills.append(str(tech_skills))
    
    if cv_data.get('tools'):
        tools = cv_data['tools']
        if isinstance(tools, list):
            skills.extend(tools)
        else:
            skills.append(str(tools))
    
    if skills:
        sections.append(f"Skills: {', '.join(skills[:15])}")  # Top 15 skills
    
    # Languages
    if cv_data.get('languages'):
        languages = cv_data['languages']
        if isinstance(languages, list):
            sections.append(f"Languages: {', '.join(languages)}")
        else:
            sections.append(f"Languages: {languages}")
    
    # Education
    if cv_data.get('education'):
        education = cv_data['education']
        if isinstance(education, str):
            education = json.loads(education)
        
        if isinstance(education, list) and education:
            edu_parts = []
            for edu in education[:2]:  # Top 2 degrees
                edu_part = []
                if edu.get('degree'):
                    edu_part.append(edu['degree'])
                if edu.get('field'):
                    edu_part.append(f"in {edu['field']}")
                if edu.get('institution'):
                    edu_part.append(f"from {edu['institution']}")
                if edu_part:
                    edu_parts.append(' '.join(edu_part))
            
            if edu_parts:
                sections.append("Education: " + ', '.join(edu_parts))
    
    # Projects
    if cv_data.get('projects'):
        projects = cv_data['projects']
        if isinstance(projects, str):
            projects = json.loads(projects)
        
        if isinstance(projects, list):
            project_names = [p.get('name', p.get('title', '')) for p in projects[:3]]
            project_names = [name for name in project_names if name]
            if project_names:
                sections.append(f"Projects: {', '.join(project_names)}")
    
    # Certificates
    if cv_data.get('certificates'):
        certificates = cv_data['certificates']
        if isinstance(certificates, list):
            sections.append(f"Certifications: {', '.join(certificates[:5])}")
    
    return '\n'.join(sections)


def prepare_job_content_from_announcement(announcement: Dict[str, Any]) -> str:
    """
    Wrapper function to prepare job content from job_announcements table.
    
    Args:
        announcement: Row from job_announcements table
    
    Returns:
        Canonical text for embedding
    """
    return prepare_job_announcement_text(announcement)


def prepare_cv_content_from_user_cv(user_cv: Dict[str, Any]) -> str:
    """
    Wrapper function to prepare CV content from user_cvs table.
    
    Args:
        user_cv: Row from user_cvs table
    
    Returns:
        Canonical text for embedding
    """
    return prepare_cv_text(user_cv)


# Example outputs for reference
EXAMPLE_JOB_TEXT = """Position: Backend Engineer needed for growing fintech startup
Contract: CDI, Location: Paris with remote options
Requirements: Experience: 3-5 years, Education: Bachelor's degree in Computer Science
Languages: French, English
Salary: 50-60K EUR, Availability: Immediate
Company: TechCorp, Sector: Financial Technology, Size: 50-100 employees
Mobility: National"""

EXAMPLE_CV_TEXT = """Jean Dupont - Summary: Senior Backend Engineer with expertise in distributed systems
Location: Paris, Email: jean.dupont@email.com
Experience: Total Experience: 6 years, Senior Backend Engineer at StartupCo (3 years), Backend Developer at TechFirm (2 years), Junior Developer at WebAgency (1 year)
Skills: Python, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS, Microservices, REST APIs, GraphQL, CI/CD, Git, Linux, RabbitMQ, Elasticsearch
Languages: French, English, Spanish
Education: Master's in Computer Science from Université Paris-Saclay, Bachelor's in Software Engineering from INSA Lyon
Projects: E-commerce Platform, Real-time Analytics Dashboard, Payment Gateway Integration
Certifications: AWS Certified Solutions Architect, Docker Certified Associate"""


