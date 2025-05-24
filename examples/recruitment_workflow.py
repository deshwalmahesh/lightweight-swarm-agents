import os
import time
import json
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import fitz  # PyMuPDF

# Import PyLate for embedding similarity
from pylate import models, retrieve, indexes

# Import core components
from src.core import tool, ToolRegistry, BaseAgent, AgentConfig, LLM


load_dotenv() 

"""### Initialize OpenAI Client"""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])



@tool("download_job_description")
def download_job_description(url: str, output_path: str = "job_description.pdf") -> str:
    """
    Download job description from a given URL.
    
    Args:
        url: The URL to download the job description PDF from
        output_path: Path where the job description PDF will be saved
        
    Returns:
        Path to the downloaded file
    """
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {output_path}")
    return output_path

@tool("download_resumes")
def download_resumes(url: str, local_dir: str = "example_data", fetch_only: int = 3) -> List[str]:
    """
    Download resumes from the given URL.
    
    Args:
        url: The GitHub API URL to fetch resume files from
        local_dir: Local directory to save the files to
        fetch_only: Maximum number of files to download
        
    Returns:
        List of downloaded file paths
    """
    response = requests.get(url)

    if response.status_code != 200:
        print("Failed to retrieve folder contents:", response.text)
        return []

    data = response.json()
    os.makedirs(local_dir, exist_ok=True)
    
    downloaded_files = []
    print(f"{min(fetch_only, len(data))} files available for download:")
    
    for file in data[:fetch_only]:
        file_name = file["name"]
        file_path = os.path.join(local_dir, file_name)
        
        if os.path.exists(file_path):
            print(f"File {file_name} already exists, skipping")
            downloaded_files.append(file_path)
            continue
            
        download_url = file["download_url"]

        r = requests.get(download_url)
        with open(file_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {file_name}")
        downloaded_files.append(file_path)
        
    return downloaded_files


url = "https://raw.githubusercontent.com/mistralai/cookbook/main/mistral/agents/recruitment_agent/job_description.pdf"
output_path = "job_description.pdf"

# Use the tool registry to call the download_job_description function
jd_file_path = download_job_description(url, output_path)

# Use the tool registry to call the download_resumes function
resume_files = download_resumes(
    url = "https://api.github.com/repos/mistralai/cookbook/contents/mistral/agents/recruitment_agent/resumes",
    local_dir="example_data"
)

print(f"Downloaded resume files: {resume_files}")



class Skill(BaseModel):
    name: str = Field(description="Name of the skill or technology")
    level: Optional[str] = Field(description="Proficiency level (beginner, intermediate, advanced)")
    years: Optional[float] = Field(description="Years of experience with this skill")

class Education(BaseModel):
    degree: str = Field(description="Type of degree or certification obtained")
    field: str = Field(description="Field of study or specialization")
    institution: str = Field(description="Name of educational institution")
    year_completed: Optional[int] = Field(description="Year when degree was completed")
    gpa: Optional[float] = Field(description="Grade Point Average, typically on 4.0 scale")

class Experience(BaseModel):
    title: str = Field(description="Job title or position held")
    company: str = Field(description="Name of employer or organization")
    duration_years: float = Field(description="Duration of employment in years")
    skills_used: List[str] = Field(description="Skills utilized in this role")
    achievements: List[str] = Field(description="Key accomplishments or responsibilities")
    relevance_score: Optional[float] = Field(description="Relevance to current job opening (0-10 scale)")

class ContactDetails(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Primary email address for contact")
    phone: Optional[str] = Field(description="Phone number with country code if applicable")
    location: Optional[str] = Field(description="Current city and country/state")
    linkedin: Optional[str] = Field(description="LinkedIn profile URL")
    website: Optional[str] = Field(description="Personal or portfolio website URL")

class JobRequirements(BaseModel):
    required_skills: List[Skill] = Field(description="Skills that are mandatory for the position")
    preferred_skills: List[Skill] = Field(description="Skills that are desired but not required")
    min_experience_years: float = Field(description="Minimum years of experience required")
    required_education: List[Education] = Field(description="Mandatory educational qualifications")
    preferred_domains: List[str] = Field(description="Industry domains or sectors preferred for experience")

class CandidateProfile(BaseModel):
    contact_details: ContactDetails = Field(description="Candidate's personal and contact information")
    skills: List[Skill] = Field(description="Technical and soft skills possessed by the candidate")
    education: List[Education] = Field(description="Educational background and qualifications")
    experience: List[Experience] = Field(description="Professional work history and experience")

class SkillMatch(BaseModel):
    skill_name: str = Field(description="Name of the skill being evaluated")
    present: bool = Field(description="Whether the candidate possesses this skill")
    match_level: float = Field(description="How well the candidate's skill matches the requirement (0-10 scale)")
    confidence: float = Field(description="Confidence in the skill evaluation (0-1 scale)")
    notes: str = Field(description="Additional context about the skill match assessment")

class CandidateScore(BaseModel):
    technical_skills_score: float = Field(description="Assessment of technical capabilities (0-40 points)")
    experience_score: float = Field(description="Evaluation of relevant work experience (0-30 points)")
    education_score: float = Field(description="Rating of educational qualifications (0-15 points)")
    additional_score: float = Field(description="Score for other relevant factors (0-15 points)")
    embedding_similarity_score: Optional[float] = Field(description="Semantic similarity score between resume and job description (0-10 points)", default=0.0)
    total_score: float = Field(description="Aggregate candidate evaluation score (0-100)")
    key_strengths: List[str] = Field(description="Primary candidate advantages for this role")
    key_gaps: List[str] = Field(description="Areas where the candidate lacks desired qualifications")
    confidence: float = Field(description="Overall confidence in the evaluation accuracy (0-1 scale)")
    notes: str = Field(description="Supplementary observations about the candidate fit")

class CandidateResult(BaseModel):
    file_name: str = Field(description="Name of the source resume file")
    contact_details: ContactDetails = Field(description="Candidate's contact information")
    candidate_profile: CandidateProfile = Field(description="Complete extracted candidate profile")
    score: CandidateScore = Field(description="Detailed evaluation scores and assessment")



class Agent(BaseAgent):
    def __init__(self, name: str, client: OpenAI):
        # Create an AgentConfig for this agent
        config = AgentConfig(
            name=name,
            llm=LLM(model_name="gpt-4", client=client),
            task="recruitment workflow"
        )
        super().__init__(config)
        self.client = client  # Keep a reference to the original client for backward compatibility

    # Base process method - to be implemented by child classes
    def process(self, message):
        # This method will be implemented by child classes
        raise NotImplementedError("Subclasses must implement this method")

    # Send message to another agent
    def communicate(self, recipient_agent, message):
        return recipient_agent.process(message)



class DocumentAgent(Agent):
    def __init__(self, client: OpenAI):
        super().__init__("DocumentAgent", client)

    def process(self, file_info):
        """Process document extraction request"""
        file_path, file_name = file_info
        return self.extract_text_from_file(file_path, file_name)

    def extract_text_from_file(self, file_path: str, file_name: str) -> str:
        """Extract text from a file using PyMuPDF"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Open the document with PyMuPDF
            doc = fitz.open(file_path)
            
            # Extract text from all pages
            text = ""
            for page in doc:
                text += page.get_text()
                
            # Close the document
            doc.close()
            
            return text

        except Exception as e:
            print(f"Error extracting text from {file_name}: {str(e)}")
            return ""



class JobAnalysisAgent(Agent):
    def __init__(self, client: OpenAI):
        super().__init__("JobAnalysisAgent", client)

    def process(self, jd_text):
        """Process job description text"""
        return self.extract_job_requirements(jd_text)

    def extract_job_requirements(self, jd_text: str) -> JobRequirements:
        """Extract structured job requirements from a job description"""
        prompt = f"""
        Extract the key job requirements from the following job description.
        Focus on required skills, preferred skills, experience requirements, and education requirements.
        
        Provide your response as a JSON object with the following structure:
        {{
            "required_skills": [{{
                "name": "skill name",
                "level": "proficiency level",
                "years": years of experience
            }}],
            "preferred_skills": [...],
            "min_experience_years": number,
            "required_education": [...],
            "preferred_domains": [...]
        }}
        
        Job Description:
        {jd_text}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract structured job requirements from the job description."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)



class ResumeAnalysisAgent(Agent):
    def __init__(self, client: OpenAI):
        super().__init__("ResumeAnalysisAgent", client)

    def process(self, resume_text):
        """Process resume text"""
        return self.extract_candidate_profile(resume_text)

    def extract_candidate_profile(self, resume_text: str) -> CandidateProfile:
        """Extract structured candidate information from resume text"""
        prompt = f"""
        Extract the candidate's contact details, skills, education, and experience from the following resume.
        Be thorough and include all relevant information.
        
        Provide your response as a JSON object with the following structure:
        {{
            "contact_details": {{
                "name": "candidate name",
                "email": "email address",
                "phone": "phone number",
                "location": "location",
                "linkedin": "linkedin profile",
                "website": "website url"
            }},
            "skills": [{{
                "name": "skill name",
                "level": "proficiency level",
                "years": years of experience
            }}],
            "education": [...],
            "experience": [...]
        }}
        
        Resume:
        {resume_text}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract structured candidate information from the resume."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)



class EmbeddingSimilarityAgent(Agent):
    def __init__(self, client: OpenAI):
        super().__init__("EmbeddingSimilarityAgent", client)
        self.model = None
        try:
            from pylate import models
            self.model = models.ColBERT(
                model_name_or_path="lightonai/Reason-ModernColBERT",
            )
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        
    def process(self, data):
        """Process job description and resume text to calculate semantic similarity"""
        jd_text, resume_text = data
        return self.calculate_similarity(jd_text, resume_text)
    
    def calculate_similarity(self, jd_text: str, resume_text: str) -> float:
        """Calculate the semantic similarity between job description and resume"""
        if not self.model:
            return 5.0
        
        try:
            # Truncate texts if they're too long
            max_length = 4096
            jd_text = jd_text[:max_length] if len(jd_text) > max_length else jd_text
            resume_text = resume_text[:max_length] if len(resume_text) > max_length else resume_text
            
            # Encode job description as query
            jd_embeddings = self.model.encode(
                [jd_text],
                batch_size=1,
                is_query=True,
                show_progress_bar=False
            )
            
            # Encode resume as document
            resume_embeddings = self.model.encode(
                [resume_text],
                batch_size=1,
                is_query=False,
                show_progress_bar=False
            )
            
            # Create index for the resume
            from pylate import indexes
            index = indexes.Voyager(
                index_folder="temp-similarity-index",
                index_name="temp-index",
                override=True
            )
            
            # Add resume to index
            index.add_documents(
                documents_ids=["resume-1"],
                documents_embeddings=resume_embeddings
            )
            
            # Initialize retriever and get scores
            from pylate import retrieve
            retriever = retrieve.ColBERT(index=index)
            
            # Get scores
            results = retriever.retrieve(
                queries_embeddings=jd_embeddings,
                k=1
            )
            
            # Default middle score
            similarity_score = 5.0
            
            if results and len(results) > 0 and isinstance(results[0], list) and len(results[0]) > 0:
                first_result = results[0][0]
                
                if isinstance(first_result, dict) and 'score' in first_result:
                    raw_score = float(first_result['score'])
                    # Normalize score to 0-10 range
                    normalized_score = min(raw_score / 10, 10.0)
                    similarity_score = normalized_score
            
            return similarity_score
            
        except Exception as e:
            return 5.0
    
    def _interpret_score(self, score: float) -> str:
        if score >= 9.0:
            return "Excellent similarity - resume is highly aligned with the job requirements"
        elif score >= 7.5:
            return "Strong similarity - resume matches most of the job requirements"
        elif score >= 6.0:
            return "Good similarity - resume matches many of the job requirements"
        elif score >= 4.0:
            return "Moderate similarity - resume matches some of the job requirements"
        elif score >= 2.5:
            return "Limited similarity - resume matches few of the job requirements"
        else:
            return "Poor similarity - resume does not match the job requirements well"


class MatchingAgent(Agent):
    def __init__(self, client: OpenAI):
        super().__init__("MatchingAgent", client)

    def process(self, data):
        """Process job requirements and candidate profile to generate score"""
        job_requirements, candidate_profile, resume_text = data
        return self.evaluate_candidate(job_requirements, candidate_profile, resume_text)

    def evaluate_candidate(self, job_requirements: JobRequirements, candidate_profile: CandidateProfile, resume_text: str) -> CandidateScore:
        """Evaluate how well a candidate matches the job requirements"""
        # Convert to JSON for inclusion in the prompt
        job_req_json = json.dumps(job_requirements, indent=2)
        candidate_json = json.dumps(candidate_profile, indent=2)

        prompt = f"""
        Evaluate how well the candidate matches the job requirements.

        Job Requirements:
        {job_req_json}

        Candidate Profile:
        {candidate_json}

        Provide a detailed scoring breakdown, highlighting strengths and gaps.
        Assess the quality and relevance of the candidate's experience, not just keyword matches.
        Include confidence levels for your assessment.

        Technical skills should be scored out of 40 points.
        Experience should be scored out of 30 points.
        Education should be scored out of 15 points.
        Additional qualifications should be scored out of 15 points.
        The total score should be out of 100 points.
        
        Return your evaluation as a JSON object with the following structure:
        {{
            "technical_skills_score": numeric value between 0-40,
            "experience_score": numeric value between 0-30,
            "education_score": numeric value between 0-15,
            "additional_qualifications_score": numeric value between 0-15,
            "total_score": numeric value between 0-100,
            "strengths": [list of strings],
            "areas_for_improvement": [list of strings],
            "overall_recommendation": string,
            "confidence": numeric value between 0-1,
            "notes": string
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Evaluate the candidate's match to the job requirements with detailed scoring."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)



class EmailCommunicationAgent(Agent):
    def __init__(self, client: OpenAI, sender_email: str, app_password: str):
        super().__init__("EmailCommunicationAgent", client)
        self.sender_email = sender_email
        self.app_password = app_password

    def process(self, data):
        """Process email sending request"""
        candidate, calendly_link, subject = data
        return self.send_interview_invitation(candidate, calendly_link, subject)

    def send_interview_invitation(self, candidate, calendly_link: str, subject: str):
        """Generate and send personalized email to candidate"""
        name = candidate["contact_details"]['name']
        email = candidate["contact_details"]['email']

        # Create email HTML content
        html_content = f"""\
        <html>
          <body>
            <p>Hello {name},</p>
            <p>I'm the Hiring Manager from HireFive. Thank you for applying for the Data Scientist position at our company.</p>
            <p>We were impressed with your background and would like to schedule an initial screening call to discuss your experience and interest in the role.</p>
            <p>Please select a suitable time slot using our <a href="{calendly_link}">Calendly link</a>.</p>
            <p>Looking forward to speaking with you soon.</p>
            <p>Best regards,<br>
            Hiring Manager<br>
            HireFive</p>
          </body>
        </html>
        """

        if self.app_password:
            try:
                self.send_email(email, subject, html_content)
                return f"Email sent to {name} at {email}"
            except Exception as e:
                return f"Failed to send email to {name} ({email}): {str(e)}"
        else:
            return f"Would send email to {name} at {email} - Email subject: {subject}"

    def send_email(self, receiver_email, subject, html_content):
        """Send an email using Gmail SMTP"""
        # Create message container
        message = MIMEMultipart('alternative')
        message['From'] = self.sender_email
        message['To'] = receiver_email
        message['Subject'] = subject

        # Attach HTML part
        message.attach(MIMEText(html_content, 'html'))

        try:
            # Create SMTP session
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()  # Enable security

            # Login with Gmail account and app password
            server.login(self.sender_email, self.app_password)

            # Send email
            text = message.as_string()
            server.sendmail(self.sender_email, receiver_email, text)

        finally:
            server.quit()  # Close the connection



class CoordinatorAgent(Agent):
    def __init__(self, client: OpenAI):
        super().__init__("CoordinatorAgent", client)
        self.document_agent = DocumentAgent(client)
        self.job_analysis_agent = JobAnalysisAgent(client)
        self.resume_analysis_agent = ResumeAnalysisAgent(client)
        self.matching_agent = MatchingAgent(client)
        self.embedding_similarity_agent = EmbeddingSimilarityAgent(client)
        self.email_communication_agent = None  # Will be initialized later with email credentials

    def set_email_communication_agent(self, sender_email: str, app_password: str):
        """Initialize communication agent with email credentials"""
        self.email_communication_agent = EmailCommunicationAgent(self.client, sender_email, app_password)

    def process_hiring_workflow(self, jd_file_path: str, resume_dir: str, output_path: str,
                               threshold_score: float, calendly_link: str, email_subject: str):
        """
        Coordinate the entire hiring workflow from document processing to interview scheduling
        """
        results = []

        # Process job description
        print(f"Extracting text from job description...")
        jd_text = self.document_agent.process((jd_file_path, os.path.basename(jd_file_path)))

        if not jd_text:
            print("Failed to extract text from job description. Aborting.")
            return results

        # Extract job requirements
        print(f"Analyzing job description...")
        job_requirements = self.job_analysis_agent.process(jd_text)

        time.sleep(10)

        # Process each resume in the directory
        resume_files = [f for f in os.listdir(resume_dir) if os.path.isfile(os.path.join(resume_dir, f))]

        for filename in resume_files[:5]:
            file_path = os.path.join(resume_dir, filename)
            print(f"\nProcessing resume: {filename}")

            # Extract text from resume
            resume_text = self.document_agent.process((file_path, filename))

            time.sleep(10)

            if resume_text:
                # Extract candidate profile
                print(f"Extracting candidate profile...")
                candidate_profile = self.resume_analysis_agent.process(resume_text)

                # Calculate embedding similarity
                print(f"Calculating semantic similarity...")
                embedding_similarity_score = self.embedding_similarity_agent.process((jd_text, resume_text))
                
                # Evaluate candidate match
                print(f"Evaluating candidate {candidate_profile['contact_details']['name']}...")
                score = self.matching_agent.process((job_requirements, candidate_profile, resume_text))
                
                # Add embedding similarity score to the overall score
                score['embedding_similarity_score'] = embedding_similarity_score
                
                # Adjust total score to include embedding similarity (optional)
                # This keeps the total at 100 by slightly reducing the weight of other factors
                score['total_score'] = (
                    score['technical_skills_score'] * 0.38 + 
                    score['experience_score'] * 0.28 + 
                    score['education_score'] * 0.14 + 
                    score['additional_qualifications_score'] * 0.14 + 
                    embedding_similarity_score * 0.06
                )

                # Create result object
                result = {
                    "file_name": filename,
                    "contact_details": candidate_profile["contact_details"],
                    "candidate_profile": candidate_profile,
                    "score": score
                }

                results.append(result)

                # Add a small delay to avoid rate limits
                time.sleep(10)
            else:
                print(f"Failed to extract text from {filename}. Skipping this resume.")

        # Sort results by total score
        results.sort(key=lambda x: x["score"]['total_score'], reverse=True)

        # Save results to file
        with open(output_path, 'w') as f:
            json.dump([result for result in results], f, indent=2)

        print(f"\nResults saved to {output_path}")

        # Print summary of results
        print("\n===== CANDIDATE RANKING =====")
        for i, result in enumerate(results, 1):
            name = result["contact_details"]['name']
            score = result["score"]['total_score']
            print(f"{i}. {name}: {score}/100")

        # Send interview invitations to candidates above threshold
        if self.email_communication_agent:
            selected_candidates = [r for r in results if r["score"]['total_score'] >= threshold_score]

            print(f"\nPreparing to send interview invitations to {len(selected_candidates)} candidates who scored {threshold_score}+ out of 100...\n")

            for candidate in selected_candidates:
                response = self.email_communication_agent.process((candidate, calendly_link, email_subject))
                time.sleep(1)

        return results



jd_file_path = "job_description.pdf"
resume_dir = "example_data/"
output_path = "candidate_results.json"



sender_email = "<Your EmailID>"
app_password = "<Your generated app password>"
calendly_link = "<Your Calendly Link>"
email_subject = "HireFive: Next Steps for Your Data Scientist Application"


coordinator = CoordinatorAgent(client)


coordinator.set_email_communication_agent(sender_email, app_password)


threshold_score = 65  # Only send to candidates with 65+ overall score
results = coordinator.process_hiring_workflow(
    jd_file_path=jd_file_path,
    resume_dir=resume_dir,
    output_path=output_path,
    threshold_score=threshold_score,
    calendly_link=calendly_link,
    email_subject=email_subject
)


print(results)