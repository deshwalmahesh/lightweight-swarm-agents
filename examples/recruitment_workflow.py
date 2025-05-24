import os
import time
import json
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import fitz  # PyMuPDF


load_dotenv() 

ai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])



def fetch_job_posting(url, output_path = "job_description.pdf"):
    """
    Download job posting document from a given URL.
    """
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {output_path}")

def retrieve_candidate_documents(url, local_dir="example_data", fetch_limit = 3):
    """
    Download candidate resume documents from the given URL.
    """

    response = requests.get(url)

    if response.status_code != 200:
        print("Failed to retrieve folder contents:", response.text)
        return

    data = response.json()
    os.makedirs(local_dir, exist_ok=True)

    print(f"{min(fetch_limit, len(data))} files available for download:")
    for file in data[:fetch_limit]:
        file_name = file["name"]
        if os.path.exists(os.path.join(local_dir, file_name)): continue
        download_url = file["download_url"]

        r = requests.get(download_url)
        with open(os.path.join(local_dir, file_name), "wb") as f:
            f.write(r.content)
        print(f"Downloaded {file_name}")


source_url = "https://raw.githubusercontent.com/mistralai/cookbook/main/mistral/agents/recruitment_agent/job_description.pdf"
destination_path = "job_description.pdf"

fetch_job_posting(source_url, destination_path)


retrieve_candidate_documents(
    url = "https://api.github.com/repos/mistralai/cookbook/contents/mistral/agents/recruitment_agent/resumes",
    local_dir="example_data"
)


class Competency(BaseModel):
    name: str = Field(description="Name of the skill or technology")
    proficiency: Optional[str] = Field(description="Proficiency level (beginner, intermediate, advanced)")
    duration: Optional[float] = Field(description="Years of experience with this skill")

class AcademicBackground(BaseModel):
    credential: str = Field(description="Type of degree or certification obtained")
    discipline: str = Field(description="Field of study or specialization")
    school: str = Field(description="Name of educational institution")
    graduation_year: Optional[int] = Field(description="Year when degree was completed")
    academic_score: Optional[float] = Field(description="Grade Point Average, typically on 1-10 CGPA or 0-100% score. Normalize it to 0-10 scale")

class WorkHistory(BaseModel):
    position: str = Field(description="Job title or position held")
    employer: str = Field(description="Name of employer or organization")
    tenure_years: float = Field(description="Duration of employment in years")
    applied_skills: List[str] = Field(description="Skills utilized in this role")
    accomplishments: List[str] = Field(description="Key accomplishments or responsibilities")
    job_relevance: Optional[float] = Field(description="Relevance to current job opening (0-10 scale)")

class ApplicantInfo(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Primary email address for contact")
    phone: Optional[str] = Field(description="Phone number with country code if applicable")
    location: Optional[str] = Field(description="Current city and country/state")
    linkedin: Optional[str] = Field(description="LinkedIn profile URL")
    portfolio: Optional[str] = Field(description="Personal or portfolio website URL")

class PositionRequirements(BaseModel):
    essential_competencies: List[Competency] = Field(description="Skills that are mandatory for the position")
    desired_competencies: List[Competency] = Field(description="Skills that are desired but not required")
    min_experience_years: float = Field(description="Minimum years of experience required")
    required_education: List[AcademicBackground] = Field(description="Mandatory educational qualifications")
    preferred_sectors: List[str] = Field(description="Industry domains or sectors preferred for experience")

class ApplicantProfile(BaseModel):
    personal_info: ApplicantInfo = Field(description="Candidate's personal and contact information")
    competencies: List[Competency] = Field(description="Technical and soft skills possessed by the candidate")
    education: List[AcademicBackground] = Field(description="Educational background and qualifications")
    work_experience: List[WorkHistory] = Field(description="Professional work history and experience")

class CompetencyMatch(BaseModel):
    competency_name: str = Field(description="Name of the skill being evaluated")
    is_present: bool = Field(description="Whether the candidate possesses this skill")
    match_quality: float = Field(description="How well the candidate's skill matches the requirement (0-10 scale)")
    assessment_confidence: float = Field(description="Confidence in the skill evaluation (0-1 scale)")
    evaluation_notes: str = Field(description="Additional context about the skill match assessment")

class ApplicantEvaluation(BaseModel):
    technical_aptitude_score: float = Field(description="Assessment of technical capabilities (0-40 points)")
    professional_experience_score: float = Field(description="Evaluation of relevant work experience (0-30 points)")
    academic_qualification_score: float = Field(description="Assessment of educational qualifications (0-15 points)")
    supplementary_qualification_score: float = Field(description="Evaluation of other relevant qualifications (0-15 points)")
    semantic_similarity_score: Optional[float] = Field(description="Semantic similarity between resume and job description (0-10)")
    overall_rating: float = Field(description="Overall candidate evaluation score (0-100 scale)")
    competency_assessments: List[Dict] = Field(description="Detailed breakdown of skill match assessments")
    key_strengths: List[str] = Field(description="Key candidate strengths relative to the position")
    qualification_gaps: List[str] = Field(description="Areas where candidate does not meet requirements")
    evaluation_confidence: float = Field(description="Overall confidence in the evaluation accuracy (0-1 scale)")
    additional_observations: str = Field(description="Supplementary observations about the candidate fit")

class ApplicantResult(BaseModel):
    document_name: str = Field(description="Name of the source resume file")
    personal_info: ApplicantInfo = Field(description="Candidate's contact information")
    profile_data: ApplicantProfile = Field(description="Complete extracted candidate profile")
    evaluation: ApplicantEvaluation = Field(description="Detailed evaluation scores and assessment")



class Processor:
    def __init__(self, identifier: str, client: OpenAI):
        self.identifier = identifier
        self.client = client

    def process(self, input_data):
        """Base process method - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement process method")

    def relay(self, target_processor, input_data):
        """Send data to another processor"""
        return target_processor.process(input_data)



class DocumentProcessor(Processor):
    def __init__(self, client: OpenAI):
        super().__init__("DocumentProcessor", client)

    def process(self, file_info):
        """Process document extraction request"""
        file_path, file_name = file_info
        return self.parse_document_content(file_path, file_name)

    def parse_document_content(self, file_path: str, file_name: str) -> str:
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



class JobSpecificationProcessor(Processor):
    def __init__(self, client: OpenAI):
        super().__init__("JobSpecificationProcessor", client)

    def process(self, job_description_text):
        """Process job description text"""
        return self.extract_position_requirements(job_description_text)

    def extract_position_requirements(self, job_description_text: str) -> PositionRequirements:
        """Extract structured job requirements from a job description"""
        prompt = f"""
        Extract the key job requirements from the following job description.
        Focus on required skills, preferred skills, experience requirements, and education requirements.
        
        Provide your response as a JSON object with the following structure:
        {{
            "essential_competencies": [{{
                "name": "skill name",
                "proficiency": "proficiency level",
                "duration": years of experience
            }}],
            "desired_competencies": [...],
            "min_experience_years": number,
            "required_education": [...],
            "preferred_sectors": [...]
        }}
        
        Job Description:
        {job_description_text}
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



class ProfileExtractionProcessor(Processor):
    def __init__(self, client: OpenAI):
        super().__init__("ProfileExtractionProcessor", client)

    def process(self, resume_text):
        """Process resume text"""
        return self.extract_applicant_profile(resume_text)

    def extract_applicant_profile(self, resume_text: str) -> ApplicantProfile:
        """Extract structured candidate information from resume text"""
        prompt = f"""
        Extract the candidate's contact details, skills, education, and experience from the following resume.
        Be thorough and include all relevant information.
        
        Provide your response as a JSON object with the following structure:
        {{
            "personal_info": {{
                "name": "candidate name",
                "email": "email address",
                "phone": "phone number",
                "location": "location",
                "linkedin": "linkedin profile",
                "portfolio": "website url"
            }},
            "competencies": [{{
                "name": "skill name",
                "proficiency": "proficiency level",
                "duration": years of experience
            }}],
            "education": [...],
            "work_experience": [...]
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



class SemanticMatchingProcessor(Processor):
    def __init__(self, client: OpenAI):
        super().__init__("SemanticMatchingProcessor", client)
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
        job_text, resume_text = data
        return self.measure_content_similarity(job_text, resume_text)
    
    def measure_content_similarity(self, job_text: str, resume_text: str) -> float:
        """Calculate the semantic similarity between job description and resume"""
        if not self.model:
            return 5.0
        
        try:
            # Truncate texts if they're too long
            max_length = 4096
            job_text = job_text[:max_length] if len(job_text) > max_length else job_text
            resume_text = resume_text[:max_length] if len(resume_text) > max_length else resume_text
            
            # Encode job description as query
            job_embeddings = self.model.encode(
                [job_text],
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
                queries_embeddings=job_embeddings,
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
            return "Excellent alignment - resume is highly relevant to the job requirements"
        elif score >= 7.5:
            return "Strong alignment - resume matches most of the job requirements"
        elif score >= 6.0:
            return "Good alignment - resume matches many of the job requirements"
        elif score >= 4.0:
            return "Moderate alignment - resume matches some of the job requirements"
        elif score >= 2.5:
            return "Limited alignment - resume matches few of the job requirements"
        else:
            return "Poor alignment - resume does not match the job requirements well"


class CandidateEvaluationProcessor(Processor):
    def __init__(self, client: OpenAI):
        super().__init__("CandidateEvaluationProcessor", client)

    def process(self, data):
        """Process job requirements and candidate profile to generate score"""
        position_requirements, applicant_profile, resume_text = data
        return self.assess_candidate_fit(position_requirements, applicant_profile, resume_text)

    def assess_candidate_fit(self, position_requirements: PositionRequirements, applicant_profile: ApplicantProfile, resume_text: str) -> ApplicantEvaluation:
        """Evaluate how well a candidate matches the job requirements"""
        # Convert to JSON for inclusion in the prompt
        job_req_json = json.dumps(position_requirements, indent=2)
        candidate_json = json.dumps(applicant_profile, indent=2)

        prompt = f"""
        Evaluate how well the candidate matches the job requirements.

        Job Requirements:
        {job_req_json}

        Candidate Profile:
        {candidate_json}

        Provide a detailed scoring breakdown, highlighting strengths and gaps.
        Assess the quality and relevance of the candidate's experience, not just keyword matches.
        Include confidence levels for your assessment.

        IMPORTANT: ALL scores must be on a scale of 0-10, where 0 is the lowest possible score and 10 is the highest.
        
        Evaluate the following categories, all on a 0-10 scale:
        - Technical aptitude (0-10)
        - Professional experience (0-10)
        - Academic qualifications (0-10)
        - Supplementary qualifications (0-10)
        
        Return your evaluation as a JSON object with the following structure:
        {{
            "technical_aptitude_score": numeric value between 0-10,
            "professional_experience_score": numeric value between 0-10,
            "academic_qualification_score": numeric value between 0-10,
            "supplementary_qualification_score": numeric value between 0-10,
            "key_strengths": [list of strings],
            "qualification_gaps": [list of strings],
            "recommendation": string,
            "evaluation_confidence": numeric value between 0-1,
            "additional_observations": string
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



class CommunicationProcessor(Processor):
    def __init__(self, client: OpenAI, sender_email: str, app_password: str):
        super().__init__("CommunicationProcessor", client)
        self.sender_email = sender_email
        self.app_password = app_password

    def process(self, data):
        """Process email sending request"""
        applicant, scheduling_link, email_subject = data
        return self.deliver_interview_invitation(applicant, scheduling_link, email_subject)

    def deliver_interview_invitation(self, applicant, scheduling_link: str, email_subject: str):
        """Generate and send personalized email to candidate"""
        name = applicant["personal_info"]['name']
        email = applicant["personal_info"]['email']

        # Create email HTML content
        html_content = f"""\
        <html>
          <body>
            <p>Dear {name},</p>
            <p>I'm the Talent Acquisition Specialist at TechTalent Solutions. We appreciate your interest in the Machine Learning Engineer position.</p>
            <p>After reviewing your qualifications, we'd like to invite you to a video interview to further discuss your background and how it aligns with our team's needs.</p>
            <p>You can reserve your preferred interview time using this <a href="{scheduling_link}">scheduling link</a>.</p>
            <p>If you have any questions before the interview, please don't hesitate to reach out.</p>
            <p>Warm regards,<br>
            Talent Acquisition Team<br>
            TechTalent Solutions</p>
          </body>
        </html>
        """

        if self.app_password:
            try:
                self.transmit_email(email, email_subject, html_content)
                return f"Interview invitation email delivered to {name} at {email}"
            except Exception as e:
                return f"Communication failed with {name} ({email}): {str(e)}"
        else:
            return f"Interview invitation would be sent to {name} at {email} with subject: {email_subject}"

    def transmit_email(self, recipient_email, subject, html_content):
        """Send an email using Gmail SMTP"""
        # Create message container
        message = MIMEMultipart('alternative')
        message['From'] = self.sender_email
        message['To'] = recipient_email
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
            server.sendmail(self.sender_email, recipient_email, text)

        finally:
            server.quit()  # Close the connection



class RecruitmentOrchestrator(Processor):
    def __init__(self, client: OpenAI):
        super().__init__("RecruitmentOrchestrator", client)
        self.document_processor = DocumentProcessor(client)
        self.job_specification_processor = JobSpecificationProcessor(client)
        self.profile_extraction_processor = ProfileExtractionProcessor(client)
        self.candidate_evaluation_processor = CandidateEvaluationProcessor(client)
        self.semantic_matching_processor = SemanticMatchingProcessor(client)
        self.communication_processor = None  # Will be initialized later with email credentials

    def configure_communication_processor(self, sender_email: str, app_password: str):
        """Initialize communication processor with email credentials"""
        self.communication_processor = CommunicationProcessor(self.client, sender_email, app_password)

    def execute_recruitment_pipeline(self, jd_file_path: str, resume_dir: str, output_path: str,
                               scheduling_link: str, email_subject: str):
        """
        Coordinate the entire hiring workflow from document processing to interview scheduling
        """
        results = []

        # Process job description
        print(f"Extracting text from job description...")
        job_description_text = self.document_processor.process((jd_file_path, os.path.basename(jd_file_path)))

        if not job_description_text:
            print("Failed to extract text from job description. Aborting.")
            return results

        # Extract job requirements
        print(f"Analyzing job description...")
        position_requirements = self.job_specification_processor.process(job_description_text)

        time.sleep(10)

        # Process each resume in the directory
        resume_files = [f for f in os.listdir(resume_dir) if os.path.isfile(os.path.join(resume_dir, f))]

        for filename in resume_files[:5]:
            file_path = os.path.join(resume_dir, filename)
            print(f"\nProcessing resume: {filename}")

            # Extract text from resume
            resume_text = self.document_processor.process((file_path, filename))

            time.sleep(10)

            if resume_text:
                # Extract applicant profile
                print(f"Extracting applicant profile...")
                applicant_profile = self.profile_extraction_processor.process(resume_text)

                # Calculate semantic similarity
                print(f"Calculating semantic similarity...")
                semantic_similarity_score = self.semantic_matching_processor.process((job_description_text, resume_text))
                
                # Evaluate applicant match
                print(f"Evaluating applicant {applicant_profile['personal_info']['name']}...")
                evaluation = self.candidate_evaluation_processor.process((position_requirements, applicant_profile, resume_text))
                
                # Add semantic similarity score to the overall evaluation (ensure it's on 0-10 scale)
                evaluation['semantic_similarity_score'] = semantic_similarity_score
  
                
                # Define weights and scores
                weights = np.array([4, 3, 1.5, 1.5, 1])
                scores = np.array([
                    evaluation['technical_aptitude_score'],
                    evaluation['professional_experience_score'],
                    evaluation['academic_qualification_score'],
                    evaluation['supplementary_qualification_score'],
                    semantic_similarity_score
                ])
                
                # Calculate weighted average
                evaluation['average_score'] = np.average(scores, weights=weights)

                # Create result object
                result = {
                    "document_name": filename,
                    "personal_info": applicant_profile["personal_info"],
                    "profile_data": applicant_profile,
                    "evaluation": evaluation
                }

                results.append(result)

                # Add a small delay to avoid rate limits
                time.sleep(10)
            else:
                print(f"Failed to extract text from {filename}. Skipping this resume.")

        # Sort results by average score
        results.sort(key=lambda x: x["evaluation"]['average_score'], reverse=True)

        # Save results to file
        with open(output_path, 'w') as f:
            json.dump([result for result in results], f, indent=2)

        print(f"\nResults saved to {output_path}")

        # Print summary of results
        print("\n===== APPLICANT RANKING =====")
        for i, result in enumerate(results, 1):
            name = result["personal_info"]['name']
            score = result["evaluation"]['average_score']
            print(f"{i}. {name}: {score:.2f}/10")

        # Send interview invitations to applicants with average score above 7.5 (0.75 on a 0-1 scale)
        if self.communication_processor:
            selected_applicants = [r for r in results if r["evaluation"]['average_score'] >= 7.5]

            print(f"\nPreparing to send interview invitations to {len(selected_applicants)} applicants who scored 7.5+ out of 10...\n")

            for applicant in selected_applicants:
                response = self.communication_processor.process((applicant, scheduling_link, email_subject))
                time.sleep(1)

        return results

if __name__ == "__main__":
    job_posting_path = "job_description.pdf"
    applicant_docs_dir = "example_data/"
    results_output_path = "applicant_results.json"

    sender_email = "<Your EmailID>"
    app_password = "<Your generated app password>"
    scheduling_link = "<Your Interview Scheduling Link>"
    email_subject = "TechTalent Solutions: Interview Invitation for Machine Learning Engineer Role"


    recruitment_system = RecruitmentOrchestrator(ai_client)


    recruitment_system.configure_communication_processor(sender_email, app_password)


    # we're using a fixed threshold of 7.5/10
    results = recruitment_system.execute_recruitment_pipeline(
        jd_file_path=job_posting_path,
        resume_dir=applicant_docs_dir,
        output_path=results_output_path,
        scheduling_link=scheduling_link,
        email_subject=email_subject
    )


    print(results)