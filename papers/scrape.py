import os  
import requests  
import logging  
from django.conf import settings
from .services import (
    extract_local_file, analyze_with_orchestrator, generate_paper_summary
)
from rest_framework.response import Response
from rest_framework import status
from .models import Paper
# Configure logging  
logging.basicConfig(  
    filename='arxiv_downloader.log',  
    filemode='a',  # Append mode  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  

def download_arxiv_pdf(arxiv_id: str, save_dir: str = 'media/papers') -> bool:  
    """  
    Downloads the PDF of an arXiv article given its arXiv ID.  

    Parameters:  
    - arxiv_id (str): The arXiv article ID (e.g., '2101.00001').  
    - save_dir (str): Directory where the PDF will be saved.  

    Returns:  
    - bool: True if download is successful, False otherwise.  
    """  
    # Clean and validate arXiv ID  
    arxiv_id = arxiv_id.strip()  
    if not arxiv_id:  
        logging.error("Empty arXiv ID provided.")  
        print("Error: Empty arXiv ID provided.")  
        return False  

    # Construct the PDF URL  
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"  
    logging.info(f"Constructed PDF URL: {pdf_url}")  
    print(f"Downloading PDF from: {pdf_url}")  

    # Ensure the save directory exists  
    os.makedirs(save_dir, exist_ok=True)  

    # Define the path to save the PDF  
    pdf_path = os.path.join(save_dir, f"{arxiv_id}.pdf")  

    # Check if the PDF already exists to prevent redundant downloads  
    if os.path.exists(pdf_path):  
        logging.info(f"PDF already exists: {pdf_path}")  
        print(f"PDF already exists at: {pdf_path}")  
        return True  # Considered successful since file exists  

    try:  
        with requests.get(pdf_url, stream=True, timeout=30) as response:  
            response.raise_for_status()  # Raise an error for bad status codes  

            # Write the PDF to the file in chunks  
            with open(pdf_path, 'wb') as f:  
                for chunk in response.iter_content(chunk_size=8192):  
                    if chunk:  # Filter out keep-alive chunks  
                        f.write(chunk)  

        logging.info(f"Successfully downloaded: {pdf_path}")  
        print(f"Successfully downloaded: {pdf_path}")  
        return True  

    except requests.exceptions.HTTPError as http_err:  
        if response.status_code == 404:  
            logging.error(f"PDF not found for arXiv ID: {arxiv_id}")  
            print(f"Error: PDF not found for arXiv ID {arxiv_id}.")  
        else:  
            logging.error(f"HTTP error for {arxiv_id}: {http_err}")  
            print(f"HTTP error occurred for {arxiv_id}: {http_err}")  
    except requests.exceptions.ConnectionError as conn_err:  
        logging.error(f"Connection error for {arxiv_id}: {conn_err}")  
        print(f"Connection error occurred for {arxiv_id}: {conn_err}")  
    except requests.exceptions.Timeout as timeout_err:  
        logging.error(f"Timeout error for {arxiv_id}: {timeout_err}")  
        print(f"Timeout error occurred for {arxiv_id}: {timeout_err}")  
    except requests.exceptions.RequestException as req_err:  
        logging.error(f"Request exception for {arxiv_id}: {req_err}")  
        print(f"Request exception occurred for {arxiv_id}: {req_err}")  
    except Exception as err:  
        logging.error(f"Unexpected error for {arxiv_id}: {err}")  
        print(f"An unexpected error occurred for {arxiv_id}: {err}")  

    # If we reach here, download failed  
    return False  

def process_arxiv_paper(arxiv_id: str, save_dir: str = 'media/papers'):  
    """  
    Downloads a single arXiv PDF, processes it with CheckPaper, and then deletes the PDF.  

    Parameters:  
    - arxiv_id (str): The arXiv article ID.  
    - save_dir (str): Directory where the PDF will be saved.  
    """  
    success = download_arxiv_pdf(arxiv_id, save_dir)  
    if success:  
        # Define the path to the PDF  
        pdf_path = os.path.join(save_dir, f"{arxiv_id}.pdf")  

        # Call the CheckPaper function  
        print(f"Processing PDF with CheckPaper: {pdf_path}")  
        logging.info(f"Processing PDF with CheckPaper: {pdf_path}")  
        
        try:  
            # You need to implement CheckPaper function according to your requirements  
            CheckPaper(pdf_path)  
        except Exception as e:  
            logging.error(f"Error during CheckPaper for {arxiv_id}: {e}")  
            print(f"Error during CheckPaper for {arxiv_id}: {e}")  
            # Decide whether to continue to delete the PDF or not  

        # After processing, delete the PDF  
        if os.path.exists(pdf_path):  
            try:  
                os.remove(pdf_path)  
                logging.info(f"Deleted PDF: {pdf_path}")  
                print(f"Deleted PDF: {pdf_path}")  
            except Exception as e:  
                logging.error(f"Error deleting PDF {pdf_path}: {e}")  
                print(f"Error deleting PDF {pdf_path}: {e}")  
    else:  
        logging.error(f"Failed to download PDF for {arxiv_id}")  
        print(f"Failed to download PDF for {arxiv_id}")  

def CheckPaper(pdf_path: str):  
    """  
    Placeholder for your CheckPaper function.  
    Implement this function based on your requirements.  

    Parameters:  
    - pdf_path (str): Path to the PDF file.  
    """  
    # Implement your processing logic here  
    try:
        new_path = pdf_path[6:]
        full_path = (os.path.join(settings.MEDIA_ROOT, str(new_path)))
        print(f"Checking paper at {full_path}")
        
        # Extract and analyze text
        content_to_analyze = extract_local_file(full_path)
        paper = Paper(title='')
        paper.save()
        document_metadata = {
            "title": paper.title,
            "paper_id": paper.id,
            "file_path": new_path
        }
        summary = generate_paper_summary(content_to_analyze, document_metadata)
        document_metadata['title'] = summary['metadata']['title']
        paper.has_summary = True
        paper.file.name = new_path
        paper.title = summary['metadata']['title']
        paper.save()
        analyze_with_orchestrator(content_to_analyze, document_metadata)
    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return document_metadata