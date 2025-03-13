import re  
import requests  
from bs4 import BeautifulSoup
import os

def sanitize_filename(filename):  
    """  
    Sanitize the filename by replacing invalid characters with underscores.  
    """  
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)  

def is_valid_arxiv_url(url):  
    # Regular expression to match arXiv URLs  
    pattern = r'^https?://arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(\.pdf)?$'  
    if not re.match(pattern, url):  
        return False  

    try:  
        response = requests.head(url, allow_redirects=True)  
        return response.status_code == 200  
    except requests.RequestException:  
        return False  
    
def is_valid_rxiv_url(url):  
    pattern = (  
        r'^https?://www\.(biorxiv|medrxiv)\.org/content/'  
        r'(10\.1101/[\d\.]+)v\d+$'  
    )  
    if not re.match(pattern, url):  
        return False  

    try:  
        response = requests.head(url, allow_redirects=True)  
        return response.status_code == 200  
    except requests.RequestException:  
        return False  
    
def is_valid_openalex_url(url):  
    pattern = r'^https?://openalex\.org/[A-Z]\d+$'  
    if not re.match(pattern, url):  
        return False  

    try:  
        response = requests.head(url, allow_redirects=True)  
        return response.status_code == 200  
    except requests.RequestException:  
        return False  
    
def is_valid_pubmed_url(url):  
    pattern = r'^https?://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?$'  
    if not re.match(pattern, url):  
        return False  

    try:  
        response = requests.head(url, allow_redirects=True)  
        return response.status_code == 200  
    except requests.RequestException:  
        return False  

def ensure_directory_exists(directory):  
    """  
    Ensure that the specified directory exists; if not, create it.  
    """  
    if not os.path.exists(directory):  
        os.makedirs(directory)  

def download_arxiv_pdf(url):  
    """  
    Download PDF from arXiv and save it as os.path.join('media/papers', {id}.pdf).  
    Returns the save path if successful, or False if not.  
    """  
    # Regular expression to match arXiv URLs and extract the ID  
    pattern = r'^https?://arxiv\.org/(abs|pdf)/(?P<id>\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(\.pdf)?$'  
    match = re.match(pattern, url)  
    if not match:  
        print('Invalid arXiv URL')  
        return False  
    arxiv_id = match.group('id')  

    # Construct the PDF URL  
    pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'  

    # Define the save path  
    directory = os.path.join('media', 'papers')  
    ensure_directory_exists(directory)  
    save_path = os.path.join(directory, f'{arxiv_id}.pdf')  # Save as {id}.pdf  

    try:  
        response = requests.get(pdf_url)  
        if response.status_code == 200:  
            # Save the PDF file  
            with open(save_path, 'wb') as f:  
                f.write(response.content)  
            print(f'Downloaded arXiv PDF to {save_path}')  
            return save_path  
        else:  
            print('Failed to download the arXiv PDF.')  
            return False  
    except requests.RequestException as e:  
        print(f'An error occurred while downloading arXiv PDF: {e}')  
        return False  

def download_rxiv_pdf(url):  
    """  
    Download PDF from bioRxiv or medRxiv and save it as os.path.join('media/papers', {id}.pdf).  
    Returns the save path if successful, or False if not.  
    """  
    pattern = (  
        r'^https?://www\.(?P<domain>biorxiv|medrxiv)\.org/content/'  
        r'(?P<doi>10\.1101/[\d\.]+)v(?P<version>\d+)$'  
    )  
    match = re.match(pattern, url)  
    if not match:  
        print('Invalid bioRxiv/medRxiv URL')  
        return False  

    domain = match.group('domain')  
    doi = match.group('doi')  
    version = match.group('version')  
    # Construct an ID suitable for filenames  
    id_str = f'{domain}_{doi}v{version}'  
    id_str = sanitize_filename(id_str)  

    # Define the save path  
    directory = os.path.join('media', 'papers')  
    ensure_directory_exists(directory)  
    save_path = os.path.join(directory, f'{id_str}.pdf')  

    try:  
        # Get the content page  
        response = requests.get(url)  
        if response.status_code != 200:  
            print('Failed to retrieve the content page.')  
            return False  

        # Parse the HTML to find the PDF link  
        soup = BeautifulSoup(response.content, 'html.parser')  
        pdf_link = soup.find('a', {'class': 'article-dl-pdf-link'})  
        if not pdf_link:  
            print('PDF link not found on the page.')  
            return False  

        pdf_url = pdf_link['href']  
        if not pdf_url.startswith('http'):  
            pdf_url = 'https://www.' + domain + '.org' + pdf_url  

        # Download the PDF  
        pdf_response = requests.get(pdf_url)  
        if pdf_response.status_code == 200:  
            # Save the PDF file  
            with open(save_path, 'wb') as f:  
                f.write(pdf_response.content)  
            print(f'Downloaded {domain} PDF to {save_path}')  
            return save_path  
        else:  
            print(f'Failed to download the {domain} PDF.')  
            return False  
    except requests.RequestException as e:  
        print(f'An error occurred while downloading {domain} PDF: {e}')  
        return False  

def download_openalex_pdf(url):  
    """  
    Download PDF from OpenAlex if available and save it as os.path.join('media/papers', {id}.pdf).  
    Returns the save path if successful, or False if not.  
    """  
    pattern = r'^https?://openalex\.org/(?P<id>[A-Z]\d+)$'  
    match = re.match(pattern, url)  
    if not match:  
        print('Invalid OpenAlex URL')  
        return False  
    openalex_id = match.group('id')  

    # Define the save path  
    directory = os.path.join('media', 'papers')  
    ensure_directory_exists(directory)  
    save_path = os.path.join(directory, f'{openalex_id}.pdf')  

    # OpenAlex API endpoint  
    api_url = f'https://api.openalex.org/works/{openalex_id}'  

    try:  
        response = requests.get(api_url)  
        if response.status_code != 200:  
            print('Failed to retrieve OpenAlex data.')  
            return False  

        data = response.json()  
        # Check for open access PDF URL  
        pdf_url = None  

        # Check for 'best_oa_location' field  
        best_oa_location = data.get('best_oa_location', {})  
        pdf_url = best_oa_location.get('url_for_pdf')  

        if not pdf_url:  
            # Check 'alternate_host_venues'  
            host_venues = data.get('alternate_host_venues', [])  
            for host in host_venues:  
                if host.get('url'):  
                    pdf_url = host.get('url')  
                    if pdf_url.endswith('.pdf'):  
                        break  
            else:  
                pdf_url = None  

        if not pdf_url:  
            print('No direct PDF URL available in OpenAlex data.')  
            return False  

        pdf_url = pdf_url.replace('\\', '')  # Remove any backslashes from URL  

        # Download the PDF  
        pdf_response = requests.get(pdf_url)  
        if pdf_response.status_code == 200:  
            # Save the PDF file  
            with open(save_path, 'wb') as f:  
                f.write(pdf_response.content)  
            print(f'Downloaded OpenAlex PDF to {save_path}')  
            return save_path  
        else:  
            print('Failed to download the OpenAlex PDF from the provided URL.')  
            return False  
    except requests.RequestException as e:  
        print(f'An error occurred while downloading OpenAlex PDF: {e}')  
        return False  
    except ValueError as e:  
        print(f'An error occurred while parsing OpenAlex data: {e}')  
        return False  

def download_pubmed_pdf(url):  
    """  
    Download PDF from PubMed Central if available and save it as os.path.join('media/papers', {id}.pdf).  
    Returns the save path if successful, or False if not.  
    """  
    pattern = r'^https?://pubmed\.ncbi\.nlm\.nih\.gov/(?P<pmid>\d+)/?$'  
    match = re.match(pattern, url)  
    if not match:  
        print('Invalid PubMed URL')  
        return False  
    pmid = match.group('pmid')  

    # Define the save path  
    directory = os.path.join('media', 'papers')  
    ensure_directory_exists(directory)  
    save_path = os.path.join(directory, f'{pmid}.pdf')  

    try:  
        # Retrieve the PubMed page  
        response = requests.get(url)  
        if response.status_code != 200:  
            print('Failed to retrieve the PubMed page.')  
            return False  

        # Parse the page to find full-text links  
        soup = BeautifulSoup(response.content, 'html.parser')  
        # Look for links to PMC articles  
        pmc_link = soup.find('a', {'ref': 'PMC'})  
        if pmc_link:  
            # Extract PMC URL  
            pmc_url = pmc_link['href']  
            if not pmc_url.startswith('http'):  
                pmc_url = 'https://pubmed.ncbi.nlm.nih.gov' + pmc_url  
            return download_pmc_pdf(pmc_url, save_path)  
        else:  
            print('PDF not available directly via PubMed.')  
            return False  
    except requests.RequestException as e:  
        print(f'An error occurred while processing PubMed URL: {e}')  
        return False  

def download_pmc_pdf(url, save_path):  
    """  
    Download PDF from PubMed Central and save it with the specified filename.  
    Returns the save path if successful, or False if not.  
    """  
    try:  
        # Retrieve the PMC page  
        response = requests.get(url)  
        if response.status_code != 200:  
            print('Failed to retrieve the PMC page.')  
            return False  

        # Parse the page to find the PDF link  
        soup = BeautifulSoup(response.content, 'html.parser')  
        pdf_link = soup.find('a', {'class': 'pdf-link'}, href=True)  
        if not pdf_link:  
            print('PDF link not found on PMC page.')  
            return False  
        pdf_url = pdf_link['href']  
        if not pdf_url.startswith('http'):  
            pdf_url = 'https://www.ncbi.nlm.nih.gov' + pdf_url  

        # Download the PDF  
        pdf_response = requests.get(pdf_url)  
        if pdf_response.status_code == 200:  
            # Ensure the directory exists (it should already exist, but just in case)  
            directory = os.path.dirname(save_path)  
            ensure_directory_exists(directory)  
            # Save the PDF file  
            with open(save_path, 'wb') as f:  
                f.write(pdf_response.content)  
            print(f'Downloaded PMC PDF to {save_path}')  
            return save_path  
        else:  
            print('Failed to download the PMC PDF.')  
            return False  
    except requests.RequestException as e:  
        print(f'An error occurred while downloading PMC PDF: {e}')  
        return False  

def download_paper(url):  
    """  
    Determine the repository based on the URL and download the PDF accordingly.  
    Returns the save path if successful, or False if not.  
    """  
    if 'arxiv.org' in url:  
        return download_arxiv_pdf(url)  
    elif 'biorxiv.org' in url or 'medrxiv.org' in url:  
        return download_rxiv_pdf(url)  
    elif 'openalex.org' in url:  
        return download_openalex_pdf(url)  
    elif 'pubmed.ncbi.nlm.nih.gov' in url:  
        return download_pubmed_pdf(url)  
    else:  
        print('Unsupported URL or domain.')  
        return False  
    
def is_valid_paper_url(url):  
    repositories = [  
        ('arXiv', is_valid_arxiv_url),  
        ('bioRxiv/medRxiv', is_valid_rxiv_url),  
        ('OpenAlex', is_valid_openalex_url),  
        ('PubMed', is_valid_pubmed_url),  
    ]  
    for name, validator in repositories:  
        if validator(url):  
            print(f'Valid {name} URL')  
            return True  
    print('Invalid paper URL')  
    return False  