import requests
from bs4 import BeautifulSoup

# Extract text content from a webpage
def extract_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

# Chunk text into smaller parts
def chunk_text(text, chunk_size=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
