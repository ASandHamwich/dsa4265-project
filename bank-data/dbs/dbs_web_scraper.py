import time
import re
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Change these variables as needed
topic = "digital services"
base_url = "https://www.dbs.com.sg/personal/deposits/atm-branch-services/default.page"
link_class = "sc-1eoexfv ewXlIj"

# Setup logging
date = time.strftime("%d%m%Y_%H%M", time.localtime())
logging.basicConfig(
    filename=f"dbs_scraper_{topic}_{date}.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Set up Chrome WebDriver using WebDriverManager
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")

# Initialize WebDriver with auto-installed ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

def clean_filename(url):
    """Clean the URL to create a valid filename."""
    # Extract the last part of the URL path and remove query parameters
    cleaned_name = url.split("/")[-1].split("?")[0]
    # Replace characters that are not allowed in filenames
    cleaned_name = re.sub(r'[^a-zA-Z0-9_-]', '_', cleaned_name)
    return f"dbs_{cleaned_name}.txt"

def get_links(base_url, link_class):
    """Extract only href links within a specific div class."""
    driver.get(base_url)
    
    time.sleep(3)
    
    # Find all <a> elements inside the specific div class
    links = set()
    
    # Locate div elements with the specified class
    divs = driver.find_elements(By.CSS_SELECTOR, f"div.{link_class} a") # f"div.{link_class} a" or f"h3.{link_class.replace(' ', '.')} a"

    for div in divs:
        href = div.get_attribute("href")
        if href:
            links.add(href)
    
    logging.info(f"Found {len(links)} links.")
    logging.info("Extracted links:")
    for link in links:
        logging.info(link)  # Log each link

    return list(links)

def extract_relevant_text(url):
    """Extract only key information from an account page."""
    try:
        driver.get(url)
        time.sleep(3)  # Allow page to load

        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract only relevant sections
        extracted_text = ""

        # Iterate over all elements (h1, h2, h3, p, ul, table, etc.)
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'table']):
            if element.name in ['h1', 'h2', 'h3']:
                extracted_text += f"\n\n<{element.name}>{element.text.strip()}</{element.name}>\n"
            elif element.name == 'p':
                extracted_text += f"\n<p>{element.text.strip()}</p>\n"
            elif element.name == 'ul':
                extracted_text += "\n<ul>\n"
                for li in element.find_all("li"):
                    extracted_text += f"  <li>{li.text.strip()}</li>\n"
                extracted_text += "</ul>\n"
            elif element.name == 'table':
                extracted_text += "\n<table>\n"
                for row in element.find_all("tr"):
                    extracted_text += "  <tr>\n"
                    for cell in row.find_all(["th", "td"]):
                        tag = "th" if cell.name == "th" else "td"
                        extracted_text += f"    <{tag}>{cell.text.strip()}</{tag}>\n"
                    extracted_text += "  </tr>\n"
                extracted_text += "</table>\n"

        # Save extracted content to a text file
        filename = clean_filename(url)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        logging.info(f"Saved cleaned text for: {url}")

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")

# Step 1: Get links (from the new base URL and link class)
# links = get_links(base_url, link_class)
links = [
    "https://www.dbs.com.sg/i-bank/deposits/atm-branch-services/dbs-pitstop",
    "https://www.dbs.com.sg/i-bank/deposits/bank-with-ease/safe_deposit_box",
    "https://www.dbs.com.sg/i-bank/deposits/bank-with-ease/self-service-banking",
    "https://www.dbs.com.sg/i-bank/deposits/bank-with-ease/sms-q",
    "https://www.dbs.com.sg/i-bank/deposits/bank-with-ease/talking-atm",
    "https://www.dbs.com.sg/i-bank/deposits/bank-with-ease/video-teller-machine"
]

# Step 2: Scrape relevant text from each link
for link in links:
    extract_relevant_text(link)

# Close WebDriver
driver.quit()
logging.info("Scraping completed successfully.")



