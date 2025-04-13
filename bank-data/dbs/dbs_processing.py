import os
import csv
import re
from bs4 import BeautifulSoup

# --- Step 1: Extract sections from HTML ---
def extract_sections(soup):
    sections = []
    current_hierarchy = []  # Stack for headers
    current_content = []

    def add_section():
        if current_hierarchy and current_content:
            title = current_hierarchy[0] if len(current_hierarchy) > 0 else ""
            subtitle = current_hierarchy[1] if len(current_hierarchy) > 1 else ""
            heading = current_hierarchy[2] if len(current_hierarchy) > 2 else ""
            info = "\n".join(current_content)
            sections.append((title, subtitle, heading, info))
            current_content.clear()

    for element in soup.find_all():
        if element.name in ["h1", "h2", "h3"]:
            add_section()
            level = int(element.name[1]) - 1
            if len(current_hierarchy) > level:
                current_hierarchy = current_hierarchy[:level]
            current_hierarchy.append(element.get_text(strip=True))
        elif element.name == "table":
            table_text = []
            for row in element.find_all("tr"):
                cols = [col.get_text(" ", strip=True) for col in row.find_all(["td", "th"])]
                table_text.append("\t".join(cols))
            if table_text:
                current_content.append("Table:\n" + "\n".join(table_text))
        elif element.name in ["ul", "ol"]:
            list_items = [li.get_text(" ", strip=True) for li in element.find_all("li")]
            if list_items:
                current_content.append("List:\n- " + "\n- ".join(list_items))
        else:
            text = element.get_text(" ", strip=True)
            if text:
                current_content.append(text)

    add_section()
    return sections

# --- Step 2: Read HTML content from file ---
def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    return extract_sections(soup)

# --- Step 3: Process all .txt files and save CSV ---
def process_text_files(directory):
    data = []

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(folder_path, file)
                    url = re.sub(r"^dbs_", "", file.replace(".txt", ""))
                    sections = extract_text_from_html(file_path)

                    for title, subtitle, heading, info in sections:
                        data.append([folder, url, title, subtitle, heading, info])

    csv_path = os.path.join(directory, "output.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Folder", "URL", "Title", "Subtitle", "Heading", "Information"])
        writer.writerows(data)

    print(f"CSV file saved at: {csv_path}")
    return csv_path  # Return path for next step

# --- Step 4: Clean known encoding issues in the final CSV ---
def clean_csv_encoding_issues(csv_path):
    replacements = {
        "â€™": "'",
        "â€“": "–",
        "â€œ": "\"",
        "â€": "\"",
        "â€˜": "'",
        "â€": "\"",  # fallback
    }

    with open(csv_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    cleaned_lines = []
    for line in lines:
        for wrong, right in replacements.items():
            line = line.replace(wrong, right)
        cleaned_lines.append(line)

    with open(csv_path, "w", encoding="utf-8", newline="") as outfile:
        outfile.writelines(cleaned_lines)

    print(f"Cleaned CSV written to: {csv_path}")

# --- Main entry point ---
if __name__ == "__main__":
    output_csv_path = process_text_files(os.getcwd())
    clean_csv_encoding_issues(output_csv_path)


