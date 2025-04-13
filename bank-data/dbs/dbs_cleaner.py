import os
import re

# Define the parent directory containing the six folders
parent_dir = os.getcwd()  # Change this if needed

# Regex patterns
h1_pattern = re.compile(r"<h1>")  # Find the first <h1>
help_support_pattern = re.compile(r"<p>Help & Support Portal</p>.*", re.DOTALL)  # Remove everything from this point onward
ul_chunk_pattern = re.compile(r"<ul>\s*<li>Created with Sketch\.At a Glance</li>.*?</ul>", re.DOTALL)
empty_tags_pattern = re.compile(r"<(h1|h2|h3|p)>\s*</\1>")  # Remove empty tags like <h1></h1>, <h2></h2>, etc.
empty_lines_pattern = re.compile(r"\n\s*\n")  # Remove empty lines

def clean_text(text):
    """Cleans text based on the defined rules."""
    # Keep content from first <h1> onward
    h1_match = h1_pattern.search(text)
    if h1_match:
        text = text[h1_match.start():]  # Trim everything before <h1>

    # Remove everything from <p>Help & Support Portal</p> onward
    text = help_support_pattern.sub("", text)

    # Remove the specific <ul> chunk
    text = ul_chunk_pattern.sub("", text)

    # Remove empty tags like <h1></h1>, <h2></h2>, <h3></h3>, <p></p>
    text = empty_tags_pattern.sub("", text)

    # Remove empty lines
    text = empty_lines_pattern.sub("\n", text)

    return text.strip()  # Remove any leading/trailing newlines

def process_files():
    """Processes all text files in the six subdirectories."""
    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):  # Only process .txt files
                    file_path = os.path.join(folder_path, filename)

                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()

                    cleaned_content = clean_text(content)

                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(cleaned_content)

                    print(f"Cleaned: {file_path}")

# Run the cleaning process
process_files()
print("Cleaning completed for all text files.")
