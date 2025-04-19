# BERT

This model's primary role in the product is to classify transactional data into specified categories for further use further down the pipeline. 

Setup:
1. Install Python 3.11 (recommended: use `venv`)
2. Install Python dependencies from `requirements.txt`.
3. Install Brew packages with `xargs brew install < brew-packages.txt`.

Usage:

To use in overall pipeline, export `process_statement_pipeline` from `bert_pipeline.py`. 
Else, run `process_statement_pipeline` as a script, and provide your bank statement `.pdf` as your argument.

FUNCTION `process_statement_pipeline(pdf_path)`:

Input: 
- `.pdf` file path of bank statement

Output: 
- `financial_data.json`: A summary of transactions made, determined from the bank statement uploaded.
- `transactions.json`: A list of JSON objects, showing all transactions made. 
