# BERT

To use in pipeline: export `process_statement_pipeline` from `bert_pipeline.py`. 

`process_statement_pipeline`:
Input: `.pdf` file path of bank statement
Output: 
- `financial_data.json`: A summary of transactions made, determined from the bank statement uploaded.
- `transactions.json`: A list of JSON objects, showing all transactions made. 
