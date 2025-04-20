# LLM Agent

FinTailorAI is a command-line financial advisor that uses OpenAI’s GPT models and Singapore banking data to provide personalized financial advice. It supports loading user financial and transaction data from JSON files and uses built-in or custom banking product data to recommend relevant financial solutions. 

To use, install Python 3.8+, run pip install openai python-dotenv, and set your OpenAI API key in a .env file. Start the advisor with python main.py and optionally provide paths to financial, transaction, or banking JSON files. You can also call generate_financial_advice() in your own Python scripts for integration. Use commands like help, status, and exit in interactive mode. 

The project retrieves default data from relative paths: ../bert/sample_output/insights.json for financial data, ../bert/sample_output/transactions.json for transaction history, and ../rag/multiqa_relevant_banking_info.json for banking RAG data.