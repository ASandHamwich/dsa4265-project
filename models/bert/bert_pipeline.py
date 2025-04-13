# Library
from bert_predict import predict
from monopoly.banks import BankDetector, banks
from monopoly.pdf import PdfDocument, PdfParser
from monopoly.pipeline import Pipeline
from monopoly.statements import Transaction
import argparse
import pandas as pd
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def statement_parser(pdf_path):
    document = PdfDocument(file_path=pdf_path)
    bank = BankDetector(document).detect_bank(banks)
    if bank is None:
        raise Exception("Bank could not be detected; please check your pdf file.")

    parser = PdfParser(BankDetector(document).detect_bank(banks), document)
    pipeline = Pipeline(parser)

    statement = pipeline.extract()
    transactions = pipeline.transform(statement)

    return transactions

def get_transactions_as_df(transactions: list[Transaction]) -> pd.DataFrame:
    transactions_as_dict = [txn.as_raw_dict() for txn in transactions]
    df = pd.DataFrame(transactions_as_dict)
    return df
    

def merge_predictions_with_df(transactions_df: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    if len(transactions_df) != len(predictions):
        raise ValueError("Number of transactions and predictions must match.")

    # Extract prediction fields
    labels = [pred["predicted_label"] for pred in predictions]
    confidences = [pred["confidence"] for pred in predictions]

    # Assign predictions to DataFrame
    transactions_df = transactions_df.copy()
    transactions_df["predicted_label"] = labels
    transactions_df["confidence"] = confidences

    # transactions_df.to_csv(os.path.join(current_dir, "prediction_txn.csv"))

    return transactions_df

def generate_transaction_dict(df: pd.DataFrame) -> dict:
    df["amount"] = df["amount"].astype(float)
    df["confidence"] = df["confidence"].astype(float)

    # Convert to list of dicts
    return df.to_dict(orient="records")


def generate_insights(df: pd.DataFrame) -> dict:
    # Convert amount column to float if needed
    df["amount"] = df["amount"].astype(float)

    # Identify income vs expenses
    income_df = df[(df["amount"] > 0)]
    expenses_df = df[(df["amount"] < 0)]

    # Compute aggregates
    income_total = income_df["amount"].sum()
    expenses_total = -expenses_df["amount"].sum()  # Make expenses positive

    # Group expenses by category
    category_totals = (
        expenses_df.groupby("predicted_label")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "DEFAULT_SPENDING_INSIGHTS": {
            "income": round(income_total, 2),
            "total_expenses": round(expenses_total, 2),
            "categories": {k.lower(): round(v, 2) for k, v in category_totals.items()}
        }
    }

def classify_statement(pdf_path):
    print(f"Parsing {pdf_path}...")
    transactions = statement_parser(pdf_path) # returns [Transaction]

    print(f"Parsed {len(transactions)} transactions.\n")

    transaction_df = get_transactions_as_df(transactions)
    descriptions = transaction_df["description"].tolist()

    results = predict(descriptions)
    final_df = merge_predictions_with_df(transaction_df, results)

    return final_df


def process_statement_pipeline(pdf_path):
    final_df = classify_statement(pdf_path)

    # Generate dictionary structures
    financial_data_dict = generate_insights(final_df)
    transactions_data_dict = generate_transaction_dict(final_df)

    # Convert to JSON strings
    financial_data_json = json.dumps(financial_data_dict, indent=2)
    transactions_json = json.dumps(transactions_data_dict, indent=2)

    return financial_data_json, transactions_json

def save_json_outputs(financial_data_json, transactions_json):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define file names
    insights_file = os.path.join(current_dir, "insights.json")
    transactions_file = os.path.join(current_dir, "transactions.json")

    # Write to files
    with open(insights_file, "w") as f:
        f.write(financial_data_json)

    with open(transactions_file, "w") as f:
        f.write(transactions_json)

    print(f"JSON outputs saved to:\n  - {insights_file}\n  - {transactions_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to bank statement PDF")
    args = parser.parse_args()

    financial_data_json, final_json = process_statement_pipeline(args.pdf_path)
    print(financial_data_json)
    print(final_json)

    # save_json_outputs(financial_data_json, final_json)
