import json
import os
import time
from typing import Dict, Any, Optional, List
import traceback
import openai

# Default banking data
DEFAULT_BANKING_RAG_DATA = [
    {
      "bank": "dbs",
      "url": "https://www.dbs.com.sg/personal/cards/credit-cards/dbs-yuu-cards",
      "tag": "credit cards",
      "section": "DBS yuu CardFor both sides of yuu! > At a Glance",
      "text_snippet": "DBS yuu CardFor both sides of yuu!\nAt a Glance\n\nEarn up to 18% cash rebates or 10 miles per S$1NEW!\nWhen you hit S$600 qualified monthly spend\n\nEnjoy 5% cash rebates or 2.8 miles per S$1NEW!\nWith no minimum spend and no cap\n\nGet S$2 off eggs and rice monthly\nAt Cold Storage and Giant\nLimited time sign-up bonus: New to DBS/POSB Cardmembers can enjoy S$300 cashback with promo code: 'CASH300'.",
      "score": 0.5197107791900635
    },
    {
      "bank": "dbs",
      "url": "https://www.dbs.com.sg/i-bank/cards/debit-cards/dbs-visa-debit",
      "tag": "debit cards",
      "section": "DBS Visa Debit Card - Your Multi-Currency Card > Features & Benefits",
      "text_snippet": "DBS Visa Debit Card - Your Multi-Currency Card\nFeatures & Benefits\n\n4% cashback on online food delivery\nValid with a minimum of S$500 on Visa and cash withdrawals limit of S$400 and below in the same month.\n\n3% cashback on local transport (Ride-hailing, taxis, Transit - Simply Go)\nValid with a minimum of S$500 on Visa and cash withdrawals limit of S$400 and below in the same month.",
      "score": 0.49988865852355957
    },
    {
      "bank": "dbs",
      "url": "https://www.dbs.com.sg/personal/deposits/bank-with-ease/dbs-digi-bank-travel-mode",
      "tag": "digital services",
      "section": "Travel Mode > Features & Benefits of Travel Mode",
      "text_snippet": "Travel Mode\nFeatures & Benefits of Travel Mode\n\nPlan your trip\nAlso don't forget to visit DBS Travel & Leisure Marketplace, the most rewarding one-stop travel platform for DBS/POSB customers. Whether you are going on an overseas holiday or a business trip, you can book your flights and hotels there.\nWhat's more, you can also enjoy complimentary travel insurance (now enhanced with COVID-19 coverage), enjoy 2x more value when you pay with DBS Points and choose to split your purchase amount into flexible monthly instalments.",
      "score": 0.5225870013237
    },
    {
      "bank": "ocbc",
      "url": "https://www.ocbc.com/personal-banking/cards/90-degrees-travel-credit-card.page",
      "tag": "credit cards",
      "section": "OCBC 90°N card > Earn miles or cash rebates that never expire",
      "text_snippet": "OCBC 90°N card\nEarn miles or cash rebates that never expire\n\nWhy you will love this\nEarn up to 7 Miles (equivalent to 7% cash rebate) for every S$1 spent on Agoda\n\nWhy you will love this\nEarn 1.3 Miles (equivalent to 1.3% cash rebate) for every S$1 spent in SGD and 2.1 Miles (equivalent to 2.1% cash rebate) for every S$1 spent in foreign currency\n\nWhy you will love this\nEarn unlimited miles that never expire",
      "score": 0.5046662092208862
    },
    {
      "bank": "ocbc",
      "url": "https://www.ocbc.com/personal-banking/cards/ocbc-debit-card.page",
      "tag": "credit cards",
      "section": "OCBC DEBIT CARD > The card that gives you limitless cashback",
      "text_snippet": "OCBC DEBIT CARD\nThe card that gives you limitless cashback\n\nWhy you will love this\nEarn unlimited 1% cashback on your daily Visa spending in selected categories",
      "score": 0.5032830238342285
    },
    {
      "bank": "ocbc",
      "url": "https://www.ocbc.com/personal-banking/cards/90-degrees-travel-credit-card.page",
      "tag": "credit cards",
      "section": "OCBC 90°N card > Earn miles or cash rebates that never expire",
      "text_snippet": "GetaCard\nLink your OCBC Debit Card to a Multi-Currency Global Savings Account to avoid foreign currency transaction fees when you make online transactions. Remember to keep sufficient funds in the relevant currencies!",
      "additional_info": "This is a composite entry created to represent the OCBC Multi-Currency Global Savings Account based on information in the original dataset."
    }
]

class FinTailorAI:
    """
    FinTailorAI class for generating personalized financial recommendations
    using OpenAI's GPT models.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the FinTailorAI with OpenAI API credentials."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided. Either pass it directly or set OPENAI_API_KEY environment variable.")

        openai.api_key = self.api_key
        self.client = openai
        self.model = model

    def _get_model(self):
        """Return the appropriate model to use, defaulting to gpt-4o-mini for gpt-4 to avoid rate limits."""
        return "gpt-4o-mini" if self.model == "gpt-4" else self.model

    def _validate_banking_data(self, banking_rag_data):
        """Return a list of valid banking data items."""
        valid_items = []

        # Handle case where banking_rag_data might be in a nested structure
        if isinstance(banking_rag_data, dict):
            # Check if there's a 'results' key that contains the actual data
            if "results" in banking_rag_data:
                banking_rag_data = banking_rag_data["results"]
            # Check if there's a 'data' key that contains the actual data
            elif "data" in banking_rag_data:
                banking_rag_data = banking_rag_data["data"]

        if banking_rag_data and isinstance(banking_rag_data, list):
            for item in banking_rag_data:
                if isinstance(item, dict) and "bank" in item and "text_snippet" in item:
                    valid_items.append(item)

        return valid_items

    def _extract_financial_summary(self, financial_data):
        """Extract key financial metrics from financial data."""
        if not financial_data:
            return None

        # Handle the financial data structure
        income = financial_data.get("DEFAULT_SPENDING_INSIGHTS", {}).get("income", 0)
        total_expenses = financial_data.get("DEFAULT_SPENDING_INSIGHTS", {}).get("total_expenses", 0)

        return {
            "income": income,
            "expenses": total_expenses,
            "surplus": income - total_expenses,
            # No debt or assets in the new data structure
            "debt": 0,
            "assets": 0,
            "rate": f"{((income - total_expenses) / income * 100):.1f}%" if income > 0 else "0%"
        }

    def _process_transaction_data(self, transaction_data):
        """Process transaction data to extract insights."""
        # Handle case where transaction_data might be in a nested structure
        if isinstance(transaction_data, dict):
            # Check if there's a 'transactions' or 'data' key that contains the actual data
            if "transactions" in transaction_data:
                transaction_data = transaction_data["transactions"]
            elif "data" in transaction_data:
                transaction_data = transaction_data["data"]

        if not transaction_data or not isinstance(transaction_data, list):
            return None

        # Initialize summaries
        transaction_summary = {
            "total_inflow": 0,
            "total_outflow": 0,
            "transaction_count": len(transaction_data),
            "categories": {},
            "recent_transactions": []
        }

        # Process each transaction
        for transaction in transaction_data:
            amount = transaction.get("amount", 0)
            category = transaction.get("predicted_label", "Unknown")

            # Add to inflow/outflow totals
            if amount > 0:
                transaction_summary["total_inflow"] += amount
            else:
                transaction_summary["total_outflow"] += abs(amount)

            # Update category summaries
            if category not in transaction_summary["categories"]:
                transaction_summary["categories"][category] = 0
            transaction_summary["categories"][category] += abs(amount)

            # Keep recent transactions (up to 5)
            if len(transaction_summary["recent_transactions"]) < 5:
                transaction_summary["recent_transactions"].append({
                    "date": transaction.get("date", ""),
                    "description": transaction.get("description", ""),
                    "amount": amount,
                    "category": category
                })

        return transaction_summary

    def _call_openai_with_retry(self, messages, model=None, max_tokens=1000, temperature=0.4):
        """Call OpenAI API with retry logic for rate limits."""
        model_to_use = model or self._get_model()
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_message = str(e)
                if "rate_limit" in error_message and attempt < max_retries - 1:
                    print(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e

    def _parse_questions(self, response_text):
        """Parse questions from API response text."""
        # First try to parse as JSON
        try:
            questions = json.loads(response_text)
            if isinstance(questions, list):
                return questions[:3]  # Limit to 3 questions max
        except json.JSONDecodeError:
            pass

        # If that fails, try to extract questions using regex
        import re

        # Look for questions in quotes
        potential_questions = re.findall(r'"([^"]+\?)"', response_text)
        if potential_questions:
            return potential_questions[:3]

        # Look for numbered questions
        numbered_questions = re.findall(r'\d+\.\s*(.+\?)', response_text)
        if numbered_questions:
            return numbered_questions[:3]

        return []

    def analyze_query_and_generate_questions(
        self,
        user_query: str,
        financial_data: Dict[str, Any] = None,
        transaction_data: List[Dict[str, Any]] = None,
        banking_rag_data: List[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate relevant follow-up questions based on user query and available data."""
        model_to_use = self._get_model()

        # Create the original detailed prompt for query analysis
        analysis_prompt = (
            "You are a financial advisor analyzing a user's financial question. "
            "Your goal is to identify what additional information would help provide "
            "a more personalized recommendation.\n\n"
            f"User's Question:\n{user_query}\n\n"
        )

        # Add financial summary if available
        if financial_data:
            user_financial_profile = self._format_financial_profile(financial_data)
            analysis_prompt += f"User's Financial Profile:\n{user_financial_profile}\n\n"
        else:
            analysis_prompt += "Note: No financial profile data has been provided by the user.\n\n"

        # Add transaction data if available
        if transaction_data:
            transaction_summary = self._process_transaction_data(transaction_data)
            if transaction_summary:
                analysis_prompt += "User's Recent Transaction Summary:\n"
                analysis_prompt += f"- Total Inflow: S${transaction_summary['total_inflow']:.2f}\n"
                analysis_prompt += f"- Total Outflow: S${transaction_summary['total_outflow']:.2f}\n"
                analysis_prompt += "- Recent Transactions:\n"
                for tx in transaction_summary["recent_transactions"]:
                    analysis_prompt += f"  * {tx['date']}: {tx['description']} - S${tx['amount']:.2f} ({tx['category']})\n"
                analysis_prompt += "\n"

        # Add banking information
        valid_banking_items = self._validate_banking_data(banking_rag_data)
        if valid_banking_items:
            banking_context = "Available Banking Products Context:\n"
            for i, item in enumerate(valid_banking_items[:5], 1):  # Limit to top 5 items for brevity
                bank = item["bank"].upper()
                tag = item.get("tag", "").title()
                text = item.get("text_snippet", "").strip()
                banking_context += f"{i}. {bank} - {tag}: {text}\n"
            analysis_prompt += f"{banking_context}\n\n"
        else:
            analysis_prompt += "Note: No banking product data has been provided.\n\n"

        analysis_prompt += (
            "Based on this information, generate 1-3 follow-up questions that would help you provide a more tailored recommendation. "
            "These questions should:\n"
            "1. Be specific and directly relevant to their financial goal\n"
            "2. Ask for information not already available (including basic financial information if none was provided)\n"
            "3. Help you understand their personal preferences, risk tolerance, timeline, or banking preferences\n"
            "4. Be conversational and easy to answer\n\n"
            "Format your response as a JSON array of strings, with each string being a question. "
            "Example format: [\"Question 1?\", \"Question 2?\", \"Question 3?\"]\n\n"
            "If you believe you have all necessary information, return an empty array: []"
        )

        try:
            response_text = self._call_openai_with_retry(
                messages=[
                    {"role": "system", "content": "You are a financial analysis assistant with specific knowledge of Singapore banking products."},
                    {"role": "user", "content": analysis_prompt}
                ],
                model=model_to_use,
                temperature=0.7,
                max_tokens=200
            )
            return self._parse_questions(response_text)
        except Exception as e:
            print(f"Error generating follow-up questions: {str(e)}")
            return []

    def generate_financial_recommendation(
        self,
        user_query: str,
        financial_data: Dict[str, Any] = None,
        transaction_data: List[Dict[str, Any]] = None,
        banking_rag_data: List[Dict[str, Any]] = None,
        additional_info: Dict[str, str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.4
    ) -> str:
        """Generate personalized financial recommendations based on user query and available data."""
        model_to_use = self._get_model()

        # Use the original comprehensive system prompt
        system_prompt = (
            "You are a highly knowledgeable and empathetic financial advisor with expertise in "
            "personal finance, budgeting, debt management, investing, and long-term financial planning, "
            "with specific knowledge about Singapore's banking products and services. "
            "Your goal is to provide personalized, actionable financial advice that addresses the user's "
            "specific situation and query, incorporating relevant banking products when appropriate. You should:"
            "\n- Analyze any provided financial data thoroughly, if available"
            "\n- Consider the user's specific goals, concerns, and personal information shared"
            "\n- Reference specific banking products from the RAG data when available and relevant"
            "\n- Provide clear, step-by-step explanations for your recommendations"
            "\n- Use specific numbers and timeframes when providing projections, if sufficient data is available"
            "\n- Adjust your advice based on the level of information available"
            "\n- Structure your response as a conversational paragraph (no bullet points or headers)"
            "\n- Acknowledge when you're providing general advice due to limited information"
        )

        # Create the detailed user prompt with the original structure
        user_prompt = f"User's Financial Goal:\n{user_query}\n\n"

        # Add financial profile if available
        if financial_data:
            user_financial_profile = self._format_financial_profile(financial_data)
            user_prompt += f"User's Financial Profile:\n{user_financial_profile}\n\n"
        else:
            user_prompt += "Note: No financial profile data has been provided by the user.\n\n"

        # Add transaction data if available
        if transaction_data:
            transaction_summary = self._process_transaction_data(transaction_data)
            if transaction_summary:
                user_prompt += "User's Recent Transaction Summary:\n"
                user_prompt += f"- Total Inflow: S${transaction_summary['total_inflow']:.2f}\n"
                user_prompt += f"- Total Outflow: S${transaction_summary['total_outflow']:.2f}\n"
                user_prompt += "- Recent Transactions:\n"
                for tx in transaction_summary["recent_transactions"]:
                    user_prompt += f"  * {tx['date']}: {tx['description']} - S${tx['amount']:.2f} ({tx['category']})\n"
                user_prompt += "\n"

        # Add banking information with the original formatting
        valid_banking_items = self._validate_banking_data(banking_rag_data)
        if valid_banking_items:
            rag_info_text = "Relevant Banking Products and Services:\n\n"

            # Group by bank and tag for better organization
            bank_tag_groups = {}
            for item in valid_banking_items:
                bank = item["bank"].upper()
                tag = item.get("tag", "").title()
                key = f"{bank} - {tag}"

                if key not in bank_tag_groups:
                    bank_tag_groups[key] = []

                bank_tag_groups[key].append(item)

            # Format each group
            for group_key, items in bank_tag_groups.items():
                rag_info_text += f"# {group_key}:\n"

                for item in items:
                    section = item.get("section", "")
                    text = item.get("text_snippet", "").strip()
                    url = item.get("url", "")

                    if section and text:
                        rag_info_text += f"- {section}: {text}\n"
                        if url:
                            rag_info_text += f"  More info: {url}\n"

                rag_info_text += "\n"

            user_prompt += f"{rag_info_text}\n"
        else:
            user_prompt += "Note: No specific banking product data has been provided.\n\n"

        # Add additional information with original formatting
        if additional_info and len(additional_info) > 0:
            additional_info_text = "Additional Information from User:\n"
            for question, answer in additional_info.items():
                additional_info_text += f"Q: {question}\nA: {answer}\n\n"
            user_prompt += f"{additional_info_text}\n"

        # Use the original detailed instructions
        user_prompt += (
            f"Based on the above information, please provide a financial recommendation "
            f"that directly addresses the user's goal. "
        )

        # Adjust instructions based on available data
        if financial_data or transaction_data:
            user_prompt += "Tailor your advice to their specific financial situation. "
        else:
            user_prompt += "Provide general advice, acknowledging the lack of specific financial details. "

        if valid_banking_items:
            user_prompt += "If appropriate, recommend specific banking products or services from the relevant banking information that would meet their needs. "

        user_prompt += (
            f"Include actions they should take and explain your reasoning. "
            f"Your recommendation should be practical, actionable, and as personalized as possible given the available information."
        )

        try:
            return self._call_openai_with_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_to_use,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            return f"Error generating recommendation: {str(e)}"

    def _format_financial_profile(self, financial_data: Dict[str, Any]) -> str:
        """Format financial data into a readable format for the simplified structure."""
        # Handle case where financial_data is a list
        if isinstance(financial_data, list) and len(financial_data) > 0:
            # Use the first item in the list
            financial_data = financial_data[0]

        # Handle case where financial_data doesn't have the expected structure
        if not isinstance(financial_data, dict):
            # Fallback to empty data
            print("Warning: Financial data format is not as expected. Using defaults.")
            spending_insights = {"income": 0, "total_expenses": 0, "categories": {}}
        else:
            # Extract from structure
            spending_insights = financial_data.get("DEFAULT_SPENDING_INSIGHTS", {})
            if not spending_insights and isinstance(financial_data, dict):
                # Try to use the financial_data directly if it doesn't have the expected structure
                spending_insights = financial_data

        # Get income and expenses with fallbacks
        income = spending_insights.get("income", 0)
        total_expenses = spending_insights.get("total_expenses", 0)

        # Calculate savings
        monthly_surplus = income - total_expenses
        savings_rate = (monthly_surplus / income * 100) if income > 0 else 0

        # Format the financial profile as a structured text block
        profile = (
            f"Monthly Income: S${income:.2f}\n"
            f"Monthly Expenses: S${total_expenses:.2f}\n"
            f"Monthly Surplus: S${monthly_surplus:.2f}\n"
            f"Savings Rate: {savings_rate:.1f}%\n\n"
        )

        # No assets or debts in the new structure
        profile += "Monthly Spending Breakdown:\n"

        # Add spending breakdown
        for category, amount in spending_insights.get("categories", {}).items():
            percentage = (amount / total_expenses) * 100 if total_expenses > 0 else 0
            profile += f"- {category.replace('_', ' ').title()}: S${amount:.2f} ({percentage:.1f}%)\n"

        return profile


def load_json_file(file_path: str) -> Any:
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded {file_path}, data type: {type(data).__name__}")
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {str(e)}")
        return {}


def generate_financial_advice(
    user_query: str,
    financial_data: Dict[str, Any] = None,
    transaction_data: List[Dict[str, Any]] = None,
    banking_rag_data: List[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    with_followup_questions: bool = True
) -> str:
    """Generate financial advice based on user query and available data."""
    # Use default banking data if none provided
    if banking_rag_data is None:
        banking_rag_data = DEFAULT_BANKING_RAG_DATA

    try:
        # Initialize the FinTailorAI advisor
        financial_advisor = FinTailorAI(api_key=api_key)

        if with_followup_questions:
            # Get follow-up questions based on the query
            followup_questions = financial_advisor.analyze_query_and_generate_questions(
                user_query, financial_data, transaction_data, banking_rag_data
            )

            # If we have follow-up questions, ask them
            additional_info = {}
            if followup_questions:
                print("\nTo provide a more tailored recommendation, I'd like to ask you a few questions:")
                for i, question in enumerate(followup_questions, 1):
                    print(f"\n{i}. {question}")
                    answer = input("Your answer (press Enter to skip): ")
                    if answer.strip():
                        additional_info[question] = answer
                    time.sleep(0.5)  # Slight delay for better user experience

                if additional_info:
                    print("\nThanks for providing additional information. Generating your recommendation...")
                else:
                    print("\nNo additional information provided. Generating recommendation based on available information...")
            else:
                data_sources = ["banking products information"]
                if financial_data:
                    data_sources.insert(0, "your financial profile")
                if transaction_data:
                    data_sources.insert(0, "your transaction history")

                print(f"\nGenerating your recommendation based on {' and '.join(data_sources)}...")

            # Generate recommendation with additional information
            return financial_advisor.generate_financial_recommendation(
                user_query=user_query,
                financial_data=financial_data,
                transaction_data=transaction_data,
                banking_rag_data=banking_rag_data,
                additional_info=additional_info
            )
        else:
            # Generate recommendation without follow-up questions
            return financial_advisor.generate_financial_recommendation(
                user_query=user_query,
                financial_data=financial_data,
                transaction_data=transaction_data,
                banking_rag_data=banking_rag_data
            )

    except Exception as e:
        return f"Error generating financial advice: {str(e)}\n{traceback.format_exc()}"


def run_interactive_advisor(financial_json_path: str = None, transaction_json_path: str = None, banking_rag_json_path: str = None):
    """Run an interactive financial advisor that uses JSON inputs for financial data and banking RAG data."""
    print("\n" + "="*80)
    print("Welcome to FinTailorAI - Your Personal Financial Advisor with Banking Product Insights!")
    print("="*80)

    # Load financial data
    financial_data = None
    if financial_json_path:
        print(f"Loading financial data from {financial_json_path}...")
        financial_data = load_json_file(financial_json_path)
        if financial_data:
            print("Financial data loaded successfully!")
        else:
            print("Failed to load financial data. No financial data will be used in analysis.")
    else:
        print("No financial data specified. No financial data will be used in analysis.")

    # Load transaction data
    transaction_data = None
    if transaction_json_path:
        print(f"Loading transaction data from {transaction_json_path}...")
        transaction_data = load_json_file(transaction_json_path)
        if transaction_data:
            print("Transaction data loaded successfully!")
        else:
            print("Failed to load transaction data. No transaction data will be used in analysis.")
    else:
        print("No transaction data specified. No transaction data will be used in analysis.")

    # Load banking RAG data if provided, otherwise use default
    banking_rag_data = DEFAULT_BANKING_RAG_DATA
    if banking_rag_json_path:
        print(f"Loading banking information data from {banking_rag_json_path}...")
        custom_banking_data = load_json_file(banking_rag_json_path)
        if custom_banking_data:
            print("Banking information data loaded successfully!")
            # If loaded as a dictionary with a root key, extract the list
            if isinstance(custom_banking_data, dict) and "results" in custom_banking_data:
                banking_rag_data = custom_banking_data["results"]
            else:
                banking_rag_data = custom_banking_data
        else:
            print("Failed to load banking information data. Using default banking product data.")
    else:
        print("Using default banking product data.")

    # Show status message based on available data
    data_sources = ["banking product information"]
    if financial_data:
        data_sources.insert(0, "financial profile")
    if transaction_data:
        data_sources.insert(0, "transaction history")

    print(f"\nReady to provide financial advice based on {' and '.join(data_sources)}!")
    print("Enter your financial questions or goals below.")
    print("Type 'exit' to quit, 'status' to see available data, or 'help' for assistance.")

    # Define command handlers
    def display_help():
        print("\nHere are some example questions you can ask:")
        print("- How can I build an emergency fund with Singapore banks?")
        print("- What's the best way to pay off credit card debt?")
        print("- Should I invest more or pay down debt?")
        print("- What credit cards in Singapore are good for my spending habits?")
        print("- What's a good savings goal for me?")
        print("- Which debit card would be best for my situation?")
        print("- How can I save money on my transportation expenses?")
        print("- What spending category should I focus on reducing?")

    def display_status():
        print("\nAvailable Data:")
        if financial_data:
            print("✓ Financial profile data is loaded")
            spending_insights = financial_data.get("DEFAULT_SPENDING_INSIGHTS", {})
            print(f"  - Income: S${spending_insights.get('income', 0)}/month")
            print(f"  - Expenses: S${spending_insights.get('total_expenses', 0)}/month")

            # Display spending categories if available
            categories = spending_insights.get("categories", {})
            if categories:
                print("  - Spending Categories:")
                for category, amount in categories.items():
                    print(f"    • {category.title()}: S${amount}/month")
        else:
            print("✗ No financial profile data loaded")

        if transaction_data:
            print("✓ Transaction data is loaded")
            print(f"  - {len(transaction_data)} transactions available")
            # Show a few recent transactions
            if transaction_data and len(transaction_data) > 0:
                print("  - Recent transactions:")
                for tx in transaction_data[:3]:  # Show only the first 3
                    date = tx.get("date", "")
                    desc = tx.get("description", "")[:30] + "..." if len(tx.get("description", "")) > 30 else tx.get("description", "")
                    amount = tx.get("amount", 0)
                    print(f"    • {date}: {desc} - S${amount}")
        else:
            print("✗ No transaction data loaded")

        print("✓ Banking product information is loaded")
        # Group banking information by bank and card type
        bank_info = {}
        for item in banking_rag_data:
            if isinstance(item, dict) and "bank" in item and "tag" in item:
                bank = item["bank"].upper()
                tag = item["tag"].title()

                if bank not in bank_info:
                    bank_info[bank] = set()

                bank_info[bank].add(tag)

        for bank, tags in bank_info.items():
            print(f"  - {bank}: {', '.join(sorted(tags))}")

    # Command mapping
    commands = {
        "exit": lambda: "exit",
        "quit": lambda: "exit",
        "help": display_help,
        "status": display_status
    }

    try:
        while True:
            user_query = input("\nYour financial question: ")
            cmd = user_query.lower()

            if cmd in commands:
                result = commands[cmd]()
                if result == "exit":
                    print("\nThank you for using FinTailorAI. Goodbye!")
                    break
                continue

            # Ensure the query ends with a question mark for better AI parsing
            if not user_query.endswith("?"):
                user_query += "?"

            # Generate and display the recommendation
            recommendation = generate_financial_advice(
                user_query=user_query,
                financial_data=financial_data,
                transaction_data=transaction_data,
                banking_rag_data=banking_rag_data,
                with_followup_questions=True
            )

            print("="*80)
            print("Financial Recommendation:")
            print("="*80)
            print(recommendation)
            print("="*80)

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        print("\nThank you for using FinTailorAI! We hope our financial advice was helpful.")
        print("Have a great day and remember to make smart financial choices!\n")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        traceback.print_exc()


# Example usage with hardcoded paths
if __name__ == "__main__":
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "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"

    # Hardcoded paths for the BERT and RAG model outputs
    financial_json_path = "../bert/__________.json"
    transaction_json_path = "../bert/__________.json"
    banking_rag_json_path = "../rag/use_relevant_banking_info.json"

    # Run the interactive advisor with the hardcoded file paths
    run_interactive_advisor(financial_json_path, transaction_json_path, banking_rag_json_path)