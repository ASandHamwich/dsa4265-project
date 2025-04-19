import os
import sys
import streamlit as st
from io import BytesIO
from openai import OpenAI
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.bert.bert_pipeline import process_statement_pipeline
from models.llm_agent.llm_agent import FinTailorAI

# Initiate FinTailorAI
os.environ["OPENAI_API_KEY"] = "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"
ai = FinTailorAI(api_key=os.environ["OPENAI_API_KEY"])

# Streamlit App UI

st.markdown("<style>button[kind='expander'] div div { font-size: 1.05em !important; }</style>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>💰 FinTailorAI 💰</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Your Personalized Financial Recommendations Generator</h6>", unsafe_allow_html=True)

if "aggregated_spendings" not in st.session_state:
    st.session_state.aggregated_spendings = None

if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []  # List of tuples (question, answer)

if "followup_questions" not in st.session_state:
    st.session_state.followup_questions = []  # list of questions (strings)

if "current_question_idx" not in st.session_state:
    st.session_state.current_question_idx = 0

if "initial_query" not in st.session_state:
    st.session_state.initial_query = ""

# 1. Upload Transactions PDF (Optional)
st.subheader("💸 Step 1: Upload Your Transaction Statement (optional)")
uploaded_file = st.file_uploader("Upload your transaction statement (PDF) here", type=["pdf"])

if uploaded_file is not None:
    if "statement_status" not in st.session_state:
        st.session_state.statement_status = "processing"  # can be "processing", "success", "error"

    # Only show processing info if still processing
    if st.session_state.statement_status == "processing":
        st.info("File uploaded. Processing your transaction statement...")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
                temp_pdf_file.write(uploaded_file.read())
                temp_pdf_path = temp_pdf_file.name

            aggregated_spendings, transactions = process_statement_pipeline(temp_pdf_path)
            st.session_state.aggregated_spendings = aggregated_spendings
            st.session_state.statement_status = "success"
            st.rerun()  # to hide st.info on next render
        except Exception as e:
            st.session_state.statement_status = "error"
            st.error(f"Error processing statement: {e}")
    elif st.session_state.statement_status == "success":
        st.success("Transactions processed successfully!")
    elif st.session_state.statement_status == "error":
        st.error("Something went wrong during processing.")

# 2. Get Initial Query from User
st.subheader("❓ Step 2: Ask Your Financial Question")
initial_query = st.text_input("Enter your question:", value=st.session_state.initial_query)

if st.button("Submit Query"):
    if initial_query.strip() == "":
        st.error("Please enter a query.")
    else:
        st.session_state.initial_query = initial_query
        followup_qs = ["What card do you use now?", "What rewards would you like?", "How much do you spend a month?"]
        # followup_qs = ai.analyze_query_and_generate_questions(user_query=initial_query)
        st.session_state.followup_questions = followup_qs
        st.session_state.current_question_idx = 0
        st.success("Your query has been submitted. Here are some follow-up questions for you!")

# 3. Handle Follow-up Questions
if st.session_state.initial_query and st.session_state.followup_questions:
    st.subheader("📝 Step 3: Follow-up Questions")

    with st.form("followup_form"):
        all_answers_filled = True
        for idx, question in enumerate(st.session_state.followup_questions):
            answer_key = f"answer_input_{idx}"

            # Initialize each answer input in session state
            if answer_key not in st.session_state:
                if idx < len(st.session_state.qa_pairs):
                    st.session_state[answer_key] = st.session_state.qa_pairs[idx][1]
                else:
                    st.session_state[answer_key] = ""

            st.markdown(f"**Question {idx + 1}:** {question}")
            st.text_input("Your Answer:", key=answer_key)

        submitted = st.form_submit_button("Submit All Answers")

    if submitted:
        updated_pairs = []
        for idx, question in enumerate(st.session_state.followup_questions):
            answer = st.session_state[f"answer_input_{idx}"].strip()
            if answer == "":
                all_answers_filled = False
                break
            updated_pairs.append((question, answer))

        if not all_answers_filled:
            st.warning("Please complete all answers before submitting.")
        else:
            st.session_state.qa_pairs = updated_pairs
            st.session_state["submitted_all_answers"] = True
            st.rerun()

if st.session_state.get("submitted_all_answers", False):
    st.success("✅ All answers submitted successfully.")

# 4. Final Response Generation
if st.session_state.get("submitted_all_answers", False) and st.session_state.initial_query and st.session_state.qa_pairs:
    if "final_response" not in st.session_state:
        followup_qa_dict = {q: a for q, a in st.session_state.qa_pairs}

        info_placeholder = st.info("⏳ Generating your final response...")

        initial_query = st.session_state.initial_query
        agg_spendings = st.session_state.aggregated_spendings if st.session_state.aggregated_spendings else ""

        try:
            relevant_info = rag(initial_query=initial_query, agg_spendings=agg_spendings)
        except Exception as e:
            st.error(f"Error during retrieval (RAG): {e}")
            relevant_info = None

        if relevant_info is not None:
            try:
                final_response = ai.generate_financial_recommendation(
                    user_query=initial_query,
                    financial_data=agg_spendings,
                    banking_rag_data=relevant_info,
                    additional_info=followup_qa_dict
                )
                print(final_response)
            except Exception as e:
                st.error(f"Error during LLM generation: {e}")
                final_response = "An error occurred during answer generation."
        else:
            final_response = "No relevant information could be retrieved."

        st.session_state.final_response = final_response

        info_placeholder.empty()
        st.rerun()
    else:
        st.subheader("💰 Your Financial Recommendation")
        st.text(st.session_state.final_response)
