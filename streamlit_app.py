import os
import sys
import streamlit as st
from io import BytesIO

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# from models.bert.bert_pipeline import process_statement_pipeline  # returns aggregated spendings
# from models.llm_agent.llm_agent import FinTailorAI   # returns list of followup questions
# from models.rag.RAG import rag  # returns relevant banking information based on query and spendings


# Streamlit App UI

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

st.subheader("Step 1: Upload Your Transaction Statement (PDF)")
uploaded_file = st.file_uploader("Upload your transaction statement (PDF) here", type=["pdf"])

if uploaded_file is not None:
    st.info("Processing your transaction statement...")
    # try:
    #     # You may need to use BytesIO(uploaded_file.read()) depending on your implementation
    #     aggregated_spendings, transactions = process_statement_pipeline(uploaded_file)
    #     st.session_state.aggregated_spendings = aggregated_spendings
    #     st.success("Transactions processed successfully!")
    #     st.write("**Aggregated Spendings:**", aggregated_spendings)
    # except Exception as e:
    #     st.error(f"Error processing statement: {e}")

# # 2. Get Initial Query from User

# st.header("Step 2: Ask Your Financial Question")
# initial_query = st.text_input("Enter your question:", value=st.session_state.initial_query)

# if st.button("Submit Initial Query"):
#     if initial_query.strip() == "":
#         st.error("Please enter a query.")
#     else:
#         st.session_state.initial_query = initial_query
#         followup_qs = FinTailorAI.analyse_query_and_generate_questions(initial_query)
#         st.session_state.followup_questions = followup_qs
#         st.session_state.current_question_idx = 0
#         st.success("Initial query saved. Follow-up questions generated!")

# # 3. Handle Follow-up Questions
# if st.session_state.initial_query and st.session_state.followup_questions:
#     st.header("Step 3: Follow-up Questions")
#     current_idx = st.session_state.current_question_idx

#     if current_idx < len(st.session_state.followup_questions):
#         question = st.session_state.followup_questions[current_idx]
#         st.write(f"**Follow-up Question {current_idx+1}:** {question}")
#         answer = st.text_input("Your Answer:", key=f"answer_{current_idx}")

#         if st.button("Submit Answer", key=f"submit_{current_idx}"):
#             if answer.strip() == "":
#                 st.error("Please enter an answer before submitting.")
#             else:
#                 # Save the Q&A pair and move to next question
#                 st.session_state.qa_pairs.append((question, answer))
#                 st.session_state.current_question_idx += 1
#                 st.experimental_rerun()
#     else:
#         st.success("All follow-up questions answered.")

# # 4. Final Response Generation

# if st.session_state.initial_query and st.session_state.qa_pairs:
#     st.header("Step 4: Generate Final Response")
#     followup_qa_dict = {q: a for q, a in st.session_state.qa_pairs}
#     st.write("**Follow-up Answers:**")
#     st.json(followup_qa_dict)
    
#     if st.button("Generate Final Response"):
#         # Retrieve relevant banking info using RAG
#         agg_spendings = st.session_state.aggregated_spendings if st.session_state.aggregated_spendings else ""
#         try:
#             relevant_info = rag(initial_query, agg_spendings)
#         except Exception as e:
#             st.error(f"Error during retrieval (RAG): {e}")
#             relevant_info = None
        
#         if relevant_info is not None:
#             # Generate final answer using LLM agent
#             try:
#                 final_response = FinTailorAI.generate_financial_recommendation(user_query=initial_query, financial_data=agg_spendings, banking_rag_data=relevant_info, additional_info=followup_qa_dict)
#             except Exception as e:
#                 st.error(f"Error during LLM generation: {e}")
#                 final_response = "An error occurred during answer generation."
#             st.write("### Final Response")
#             st.success(final_response)
#             st.write("### Relevant Banking Information")
#             st.write(relevant_info)
