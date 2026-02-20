import os
import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ==============================
# 🔐 Set Your Groq API Key
# ==============================
os.environ["GROQ_API_KEY"] = ""

# ==============================
# 🤖 Initialize Groq Model
# ==============================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0  # deterministic summary
)

# ==============================
# 🎨 Streamlit UI
# ==============================
st.set_page_config(page_title="AI PDF Summarizer", page_icon="📄")
st.title("📄 AI PDF Summarizer")
st.write("Upload a PDF file and get an AI-generated summary.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    if st.button("Summarize PDF"):

        with st.spinner("Reading and summarizing document... ⏳"):

            # Load PDF
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(documents)

            # ==============================
            # 🧠 STEP 1: Summarize Each Chunk
            # ==============================
            chunk_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert document summarizer."),
                ("human", "Summarize the following text:\n\n{text}")
            ])

            chunk_chain = chunk_prompt | llm

            chunk_summaries = []

            for doc in docs:
                result = chunk_chain.invoke({"text": doc.page_content})
                chunk_summaries.append(result.content)

            # ==============================
            # 🧠 STEP 2: Combine Summaries
            # ==============================
            combined_text = "\n\n".join(chunk_summaries)

            final_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert summarizer."),
                ("human", 
                 "Combine the following summaries into a clear and concise final summary:\n\n{summaries}")
            ])

            final_chain = final_prompt | llm

            final_result = final_chain.invoke({"summaries": combined_text})

            # ==============================
            # 📌 Display Result
            # ==============================
            st.subheader("📌 Final Summary")
            st.write(final_result.content)