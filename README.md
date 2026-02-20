# PDF-Summarizer_ai
# 📄 DocuMind AI – PDF Summarizer (LangChain + Groq)

## Setup Commands

1. Create project folder  
mkdir PDFSummarizerProject  

2. Enter folder  
cd PDFSummarizerProject  

3. Create virtual environment  
python -m venv venv  

4. Activate environment (Windows)  
venv\Scripts\activate  

5. Install dependencies  
pip install langchain langchain-groq langchain-community langchain-text-splitters pypdf streamlit  

6. Run the application  
streamlit run app.py  

## Flow
Upload PDF → Load & Split Text → Summarize Chunks → Combine → Final Summary Output
