import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DocxLoader, ExcelLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from transformers import pipeline

# Set environment variables for API keys
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "###add your key here###"
os.environ["GOOGLE_API_KEY"] = "###add your key here###"

base_addr = "add your address"
documents_directory = "add your address"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Cache FAISS index to avoid loading it multiple times
@st.cache_resource
def load_faiss_index():
    return FAISS.load_local(documents_directory + "faiss_pdf", embeddings, allow_dangerous_deserialization=True)

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="/n",
        chunk_size=750,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def generate_summary(docs):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    
    for doc in docs:
        text = doc.page_content
        tokens = summarizer.tokenizer.encode(text)
        for i in range(0, len(tokens), 1024):
            chunk = summarizer.tokenizer.decode(tokens[i:i + 1024], skip_special_tokens=True)
            input_length = len(chunk.split())
            max_length = max(48, min(130, input_length // 2))
            summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

def handle_user_input(user_question):
    if 'conversation' in st.session_state:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        display_chat_history()

def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User messages
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Bot messages
            citations = st.session_state.get('citations', '')
            bot_message = f"{message.content}\n\n{citations}"
            st.write(bot_template.replace("{{MSG}}", bot_message), unsafe_allow_html=True)

def main():
    citations = ""
    st.set_page_config(page_title="Cybersecurity Audit Report Analyzer", page_icon=":shield:")
    st.write(css, unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.header("Cybersecurity Audit Report Analyzer")
    user_question = st.text_input("Ask a question about your document and hit Submit button")
    submit = st.button("Submit")
    st.session_state['user_question'] = user_question

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload your pdf, docx, or xlsx files and then click on Process", type=['pdf', 'docx', 'xlsx'], accept_multiple_files=True)
        process = st.button("Process")
        summarize = st.button("Generate Summary")

        if process:
            if uploaded_files:
                documents = []
                for file in uploaded_files:
                    if file.name.lower().endswith('.pdf'):
                        loader = PyPDFLoader(file_path=file)
                    elif file.name.lower().endswith('.docx'):
                        loader = DocxLoader(file_path=file)
                    elif file.name.lower().endswith('.xlsx'):
                        loader = ExcelLoader(file_path=file)

                    if loader:
                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)

                if documents:
                    text_chunks = get_text_chunks(documents)
                    if os.path.exists(documents_directory + "faiss_pdf"):
                        db2 = FAISS.from_documents(text_chunks, embeddings)
                        db1 = load_faiss_index()
                        db1.merge_from(db2)
                        db1.save_local(documents_directory + "faiss_pdf")
                        retriever = db1.as_retriever()
                        st.session_state['retriever'] = retriever
                    else:
                        db1 = FAISS.from_documents(text_chunks, embeddings)
                        db1.save_local(documents_directory + "faiss_pdf")
                        retriever = db1.as_retriever()
                        st.session_state['retriever'] = retriever

        if summarize:
            if uploaded_files:
                documents = []
                for file in uploaded_files:
                    if file.name.lower().endswith('.pdf'):
                        loader = PyPDFLoader(file_path=file)
                    elif file.name.lower().endswith('.docx'):
                        loader = DocxLoader(file_path=file)
                    elif file.name.lower().endswith('.xlsx'):
                        loader = ExcelLoader(file_path=file)

                    if loader:
                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)

                if documents:
                    text_chunks = get_text_chunks(documents)
                    summary = generate_summary(text_chunks)
                    st.write("Summary:", summary)

    if submit:
        faiss_index = load_faiss_index()
        retriever = faiss_index.as_retriever()
        st.session_state['retriever'] = retriever
        docs_and_scores = faiss_index.similarity_search_with_score(user_question)
        top_3_search_results_text = [docs_and_scores[i] for i in range(0, 2)]
        top_1_result = [docs_and_scores[0]]

        prompt = f"""Context: You are a virtual assistant with access to a document. Your task is to provide accurate responses to specific queries based on the content of the document.
                      Documents: {top_3_search_results_text}
                      Question: {user_question}
                      Response Format: Please provide relevant answer to the question based on the information provided in the Documents."""
        
        st.session_state['top1result'] = top_1_result
        st.session_state['FAISS_Context'] = top_3_search_results_text

        for res in top_3_search_results_text[0:-1]:
            doc = res[0]
            citations += "Source_Name: " + os.path.basename(doc.metadata["source"]) + ", Page: " + str(int(doc.metadata["page"]) + 1) + " \n"
        st.session_state['citations'] = citations

        llm = GoogleGenerativeAI(model="gemini-pro")
        conversation_chain = RetrievalQA.from_chain_type(llm,
                                                         chain_type='stuff',
                                                         return_source_documents=True,
                                                         retriever=faiss_index.as_retriever(),
                                                         chain_type_kwargs={"verbose": True,
                                                                            "memory": ConversationBufferMemory(
                                                                                input_key="question",
                                                                                memory_key="chat_history",
                                                                                return_messages=True
                                                                            )})
        st.session_state.conversation = conversation_chain

        if user_question:
            res = conversation_chain(prompt)
            response = res['result']
            st.session_state.chat_history = conversation_chain.combine_documents_chain.memory
            display_chat_history()

if __name__ == "__main__":
    main()
