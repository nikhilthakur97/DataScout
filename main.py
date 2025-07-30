import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

# Modern UI Configuration
st.set_page_config(
    page_title="DataScout - AI News Research",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

# Modern Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🔍 DataScout</h1>
    <p class="header-subtitle">AI-Powered News Research Assistant | Powered by Google Gemini</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with modern styling
with st.sidebar:
    st.markdown("""
    <div class="url-card">
        <h2 style="margin-top: 0; color: white;">📰 News Article URLs</h2>
        <p style="color: rgba(255,255,255,0.9); margin-bottom: 1rem;">
            Enter 1-3 news article URLs to analyze
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    urls = []
    for i in range(3):
        url = st.text_input(f"🔗 URL {i+1}", key=f"url_{i}", placeholder="https://example.com/news-article")
        urls.append(url)
    
    st.markdown("<br>", unsafe_allow_html=True)
    process_url_clicked = st.button("🚀 Process URLs", use_container_width=True)
    
    # API Status indicator
    if os.getenv("GEMINI_API_KEY"):
        st.success("✅ Gemini API Connected")
    else:
        st.error("❌ Add GEMINI_API_KEY to .env file")

# Main content area
st.markdown("""
<div class="chat-card">
    <h2 style="margin-top: 0; color: white;">💬 Ask Questions</h2>
    <p style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
        Once URLs are processed, ask any questions about the articles
    </p>
</div>
""", unsafe_allow_html=True)

# Processing section
main_placeholder = st.empty()

file_path = "datascout_knowledge.pkl"

# Use Google Gemini as the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

if process_url_clicked:
    # Filter out empty URLs
    valid_urls = [url for url in urls if url.strip()]
    if valid_urls:
        with main_placeholder.container():
            try:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Load data
                status_text.markdown("""
                <div class="progress-text">
                    🔄 Loading articles from URLs...
                </div>
                """, unsafe_allow_html=True)
                progress_bar.progress(25)
                
                loader = UnstructuredURLLoader(urls=valid_urls)
                data = loader.load()
                
                if not data:
                    status_text.markdown("""
                    <div class="status-error">
                        ❌ No content found from the provided URLs. Please check if the URLs are accessible.
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.empty()
                else:
                    # Step 2: Split text
                    status_text.markdown("""
                    <div class="progress-text">
                        ✂️ Breaking articles into chunks...
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress(50)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=['\n\n', '\n', '.', ','],
                        chunk_size=1000
                    )
                    docs = text_splitter.split_documents(data)
                    
                    if not docs:
                        status_text.markdown("""
                        <div class="status-error">
                            ❌ No text content found to process from the URLs.
                        </div>
                        """, unsafe_allow_html=True)
                        progress_bar.empty()
                    else:
                        # Step 3: Create embeddings
                        status_text.markdown("""
                        <div class="progress-text">
                            🧠 Creating AI embeddings...
                        </div>
                        """, unsafe_allow_html=True)
                        progress_bar.progress(75)
                        
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                        vectorstore_hf = FAISS.from_documents(docs, embeddings)
                        
                        # Step 4: Save database
                        status_text.markdown("""
                        <div class="progress-text">
                            💾 Saving knowledge database...
                        </div>
                        """, unsafe_allow_html=True)
                        progress_bar.progress(100)
                        
                        with open(file_path, "wb") as f:
                            pickle.dump(vectorstore_hf, f)
                        
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.markdown("""
                        <div class="status-success">
                            ✅ Processing Complete! Ready to answer your questions.
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                status_text.markdown(f"""
                <div class="status-error">
                    ❌ Error processing URLs: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                progress_bar.empty()
    else:
        main_placeholder.markdown("""
        <div class="status-error">
            ⚠️ Please enter at least one valid URL to continue.
        </div>
        """, unsafe_allow_html=True)

# Query interface
query = st.text_input(
    "💭 Ask your question:", 
    placeholder="What is the main topic of these articles?",
    help="Ask any question about the processed articles"
)

if query:
    if os.path.exists(file_path):
        with st.spinner("🤔 Thinking..."):
            try:
                with open(file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": query}, return_only_outputs=True)
                
                # Display answer with modern styling
                st.markdown("""
                <div class="answer-container">
                    <h3>🤖 Answer</h3>
                """, unsafe_allow_html=True)
                
                st.write(result["answer"])
                
                # Display sources if available
                sources = result.get("sources", "")
                if sources:
                    st.markdown("""
                    <div class="sources-container">
                        <h4>📚 Sources</h4>
                    """, unsafe_allow_html=True)
                    
                    sources_list = sources.split("\n")
                    for i, source in enumerate(sources_list, 1):
                        if source.strip():
                            st.markdown(f"**{i}.** {source}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
    else:
        st.warning("⚠️ Please process some URLs first before asking questions!")





