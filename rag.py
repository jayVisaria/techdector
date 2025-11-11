"""
Simple RAG Application with Streamlit
======================================
A beginner-friendly RAG (Retrieval-Augmented Generation) system.

This app demonstrates the core concepts of RAG:
1. Load documents
2. Split into chunks
3. Create embeddings and store in vector database
4. Retrieve relevant chunks
5. Generate answers using LLM

Perfect for learning and teaching RAG concepts!
"""
# pip install chromadb langchain langchain-community langchain-google-genai langchain-text-splitters langchain-chroma pypdf

import streamlit as st
import os
from pathlib import Path

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# Page configuration
st.set_page_config(
    page_title="Simple RAG App",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Simple RAG Application")
st.markdown("""
This app demonstrates **Retrieval-Augmented Generation (RAG)** - a technique that allows 
AI to answer questions based on your own documents.

### How it works:
1. **Upload** your documents (PDF or TXT)
2. **Process** - The app splits documents into chunks and creates embeddings
3. **Ask** questions about your documents
4. **Get** accurate answers based on the content
""")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password", help="Get your API key from https://ai.google.dev/")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ API Key set!")
    else:
        st.warning("⚠️ Please enter your Google API Key to continue")
    
    st.markdown("---")
    st.markdown("""
    ### About RAG
    **RAG** combines:
    - **Retrieval**: Finding relevant information
    - **Augmentation**: Adding context to queries
    - **Generation**: Creating answers with LLM
    """)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main content
if not api_key:
    st.info("👈 Please enter your Google API Key in the sidebar to get started")
    st.stop()

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Step 1: Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT files that you want to ask questions about"
    )
    
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing your documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_dir = Path("./temp_docs")
                    temp_dir.mkdir(exist_ok=True)
                    
                    all_documents = []
                    
                    # Load each file
                    for uploaded_file in uploaded_files:
                        file_path = temp_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load document based on type
                        if uploaded_file.name.endswith('.pdf'):
                            loader = PyPDFLoader(str(file_path))
                        else:
                            loader = TextLoader(str(file_path))
                        
                        docs = loader.load()
                        all_documents.extend(docs)
                        
                        st.success(f"✅ Loaded: {uploaded_file.name}")
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                    )
                    splits = text_splitter.split_documents(all_documents)
                    
                    st.info(f"📄 Created {len(splits)} text chunks")
                    
                    # Create embeddings and vector store
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004"
                    )
                    
                    vector_store = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        collection_name="simple_rag"
                    )
                    
                    st.session_state.vector_store = vector_store
                    
                    st.success("🎉 Documents processed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")

with col2:
    st.header("💬 Step 2: Ask Questions")
    
    if st.session_state.vector_store is None:
        st.info("👈 Please upload and process documents first")
    else:
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        if st.button("🔍 Get Answer", type="primary") and question:
            with st.spinner("Thinking..."):
                try:
                    # Initialize LLM
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash-lite",
                        temperature=0.3,
                        convert_system_message_to_human=True
                    )
                    
                    # Create prompt
                    system_prompt = (
                        "You are a helpful assistant answering questions based on the provided context. "
                        "Use the context below to answer the question. "
                        "If you cannot find the answer in the context, say so clearly.\n\n"
                        "Context:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    # Create RAG chain
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(
                        st.session_state.vector_store.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        question_answer_chain
                    )
                    
                    # Get answer
                    response = rag_chain.invoke({"input": question})
                    
                    # Display answer
                    st.markdown("### 💡 Answer:")
                    st.write(response["answer"])
                    
                    # Show retrieved context
                    with st.expander("📚 View Retrieved Context"):
                        for i, doc in enumerate(response["context"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(doc.page_content[:300] + "...")
                            st.markdown("---")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response["answer"]
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error getting answer: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.header("📜 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {chat['question'][:50]}..."):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using LangChain & Streamlit | Perfect for learning RAG!</p>
</div>
""", unsafe_allow_html=True)
