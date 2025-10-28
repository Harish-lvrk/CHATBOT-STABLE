# ------------------------- Phase 1 Libraries -------------------------
import os
import warnings
import logging
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ------------------------- Phase 2 Libraries -------------------------
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# Initialize an empty list to store conversation history
conversation_history: list[BaseMessage] = []

# ------------------------- Phase 3 Libraries -------------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# ------------------------- Constants -------------------------
EMBEDDING_MODEL = "all-MiniLM-L12-v2"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
# Curated list of powerful chat models from Groq
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
]

# ------------------------- Setup & Configuration -------------------------
# Suppress warnings for a cleaner interface
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
load_dotenv()

# ------------------------- Caching & Helper Functions -------------------------
@st.cache_resource
def get_embeddings_model():
    """Loads the HuggingFace embeddings model and caches it."""
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        st.error(f"Error loading embeddings model: {str(e)}")
        return None

@st.cache_resource
def get_llm(_groq_api_key, model_name):
    """Loads the Groq LLM for a given model and caches it."""
    return ChatGroq(groq_api_key=_groq_api_key, model=model_name)

def get_vectorstore(_uploaded_file, embeddings):
    """
    Load an uploaded PDF, create a temporary file, and build a vector database.
    """
    if _uploaded_file is None or embeddings is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    vectorstore = None
    try:
        loader = PyPDFLoader(tmp_file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        documents = loader.load_and_split(text_splitter=text_splitter)
        if documents:
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings,
                normalize_L2=True  # Add L2 normalization for better similarity search
            )
        else:
            st.warning("No text content was extracted from the PDF.")
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
    return vectorstore

def get_response_stream(prompt, llm, selected_docs, messages):
    """
    Determines the response mode (document-based or standard) and returns a streaming response.
    """
    if selected_docs:
        # --- Document-based RAG response ---
        context_parts = []
        for doc_name in selected_docs:
            vectorstore = st.session_state.vectorstores[doc_name]
            # Create retriever with explicit parameters
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            # Use invoke instead of get_relevant_documents
            try:
                relevant_docs = retriever.invoke(prompt)
                if relevant_docs:
                    doc_context = "\n\n".join(doc.page_content for doc in relevant_docs)
                    context_parts.append(f"--- CONTEXT FROM '{doc_name}' ---\n{doc_context}\n--- END OF CONTEXT ---")
            except Exception as e:
                st.error(f"Error retrieving documents from {doc_name}: {str(e)}")
        
        if not context_parts:
            return iter(["I could not find any relevant information in the selected documents to answer your question."])
        
        context = "\n\n".join(context_parts)
        
        # Create message history for the prompt
        history_messages = []
        for msg in messages[:-1]:  # Exclude the last message as it's the current prompt
            if isinstance(msg, HumanMessage):
                history_messages.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                history_messages.append(("assistant", msg.content))
        
        # Create a RAG prompt that incorporates chat history
        rag_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n\nContext:\n{context}"),
            *history_messages,  # Unpack the history messages
            ("human", "{question}")
        ])

        # Create the LCEL chain
        rag_chain = rag_prompt_template | llm | StrOutputParser()
        
        # Stream the response
        return rag_chain.stream({
            "question": prompt,
            "context": context
        })

    else:
        # --- Standard conversational response ---
        # Create message history for the prompt
        history_messages = []
        for msg in messages[:-1]:  # Exclude the last message as it's the current prompt
            if isinstance(msg, HumanMessage):
                history_messages.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                history_messages.append(("assistant", msg.content))

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            *history_messages,  # Unpack the history messages
            ("human", "{question}")
        ])
        
        chain = prompt_template | llm | StrOutputParser()
        return chain.stream({"question": prompt})

# ------------------------- Main Application -------------------------
def main():
    """
    Main function to run the Streamlit application.
    """
    st.title('📘 Ask Chatbot! (Multi-PDF + Groq LLM)')
    st.write("Upload, select, and chat with multiple PDFs, or just have a normal conversation.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.warning("Groq API key not found. Please add it to your environment variables (.env file).")
        st.stop()

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = {}
    if "selected_documents" not in st.session_state:
        st.session_state.selected_documents = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_LLM_MODEL
    
    if not st.session_state.messages:
         st.session_state.messages.append(AIMessage(content="Hello! Please upload PDFs or ask me anything."))

    # --- Sidebar UI ---
    with st.sidebar:
        st.header("Controls")

        if st.button("Clear Conversation"):
            st.session_state.messages = [AIMessage(content="Conversation cleared. How can I help you?")]
            st.rerun()
        
        st.session_state.selected_model = st.selectbox(
            "Choose a model:", options=GROQ_MODELS, index=GROQ_MODELS.index(st.session_state.selected_model)
        )
        
        st.header("Upload & Select Documents")
        uploaded_file = st.file_uploader("Upload a new PDF", type="pdf")

        if uploaded_file and uploaded_file.name not in st.session_state.vectorstores:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                embeddings = get_embeddings_model()
                vectorstore = get_vectorstore(uploaded_file, embeddings)
                if vectorstore:
                    st.session_state.vectorstores[uploaded_file.name] = vectorstore
                    if uploaded_file.name not in st.session_state.selected_documents:
                        st.session_state.selected_documents.append(uploaded_file.name)
                    st.success(f"Processed and selected {uploaded_file.name}")
                    st.rerun()

        if st.session_state.vectorstores:
            st.session_state.selected_documents = st.multiselect(
                "Choose documents to chat with:",
                options=list(st.session_state.vectorstores.keys()),
                default=st.session_state.selected_documents
            )

    # --- Load Model ---
    llm = get_llm(groq_api_key, st.session_state.selected_model)
    
    # --- Chat History Display ---
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # --- Chat Input Handling ---
    if prompt := st.chat_input("💬 Ask your question..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Get the stream from the appropriate chainexplain about these papers
                stream = get_response_stream(prompt, llm, st.session_state.selected_documents, st.session_state.messages)
                
                # Stream the response to the UI
                response_content = st.write_stream(stream)
                
                # Add model attribution and save the full message
                model_attribution = f"\n\n*— Responded by **{st.session_state.selected_model}***"
                full_response = response_content + model_attribution
                st.session_state.messages.append(AIMessage(content=full_response))
                
                # Need to rerun to append the model attribution to the last message
                st.rerun()

            except Exception as e:
                error_message = f"🚨 An error occurred: {str(e)}"
                st.markdown(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))

if __name__ == "__main__":
    main()

