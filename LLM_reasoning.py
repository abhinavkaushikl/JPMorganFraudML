import streamlit as st
import pandas as pd
import os
import time
import datetime
import tempfile
from pathlib import Path
import traceback

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Fraud Detection RAG System")
st.markdown("""
This application uses a Retrieval-Augmented Generation (RAG) system to explain fraud detection results.
Upload your fraud data and feature importance files, train the system, and then ask questions about the data.
""")

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'limit_data' not in st.session_state:
    st.session_state.limit_data = True
if 'data_limit' not in st.session_state:
    st.session_state.data_limit = 1000

# Sidebar for configuration
st.sidebar.header("Configuration")

# Model selection
model_options = {
    "google/flan-t5-small": "Flan-T5 Small (fastest)",
    "google/flan-t5-base": "Flan-T5 Base (balanced)",
    "google/flan-t5-large": "Flan-T5 Large (better quality)"
}
selected_model = st.sidebar.selectbox(
    "Select Language Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0
)

# Data limiting options
st.sidebar.header("Data Options")
st.session_state.limit_data = st.sidebar.checkbox("Limit data size for faster processing", value=True)
if st.session_state.limit_data:
    st.session_state.data_limit = st.sidebar.slider("Maximum number of rows", 100, 10000, 1000, 100)

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Data & Training", "❓ Ask Questions", "🔍 Customer Lookup"])

# Define RAG system classes
def initialize_rag_system(fraud_path, importance_path, model_name, data_limit=1000):
    """Initialize and return the RAG system with all components."""
    
    # Import libraries here to avoid conflicts with Streamlit
    import torch
    from typing import List, Dict, Any, Optional, Tuple, Union
    from langchain.schema import Document
    
    # Use updated imports for LangChain components
    try:
        # Try to import from langchain-community and langchain-huggingface
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_huggingface import HuggingFacePipeline
        st.write("Using updated LangChain packages")
    except ImportError:
        # Fall back to legacy imports with warnings
        st.warning("Using deprecated LangChain imports. Consider upgrading with: pip install -U langchain-community langchain-huggingface")
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.llms import HuggingFacePipeline
    
    from langchain.chains import RetrievalQA
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, T5ForConditionalGeneration
    
    # Display PyTorch info
    st.sidebar.write(f"PyTorch version: {torch.__version__}")
    st.sidebar.write(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.sidebar.write(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Safe model loading function
    def safe_load_model(model_name, task=None):
        """
        Safely load a model with proper error handling
        
        Args:
            model_name: Name of the HuggingFace model to load
            task: The task for the pipeline (if None, will be inferred)
            
        Returns:
            tuple of (model, tokenizer, task)
        """
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Determine device and precision
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"
                dtype = torch.float32
                
            # Determine task based on model name if not provided
            if task is None:
                if "t5" in model_name.lower():
                    task = "text2text-generation"
                else:
                    task = "text-generation"
            
            # Load model with appropriate class based on model type
            if "t5" in model_name.lower():
                # Avoid using device_map="auto" which can cause the __path__._path error
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                # Move model to device explicitly
                model = model.to(device)
            else:
                # Avoid using device_map="auto" which can cause the __path__._path error
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                # Move model to device explicitly
                model = model.to(device)
            
            return model, tokenizer, task
            
        except Exception as e:
            st.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    class Timer:
        """Utility class for timing operations and printing progress."""
        
        def __init__(self, name: str):
            self.name = name
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            st.write(f"⏱️ Starting: {self.name}...")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                time_str = f"{elapsed:.2f} seconds"
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes} min {seconds:.2f} sec"
            st.write(f"✅ Completed: {self.name} in {time_str}")
    
    class DataLoader:
        """Handles loading and preprocessing of CSV data for fraud detection."""
        
        def __init__(self, fraud_path: str, importance_path: str, data_limit: int = None):
            """
            Initialize the data loader with paths to CSV files.
            
            Args:
                fraud_path: Path to the fraud data CSV
                importance_path: Path to the feature importance CSV
                data_limit: Maximum number of rows to load (for faster processing)
            """
            self.fraud_path = fraud_path
            self.importance_path = importance_path
            self.data_limit = data_limit
            self.fraud_df = None
            self.importance_df = None
            self.feature_importance = {}
            self.fraud_col = None
            self.id_col = None
            self.customer_id_col = None
        
        def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Load fraud and importance data from CSV files."""
            with Timer("Loading CSV data"):
                try:
                    self.fraud_df = pd.read_csv(self.fraud_path)
                    st.write(f"📊 Fraud data: {self.fraud_df.shape}")
                    
                    # Limit data size for faster processing if needed
                    if self.data_limit and len(self.fraud_df) > self.data_limit:
                        st.warning(f"Limiting data to {self.data_limit} rows for faster processing (from {len(self.fraud_df)} rows)")
                        self.fraud_df = self.fraud_df.head(self.data_limit)
                    
                    # Load feature importance if available
                    try:
                        self.importance_df = pd.read_csv(self.importance_path)
                        st.write(f"📊 Feature importance data: {self.importance_df.shape}")
                    except Exception as e:
                        st.warning(f"Could not load feature importance data: {e}")
                        self.importance_df = None
                    
                except FileNotFoundError as e:
                    st.error(f"Error loading CSV files: {e}")
                    raise
                
            return self.fraud_df, self.importance_df
        
        def prepare_feature_importance(self) -> Dict[str, float]:
            """Extract feature importance from the importance DataFrame."""
            with Timer("Preparing feature importance"):
                if self.importance_df is None:
                    st.warning("No feature importance data available")
                    return {}
                    
                if 'Feature' in self.importance_df.columns and 'Importance' in self.importance_df.columns:
                    self.feature_importance = dict(zip(self.importance_df['Feature'], self.importance_df['Importance']))
                    st.write(f"📊 Loaded {len(self.feature_importance)} feature importance values")
                else:
                    st.warning("Missing 'Feature' or 'Importance' column")
                    # Try to infer column names
                    if len(self.importance_df.columns) >= 2:
                        col1, col2 = self.importance_df.columns[:2]
                        self.feature_importance = dict(zip(self.importance_df[col1], self.importance_df[col2]))
                        st.write(f"Using columns {col1} and {col2} for feature importance")
                    else:
                        self.feature_importance = {}
                    
            return self.feature_importance
        
        def identify_key_columns(self) -> Tuple[str, str, str]:
            """Identify fraud indicator, transaction ID, and customer ID columns."""
            with Timer("Identifying key columns"):
                if self.fraud_df is None:
                    self.load_data()
                    
                # Find fraud column
                fraud_candidates = ['Is_Fraud', 'is_fraud', 'Fraud', 'fraud', 'isFraud', 'is_fraud_flag']
                self.fraud_col = next((col for col in fraud_candidates if col in self.fraud_df.columns), None)
                
                # Find ID column
                id_candidates = ['transaction_id', 'TransactionID', 'id', 'ID', 'trans_id']
                self.id_col = next((col for col in id_candidates if col in self.fraud_df.columns), None)
                
                # Find customer ID column
                customer_id_candidates = ['customer_id', 'CustomerID', 'customer', 'user_id', 'client_id', 'customer_id_hash']
                self.customer_id_col = next((col for col in customer_id_candidates if col in self.fraud_df.columns), None)
                
                # If we can't find the columns, try to infer them
                if not self.fraud_col:
                    st.warning("Could not find fraud indicator column")
                    # Try to infer fraud column - look for binary columns
                    binary_cols = [col for col in self.fraud_df.columns if set(self.fraud_df[col].unique()).issubset({0, 1})]
                    if binary_cols:
                        self.fraud_col = binary_cols[0]
                        st.write(f"Using {self.fraud_col} as fraud indicator column")
                    else:
                        raise ValueError("No fraud indicator column found")
                    
                if not self.id_col:
                    self.fraud_df['transaction_id'] = range(len(self.fraud_df))
                    self.id_col = 'transaction_id'
                    st.info("Created new ID column")
                
                if not self.customer_id_col:
                    # Look for columns that might contain UUIDs or long strings
                    for col in self.fraud_df.columns:
                        if self.fraud_df[col].dtype == 'object' and self.fraud_df[col].str.contains('-').any():
                            self.customer_id_col = col
                            st.info(f"Inferred customer ID column: {col}")
                            break
                
                st.write(f"📊 Using {self.fraud_col} as fraud indicator, {self.id_col} as transaction ID, and {self.customer_id_col or 'None'} as customer ID")
                
            return self.fraud_col, self.id_col, self.customer_id_col
    
    class DocumentCreator:
        """Creates document objects for the RAG system from fraud data."""
        
        def __init__(self, fraud_df: pd.DataFrame, fraud_col: str, id_col: str, 
                     feature_importance: Dict[str, float], customer_id_col: Optional[str] = None):
            """
            Initialize the document creator.
            
            Args:
                fraud_df: DataFrame containing fraud data
                fraud_col: Name of the column indicating fraud
                id_col: Name of the column containing transaction IDs
                feature_importance: Dictionary mapping features to importance scores
                customer_id_col: Name of the column containing customer IDs (optional)
            """
            self.fraud_df = fraud_df
            self.fraud_col = fraud_col
            self.id_col = id_col
            self.customer_id_col = customer_id_col
            self.feature_importance = feature_importance
        
        def create_documents(self) -> List[Document]:
            """Create document objects from all transactions."""
            with Timer(f"Creating documents for {len(self.fraud_df)} transactions"):
                documents = []
                
                # Create documents for all transactions
                for _, row in self.fraud_df.iterrows():
                    transaction_id = row[self.id_col]
                    is_fraud = bool(row[self.fraud_col] == 1)  # Ensure boolean conversion
                    customer_id = row[self.customer_id_col] if self.customer_id_col else "Unknown"
                    
                    # Create a clean representation of transaction details
                    # Limit the number of features to avoid token length issues
                    feature_items = [(col, row[col]) for col in row.index 
                                    if col.lower() != self.fraud_col.lower()]
                    
                    # Limit to 20 features to avoid token length issues
                    if len(feature_items) > 20:
                        feature_items = feature_items[:20]
                        
                    feature_lines = "\n".join([f"{col}: {val}" for col, val in feature_items])
                    
                    # Include feature importance for context
                    if self.feature_importance:
                        # Filter to only include features present in this transaction
                        relevant_features = {f: i for f, i in self.feature_importance.items() 
                                            if f in self.fraud_df.columns}
                        if relevant_features:
                            top_feats = sorted(relevant_features.items(), key=lambda x: x[1], reverse=True)[:5]
                            top_feature_text = "\n".join([f"{f} (importance: {i:.4f})" for f, i in top_feats])
                        else:
                            top_feature_text = "No matching feature importance data available"
                    else:
                        top_feature_text = "Feature importance data not available"
                    
                    fraud_label = "FRAUD" if is_fraud else "NOT FRAUD"
                    
                    doc = Document(
                        page_content=f"""Customer ID: {customer_id}
Transaction ID: {transaction_id}
Transaction Details:
{feature_lines}
Top Contributing Features:
{top_feature_text}
This transaction is labeled as {fraud_label}.""",
                        metadata={
                            "transaction_id": str(transaction_id), 
                            "customer_id": str(customer_id),
                            "is_fraud": is_fraud
                        }
                    )
                    documents.append(doc)
                        
                st.write(f"📊 Created {len(documents)} document chunks")
                
            return documents
    
    class VectorDBBuilder:
        """Creates and manages the vector database for document retrieval."""
        
        def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
            """
            Initialize the vector database builder.
            
            Args:
                embedding_model_name: Name of the HuggingFace model to use for embeddings
            """
            self.embedding_model_name = embedding_model_name
            self.embedding = None
            self.db = None
        
        def create_embeddings(self) -> HuggingFaceEmbeddings:
            """Create the embedding model."""
            with Timer(f"Loading embedding model {self.embedding_model_name}"):
                # Use model_kwargs to avoid PyTorch issues
                self.embedding = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": "cpu"}  # Force CPU to avoid CUDA issues
                )
            return self.embedding
        
        def build_vector_db(self, documents: List[Document]) -> FAISS:
            """Build the FAISS vector database from documents."""
            if self.embedding is None:
                self.create_embeddings()
                
            with Timer(f"Building FAISS index for {len(documents)} documents"):
                try:
                    self.db = FAISS.from_documents(documents, self.embedding)
                except Exception as e:
                    st.error(f"Error creating FAISS index: {e}")
                    raise
                    
            return self.db
    
    class LLMProvider:
        """Provides language model capabilities for the RAG system."""
        
        def __init__(self, model_name: str = "google/flan-t5-small"):
            """
            Initialize the LLM provider.
            
            Args:
                model_name: Name of the HuggingFace model to use
            """
            self.model_name = model_name
            self.tokenizer = None
            self.model = None
            self.pipe = None
            self.llm = None
        
        def load_model(self) -> HuggingFacePipeline:
            """Load the language model from HuggingFace."""
            with Timer(f"Loading language model {self.model_name}"):
                try:
                    st.write(f"📦 Model: {self.model_name}")
                    
                    # Check if CUDA is available
                    if torch.cuda.is_available():
                        device_info = f"GPU: {torch.cuda.get_device_name(0)}"
                        dtype_info = "FP16 precision"
                    else:
                        device_info = "CPU only"
                        dtype_info = "FP32 precision"
                    
                    st.write(f"🖥️ Hardware: {device_info}, {dtype_info}")
                    
                    # Use our safe model loader
                    with Timer("Loading model and tokenizer"):
                        self.model, self.tokenizer, task = safe_load_model(self.model_name)
                    
                    # Create pipeline with smaller max_length to avoid token length issues
                    with Timer("Creating inference pipeline"):
                        self.pipe = pipeline(
                            task, 
                            model=self.model, 
                            tokenizer=self.tokenizer, 
                            max_new_tokens=128,  # Reduced for faster inference
                            max_length=512  # Limit input length to avoid token length issues
                        )
                        
                        self.llm = HuggingFacePipeline(pipeline=self.pipe)
                    
                except Exception as e:
                    st.error(f"Error loading Hugging Face model: {e}")
                    raise
                    
            return self.llm
    
    class RAGSystem:
        """Retrieval-Augmented Generation system for fraud explanation."""
        
        def __init__(self, 
                     data_loader: DataLoader, 
                     document_creator: Optional[DocumentCreator] = None,
                     vector_db_builder: Optional[VectorDBBuilder] = None,
                     llm_provider: Optional[LLMProvider] = None):
            """
            Initialize the RAG system.
            
            Args:
                data_loader: DataLoader instance for loading data
                document_creator: DocumentCreator instance (created if None)
                vector_db_builder: VectorDBBuilder instance (created if None)
                llm_provider: LLMProvider instance (created if None)
            """
            self.data_loader = data_loader
            self.document_creator = document_creator
            self.vector_db_builder = vector_db_builder or VectorDBBuilder()
            self.llm_provider = llm_provider or LLMProvider()
            self.documents = None
            self.db = None
            self.llm = None
            self.rag_chain = None
            self.is_trained = False
            self.training_time = None
        
        def train(self) -> 'RAGSystem':
            """
            Train/setup the RAG system (load data, create documents, build vector DB, load LLM).
            This is separated from the query functionality.
            """
            train_start_time = time.time()
            st.write(f"🚀 Starting RAG system training at {datetime.datetime.now().strftime('%H:%M:%S')}")
            
            # Load and prepare data
            self.data_loader.load_data()
            feature_importance = self.data_loader.prepare_feature_importance()
            fraud_col, id_col, customer_id_col = self.data_loader.identify_key_columns()
            
            # Create document creator if not provided
            if self.document_creator is None:
                self.document_creator = DocumentCreator(
                    self.data_loader.fraud_df, 
                    fraud_col, 
                    id_col, 
                    feature_importance,
                    customer_id_col
                )
            
            # Create documents
            self.documents = self.document_creator.create_documents()
            
            # Build vector database
            self.db = self.vector_db_builder.build_vector_db(self.documents)
            
            # Load language model
            self.llm = self.llm_provider.load_model()
            
            # Create RAG chain
            with Timer("Building RAG chain"):
                try:
                    self.rag_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        retriever=self.db.as_retriever(search_kwargs={"k": 3}),
                        chain_type="stuff",
                        return_source_documents=True
                    )
                    self.is_trained = True
                except Exception as e:
                    st.error(f"Error building RAG pipeline: {e}")
                    raise
            
            self.training_time = time.time() - train_start_time
            minutes = int(self.training_time // 60)
            seconds = self.training_time % 60
            
            st.success(f"RAG system training complete in {minutes} min {seconds:.2f} sec")
            st.write(f"📊 System ready with {len(self.documents)} documents indexed")
                
            return self
        
        def query(self, query_text: str) -> Dict[str, Any]:
            """
            Run a query through the RAG system.
            
            Args:
                query_text: The query to answer
                
            Returns:
                The response from the RAG system
            """
            if not self.is_trained or self.rag_chain is None:
                raise ValueError("RAG system not trained. Call train() first.")
                
            st.write(f"💬 Running query: {query_text}")
            query_start_time = time.time()
            
            try:
                # Use invoke instead of run (to fix deprecation warning)
                with Timer("Generating answer"):
                    response = self.rag_chain.invoke({"query": query_text})
                
                query_time = time.time() - query_start_time
                st.write(f"⏱️ Query completed in {query_time:.2f} seconds")
                
                return response
            except Exception as e:
                st.error(f"Error running query: {e}")
                raise
        
        def direct_lookup(self, customer_id: str) -> Dict[str, Any]:
            """
            Directly look up a customer by ID without using the RAG system.
            
            Args:
                customer_id: The customer ID to look up
                
            Returns:
                Information about the customer
            """
            if not self.is_trained:
                raise ValueError("RAG system not trained. Call train() first.")
                
            with Timer(f"Looking up customer {customer_id}"):
                if self.data_loader.customer_id_col is None:
                    return {"error": "No customer ID column identified in the data"}
                
                customer_data = self.data_loader.fraud_df[
                    self.data_loader.fraud_df[self.data_loader.customer_id_col] == customer_id
                ]
                
                if customer_data.empty:
                    return {"error": f"Customer ID {customer_id} not found in the data"}
                
                is_fraud = bool(customer_data[self.data_loader.fraud_col].iloc[0] == 1)
                
                return {
                    "customer_id": customer_id,
                    "is_fraud": is_fraud,
                    "fraud_column_value": customer_data[self.data_loader.fraud_col].iloc[0],
                    "transaction_count": len(customer_data),
                    "data": customer_data.to_dict('records')
                }
    
    # Create and return the RAG system
    data_loader = DataLoader(fraud_path, importance_path, data_limit)
    llm_provider = LLMProvider(model_name=model_name)
    rag_system = RAGSystem(data_loader, llm_provider=llm_provider)
    
    return rag_system

# Tab 1: Data & Training
with tab1:
    st.header("Data Upload & System Training")
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_file = st.file_uploader("Upload Fraud Data CSV", type=["csv"])
    
    with col2:
        importance_file = st.file_uploader("Upload Feature Importance CSV (optional)", type=["csv"])
        if not importance_file:
            st.info("Feature importance data helps explain why transactions are flagged as fraud.")
    
    # Save uploaded files to temp location
    fraud_path = None
    importance_path = None
    
    if fraud_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            f.write(fraud_file.getvalue())
            fraud_path = f.name
        
        # For importance file, create an empty one if not provided
        if importance_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                f.write(importance_file.getvalue())
                importance_path = f.name
        else:
            # Create an empty importance file
            empty_importance = pd.DataFrame(columns=['Feature', 'Importance'])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                empty_importance.to_csv(f.name, index=False)
                importance_path = f.name
        
        st.success("Files processed successfully!")
        
        # Display data preview
        st.subheader("Fraud Data Preview")
        fraud_df = pd.read_csv(fraud_path)
        st.dataframe(fraud_df.head())
        
        # Train button
        if st.button("Train RAG System"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize and train the RAG system
                with st.spinner("Initializing RAG system..."):
                    progress_bar.progress(0.1)
                    status_text.text("Loading libraries and initializing system...")
                    
                    # Create and train the RAG system
                    rag_system = initialize_rag_system(
                        fraud_path, 
                        importance_path,
                        selected_model,
                        st.session_state.data_limit if st.session_state.limit_data else None
                    )
                    
                    progress_bar.progress(0.3)
                    status_text.text("Training RAG system...")
                
                # Train the system
                with st.expander("Training Log", expanded=True):
                    log_container = st.container()
                    with log_container:
                        rag_system.train()
                
                progress_bar.progress(1.0)
                status_text.text("Training complete!")
                
                # Store in session state
                st.session_state.rag_system = rag_system
                st.session_state.is_trained = True
                
                st.success("🎉 RAG system trained successfully!")
                
            except Exception as e:
                st.error(f"Error training RAG system: {e}")
                st.error("Please check the details below for more information.")
                st.code(traceback.format_exc())
    else:
        st.info("Please upload a fraud data CSV file to proceed.")

# Tab 2: Ask Questions
with tab2:
    st.header("Ask Questions About Fraud")
    st.markdown("""
    Ask questions about fraud patterns and get explanations based on the data and feature importance.
    The system will use feature importance to explain why certain transactions are flagged as fraudulent.
    """)
    
    if not st.session_state.is_trained:
        st.warning("Please train the RAG system first in the 'Data & Training' tab.")
    else:
        # Question input
        question = st.text_input("Enter your question about fraud detection:", 
                                "Why are certain transactions marked as fraud? What features are most important?")
        
        if st.button("Ask Question"):
            with st.spinner("Generating answer..."):
                try:
                    # Run the query
                    response = st.session_state.rag_system.query(question)
                    
                    # Extract answer
                    answer = response.get("result", "No answer found")
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(answer)
                    
                    # Display source documents if available
                    source_docs = response.get("source_documents", [])
                    if source_docs:
                        with st.expander("Source Documents", expanded=False):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(f"Metadata: {doc.metadata}")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error asking question: {e}")
                    st.code(traceback.format_exc())

# Tab 3: Customer Lookup
with tab3:
    st.header("Customer Lookup")

    if not st.session_state.is_trained:
        st.warning("Please train the RAG system first in the 'Data & Training' tab.")
    else:
        customer_id = st.text_input("Enter Customer ID:")
        if st.button("Lookup Customer"):
            with st.spinner("Looking up customer..."):
                try:
                    # Run the query
                    response = st.session_state.rag_system.query(f"Customer ID: {customer_id}")
                    # Display customer information
                    st.subheader("Customer Information:")
                    st.write(response.get("result", "No information found"))
                except Exception as e:
                    st.error(f"Error looking up customer: {e}")
                    st.code(traceback.format_exc())
