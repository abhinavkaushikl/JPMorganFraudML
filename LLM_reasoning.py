import pandas as pd
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
from pathlib import Path

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class DataLoader:
    """Handles loading and preprocessing of CSV data for fraud detection."""
    
    def __init__(self, fraud_path: str, importance_path: str):
        """
        Initialize the data loader with paths to CSV files.
        
        Args:
            fraud_path: Path to the fraud data CSV
            importance_path: Path to the feature importance CSV
        """
        self.fraud_path = fraud_path
        self.importance_path = importance_path
        self.fraud_df = None
        self.importance_df = None
        self.feature_importance = {}
        self.fraud_col = None
        self.id_col = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load fraud and importance data from CSV files."""
        try:
            self.fraud_df = pd.read_csv(self.fraud_path)
            self.importance_df = pd.read_csv(self.importance_path)
            print(f" Loaded fraud data: {self.fraud_df.shape}, importance data: {self.importance_df.shape}")
        except FileNotFoundError as e:
            print(f"Error loading CSV files: {e}")
            raise
        
        return self.fraud_df, self.importance_df
    
    def prepare_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from the importance DataFrame."""
        if self.importance_df is None:
            self.load_data()
            
        if 'Feature' in self.importance_df.columns and 'Importance' in self.importance_df.columns:
            self.feature_importance = dict(zip(self.importance_df['Feature'], self.importance_df['Importance']))
        else:
            print("⚠️ Missing 'Feature' or 'Importance' column")
            self.feature_importance = {}
            
        return self.feature_importance
    
    def identify_key_columns(self) -> Tuple[str, str]:
        """Identify fraud indicator and transaction ID columns."""
        if self.fraud_df is None:
            self.load_data()
            
        # Find fraud column
        fraud_candidates = ['Is_Fraud', 'is_fraud', 'Fraud', 'fraud']
        self.fraud_col = next((col for col in fraud_candidates if col in self.fraud_df.columns), None)
        
        # Find ID column
        id_candidates = ['transaction_id', 'TransactionID', 'id', 'ID']
        self.id_col = next((col for col in id_candidates if col in self.fraud_df.columns), None)
        
        if not self.fraud_col:
            print(" Could not find fraud indicator column")
            raise ValueError("No fraud indicator column found")
            
        if not self.id_col:
            self.fraud_df['transaction_id'] = range(len(self.fraud_df))
            self.id_col = 'transaction_id'
            print("ℹ️ Created new ID column")
            
        return self.fraud_col, self.id_col


class DocumentCreator:
    """Creates document objects for the RAG system from fraud data."""
    
    def __init__(self, fraud_df: pd.DataFrame, fraud_col: str, id_col: str, feature_importance: Dict[str, float]):
        """
        Initialize the document creator.
        
        Args:
            fraud_df: DataFrame containing fraud data
            fraud_col: Name of the column indicating fraud
            id_col: Name of the column containing transaction IDs
            feature_importance: Dictionary mapping features to importance scores
        """
        self.fraud_df = fraud_df
        self.fraud_col = fraud_col
        self.id_col = id_col
        self.feature_importance = feature_importance
    
    def create_documents(self) -> List[Document]:
        """Create document objects from fraudulent transactions."""
        documents = []
        fraud_rows = self.fraud_df[self.fraud_df[self.fraud_col] == 1]
        print(f"🔍 Found {len(fraud_rows)} fraudulent transactions")
        
        if fraud_rows.empty:
            documents.append(Document(
                page_content="No fraudulent transactions found in the dataset.",
                metadata={"transaction_id": "none"}
            ))
        else:
            for _, row in fraud_rows.iterrows():
                transaction_id = row[self.id_col]
                feature_lines = "\n".join([f"{col}: {row[col]}" for col in row.index 
                                          if col.lower() != self.fraud_col.lower()])
                
                if self.feature_importance:
                    top_feats = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_feature_text = "\n".join([f"{f} (importance: {i:.2f})" for f, i in top_feats])
                else:
                    top_feature_text = "Feature importance data not available"
                
                doc = Document(
                    page_content=f"""Transaction ID: {transaction_id}
Transaction Details:
{feature_lines}
Top Contributing Features:
{top_feature_text}
This transaction is labeled as FRAUD.""",
                    metadata={"transaction_id": transaction_id}
                )
                documents.append(doc)
                
        print(f"Created {len(documents)} document chunks")
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
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        return self.embedding
    
    def build_vector_db(self, documents: List[Document]) -> FAISS:
        """Build the FAISS vector database from documents."""
        if self.embedding is None:
            self.create_embeddings()
            
        try:
            self.db = FAISS.from_documents(documents, self.embedding)
            print("✅ Successfully created FAISS index")
        except Exception as e:
            print(f" Error creating FAISS index: {e}")
            raise
            
        return self.db


class LLMProvider:
    """Provides language model capabilities for the RAG system."""
    
    def __init__(self, model_name: str = "tiiuae/falcon-7b-instruct"):
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
        try:
            print(f"📦 Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            self.pipe = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                max_new_tokens=512
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            print("✅ Loaded Hugging Face model")
            
        except Exception as e:
            print(f" Error loading Hugging Face model: {e}")
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
    
    def setup(self) -> 'RAGSystem':
        """Set up the complete RAG system."""
        # Load and prepare data
        self.data_loader.load_data()
        feature_importance = self.data_loader.prepare_feature_importance()
        fraud_col, id_col = self.data_loader.identify_key_columns()
        
        # Create document creator if not provided
        if self.document_creator is None:
            self.document_creator = DocumentCreator(
                self.data_loader.fraud_df, 
                fraud_col, 
                id_col, 
                feature_importance
            )
        
        # Create documents
        self.documents = self.document_creator.create_documents()
        
        # Build vector database
        self.db = self.vector_db_builder.build_vector_db(self.documents)
        
        # Load language model
        self.llm = self.llm_provider.load_model()
        
        # Create RAG chain
        try:
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.db.as_retriever(search_kwargs={"k": 1}),
                chain_type="stuff"
            )
            print("✅ Built RAG pipeline")
        except Exception as e:
            print(f" Error building RAG pipeline: {e}")
            raise
            
        return self
    
    def query(self, query_text: str) -> str:
        """
        Run a query through the RAG system.
        
        Args:
            query_text: The query to answer
            
        Returns:
            The response from the RAG system
        """
        if self.rag_chain is None:
            raise ValueError("RAG system not set up. Call setup() first.")
            
        print(f"💬 Running query: {query_text}")
        try:
            # Use invoke instead of run (to fix deprecation warning)
            response = self.rag_chain.invoke({"query": query_text})
            return response
        except Exception as e:
            print(f" Error running query: {e}")
            raise
#

def main():
    """Main function to run the RAG system."""
    # File paths
    fraud_path = "/data/processed/Bank_Transaction_Fraud_Detection_Processed.csv"
    importance_path = "/data/processed/feature_importance.csv"
    
    # Create and set up the RAG system
    data_loader = DataLoader(fraud_path, importance_path)
    
    # For smaller models, you can use:
    # llm_provider = LLMProvider(model_name="google/flan-t5-base")
    
    rag_system = RAGSystem(data_loader)
    
    try:
        rag_system.setup()
        
        # Run a test query
        query = "Why is transaction 101 fraudulent?"
        response = rag_system.query(query)
        print("\nExplanation:\n", response)
        
        # You can run more queries here
        # query2 = "What are common fraud patterns?"
        # response2 = rag_system.query(query2)
        # print("\n📢 Explanation:\n", response2)
        
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    main()
