import os
import json
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime

# Load environment variables
load_dotenv()

class SimpleVectorDB:
    """Simple in-memory vector database for storing embeddings"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.metadata = []
    
    def add_document(self, chunks, embeddings, metadata):
        """Add document chunks with embeddings to the database"""
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding, top_k=2):
        """Search for most similar chunks"""
        if not self.embeddings:
            return []
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'similarity': similarities[idx],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def save(self, filename):
        """Save the database to file"""
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filename):
        """Load the database from file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            return True
        except FileNotFoundError:
            return False

class RAGSystem:
    """Retrieval Augmented Generation System"""
    
    def __init__(self):
        self.client = self.initialize_client()
        self.vector_db = SimpleVectorDB()
        
    def initialize_client(self):
        """Initialize Azure OpenAI client"""
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-01-01-preview",   
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > chunk_size * 0.7:  # If boundary is reasonably close
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
                
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter very short chunks
    
    def get_embedding(self, text):
        """Get embedding for text using Azure OpenAI"""
        try:
            # Clean the text
            text = text.replace("\n", " ").strip()
            
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",  # Update this to your embedding model deployment
                input=text
            )
            
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def process_documents(self, pdf_files):
        """Process PDF documents and store in vector database"""
        print("📚 PROCESSING DOCUMENTS")
        print("=" * 50)
        
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        
        for pdf_file in pdf_files:
            print(f"\n📖 Processing: {pdf_file}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                print(f"❌ Failed to extract text from {pdf_file}")
                continue
                
            print(f"✅ Extracted {len(text)} characters")
            
            # Chunk text
            chunks = self.chunk_text(text)
            print(f"✅ Created {len(chunks)} chunks")
            
            # Create embeddings
            print("🔄 Creating embeddings...")
            doc_embeddings = []
            doc_metadata = []
            
            for i, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)
                if embedding:
                    doc_embeddings.append(embedding)
                    doc_metadata.append({
                        'source': pdf_file,
                        'chunk_id': i,
                        'chunk_size': len(chunk),
                        'processed_at': datetime.now().isoformat()
                    })
                
                if (i + 1) % 5 == 0:  # Progress indicator
                    print(f"   Processed {i + 1}/{len(chunks)} chunks...")
            
            print(f"✅ Created {len(doc_embeddings)} embeddings")
            
            # Store in database
            self.vector_db.add_document(chunks, doc_embeddings, doc_metadata)
            
            all_chunks.extend(chunks)
            all_embeddings.extend(doc_embeddings)
            all_metadata.extend(doc_metadata)
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Total embeddings: {len(all_embeddings)}")
        
        # Save the database
        self.vector_db.save("vector_db.pkl")
        print("💾 Vector database saved to 'vector_db.pkl'")
    
    def retrieve_relevant_chunks(self, question, top_k=2):
        """Retrieve most relevant chunks for a question"""
        print(f"\n🔍 RETRIEVING RELEVANT CHUNKS")
        print("-" * 40)
        
        # Get question embedding
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            print("❌ Failed to create question embedding")
            return []
        
        # Search in vector database
        results = self.vector_db.search(question_embedding, top_k=top_k)
        
        print(f"Found {len(results)} relevant chunks:")
        for i, result in enumerate(results, 1):
            print(f"\n📄 Chunk {i} (Similarity: {result['similarity']:.3f})")
            print(f"Source: {result['metadata']['source']}")
            print(f"Preview: {result['chunk'][:150]}...")
        
        return results
    
    def generate_rag_response(self, question, retrieved_chunks):
        """Generate response using retrieved chunks as context"""
        print(f"\n🤖 GENERATING RAG RESPONSE")
        print("-" * 40)
        
        # Build context from retrieved chunks
        context = ""
        for i, chunk_data in enumerate(retrieved_chunks, 1):
            context += f"\n--- Context {i} ---\n"
            context += chunk_data['chunk']
            context += f"\n(Source: {chunk_data['metadata']['source']})\n"
        
        # Create RAG prompt
        rag_prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context documents. 

If the answer is not found in the context, say "I cannot find this information in the provided documents."

Context Documents:
{context}

Question: {question}

Answer based on the context:"""
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided context."},
                    {"role": "user", "content": rag_prompt}
                ],
                temperature=0.3,  # Lower temperature for factual responses
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_baseline_response(self, question):
        """Generate response without RAG for comparison"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating baseline response: {e}"
    
    def query_rag_system(self, question):
        """Complete RAG pipeline: retrieve + generate"""
        print(f"\n{'='*80}")
        print(f"❓ QUESTION: {question}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(question, top_k=2)
        
        if not retrieved_chunks:
            print("❌ No relevant chunks found")
            return
        
        # Step 2: Generate RAG response
        rag_response = self.generate_rag_response(question, retrieved_chunks)
        
        # Step 3: Generate baseline response for comparison
        print(f"\n📊 GENERATING BASELINE RESPONSE (No RAG)")
        print("-" * 40)
        baseline_response = self.generate_baseline_response(question)
        
        # Display results
        print(f"\n📋 RESULTS COMPARISON")
        print("=" * 80)
        
        print(f"\n🔍 RAG RESPONSE (With Context):")
        print("-" * 40)
        print(rag_response)
        
        print(f"\n🤖 BASELINE RESPONSE (Without Context):")
        print("-" * 40)
        print(baseline_response)
        
        print(f"\n💡 ANALYSIS:")
        print("-" * 40)
        if len(rag_response) > len(baseline_response):
            print("✅ RAG response is more detailed")
        if "cannot find" not in rag_response.lower():
            print("✅ RAG found relevant information")
        if any(chunk['similarity'] > 0.7 for chunk in retrieved_chunks):
            print("✅ High similarity chunks retrieved")
        
        return {
            'question': question,
            'rag_response': rag_response,
            'baseline_response': baseline_response,
            'retrieved_chunks': retrieved_chunks
        }



def main():
    """Main function for RAG system"""
    print("🔄 Assignment 3: Retrieval Augmented Generation (RAG)")
    print("🎯 Goal: Improve LLM answers using document retrieval")
    print("=" * 80)
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Check if vector database already exists
    if rag_system.vector_db.load("vector_db.pkl"):
        print("✅ Loaded existing vector database!")
        print("📊 Database stats:")
        print(f"   Chunks: {len(rag_system.vector_db.chunks)}")
        print(f"   Embeddings: {len(rag_system.vector_db.embeddings)}")
    else:
        print("📂 No existing database found. Let's process your PDF documents...")
        
        print("\nEnter your PDF file paths (one per line, empty line to finish):")
        pdf_files = []
        while True:
            path = input("PDF path: ").strip()
            if not path:
                break
            if os.path.exists(path) and path.endswith('.pdf'):
                pdf_files.append(path)
                print(f"✅ Added {path}")
            else:
                print(f"❌ File not found or not a PDF: {path}")
        
        if pdf_files:
            rag_system.process_documents(pdf_files)
        else:
            print("❌ No valid PDF files provided")
            return
    
    # Interactive query loop
    print(f"\n{'='*80}")
    print("🔍 RAG SYSTEM READY! Ask questions about your documents.")
    print("💡 Sample questions to try:")
    print("   • How many paid leaves do we get per year?")
    print("   • What are the system requirements?")
    print("   • What is the work from home policy?")
    print("   • How do I contact technical support?")
    print("='*80")
    
    while True:
        print(f"\n{'='*80}")
        question = input("Enter your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not question:
            print("❌ Please enter a valid question.")
            continue
        
        # Query the RAG system
        rag_system.query_rag_system(question)

if __name__ == "__main__":
    # Install required packages reminder
    print("📦 Required packages: PyPDF2, scikit-learn, numpy")
    print("Install with: pip install PyPDF2 scikit-learn numpy")
    print("-" * 60)
    
    main()
