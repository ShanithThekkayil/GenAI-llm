import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv  
load_dotenv()  

# Initialize Azure OpenAI client
# Shanith Thekkayil
def initialize_client():
    """
    Initialize the Azure OpenAI client 
    """
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # API key from .env file
        api_version="2025-01-01-preview",  # API version
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  # endpoint URL  from .env file
    )
    return client

def query_llm(client, user_question, temperature=0.7, top_p=0.95, max_tokens=500):
    """
    Send a question to the LLM and get response
    
    Parameters:
    - client: Azure OpenAI client
    - user_question: The question from user
    - temperature: Controls randomness (0.0 to 1.0)
    - top_p: Controls diversity (0.0 to 1.0)
    - max_tokens: Maximum response length
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",   
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_question}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error occurred: {str(e)}"

def demonstrate_parameter_effects(client, question):
    """
    Demonstrate how different parameters affect the response
    """
    print("=" * 80)
    print("DEMONSTRATING PARAMETER EFFECTS")
    print("=" * 80)
    
    # Different parameter combinations
    parameter_sets = [
        {"temperature": 0.1, "top_p": 0.5, "description": "Low Temperature & Top_p (More focused/deterministic)"},
        {"temperature": 0.7, "top_p": 0.9, "description": "Medium Temperature & Top_p (Balanced)"},
        {"temperature": 1.0, "top_p": 1.0, "description": "High Temperature & Top_p (More creative/random)"}
    ]
    
    for i, params in enumerate(parameter_sets, 1):
        print(f"\n--- Test {i}: {params['description']} ---")
        print(f"Parameters: Temperature={params['temperature']}, Top_p={params['top_p']}")
        
        response = query_llm(
            client, 
            question, 
            temperature=params['temperature'], 
            top_p=params['top_p']
        )
        
        print(f"Response: {response}\n")

def main():
    """
    Main function to run the LLM interaction
    """
    print(" Basic LLM Interaction Assignment")
    print("=" * 50)
    
    # Initialize client
    try:
        client = initialize_client()
        print("✅ Azure OpenAI client initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        print("\n Setup Instructions:")
        print("1. Set environment variables:")
        print("   - AZURE_OPENAI_API_KEY=your_api_key")
        print("   - AZURE_OPENAI_ENDPOINT=your_endpoint_url")
        print("2. Update the model name in the code")
        return
    
    # Get user input
    while True:
        print("\n" + "=" * 50)
        user_question = input("Enter your question (or 'quit' to exit): ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print(" Done!")
            break
            
        if not user_question:
            print(" Please enter a valid question.")
            continue
        
        # Basic query with default parameters
        print(f"\n Sending question: '{user_question}'")
        print("\n Querying LLM with default parameters...")
        
        response = query_llm(client, user_question)
        print(f"\n LLM Response:\n{response}")
        
        # Ask if user wants to see parameter effects
        show_params = input("\nWould you like to see how different parameters affect the response? (y/n): ").strip().lower()
        
        if show_params in ['y', 'yes']:
            demonstrate_parameter_effects(client, user_question)

if __name__ == "__main__":
    main()
