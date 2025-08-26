import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def initialize_client():
    """Initialize Azure OpenAI client"""
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # API key from .env file
        api_version="2025-01-01-preview",  # API version
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  # endpoint URL  from .env file
    )
    return client

def create_zero_shot_prompt(user_question):
    """
    Create a zero-shot prompt without examples
    """
    prompt = f"""You are an insurance customer service classifier.

Classify the following insurance-related question into one of these categories:
- Claim
- Premium  
- Coverage
- Policy

Question: "{user_question}"

Respond with only the category name (one word).
Category:"""
    
    return prompt

def create_few_shot_prompt(user_question):
    """
    Create a few-shot prompt with examples
    """
    prompt = f"""You are an insurance customer service classifier.

Classify insurance-related questions into one of these categories:
- Claim
- Premium
- Coverage
- Policy

Here are some examples:

Question: "How do I file a claim for my car accident?"
Category: Claim

Question: "What is my monthly premium amount?"
Category: Premium

Question: "Does my policy cover flood damage?"
Category: Coverage

Question: "When does my policy expire?"
Category: Policy

Question: "What documents do I need to submit a claim?"
Category: Claim

Now classify this question:
Question: "{user_question}"

Respond with only the category name (one word).
Category:"""
    
    return prompt

def classify_question(client, prompt, prompt_type):
    """
    Send the classification prompt to LLM and get the category
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=50,    # Short response expected
            top_p=0.9
        )
        
        # Extract the category from response
        category = response.choices[0].message.content.strip()
        
        # Clean up the response (remove "Category:" if present)
        if "Category:" in category:
            category = category.split("Category:")[-1].strip()
        
        # Ensure it's one of the valid categories
        valid_categories = ["Claim", "Premium", "Coverage", "Policy"]
        
        # Find the closest match (case insensitive)
        for valid_cat in valid_categories:
            if valid_cat.lower() in category.lower():
                return valid_cat
        
        return category  # Return as-is if no exact match
        
    except Exception as e:
        return f"Error: {str(e)}"

def demonstrate_classification(client, user_question):
    """
    Demonstrate both zero-shot and few-shot classification
    """
    print(f"\n{'='*80}")
    print(f"🔍 CLASSIFYING QUESTION: '{user_question}'")
    print(f"{'='*80}")
    
    # Zero-shot classification
    print("\n📋 ZERO-SHOT PROMPTING")
    print("-" * 50)
    zero_shot_prompt = create_zero_shot_prompt(user_question)
    print("Prompt structure:")
    print("- Direct instruction to classify")
    print("- No examples provided")
    print("- Relies on LLM's general knowledge")
    
    zero_shot_result = classify_question(client, zero_shot_prompt, "Zero-shot")
    print(f"\n✅ Zero-shot Result: {zero_shot_result}")
    
    # Few-shot classification  
    print("\n📚 FEW-SHOT PROMPTING")
    print("-" * 50)
    few_shot_prompt = create_few_shot_prompt(user_question)
    print("Prompt structure:")
    print("- Includes 5 example question-category pairs")
    print("- Shows the LLM the expected format")
    print("- Provides context and pattern recognition")
    
    few_shot_result = classify_question(client, few_shot_prompt, "Few-shot")
    print(f"\n✅ Few-shot Result: {few_shot_result}")
    
    # Comparison
    print(f"\n🔄 COMPARISON")
    print("-" * 50)
    print(f"Zero-shot: {zero_shot_result}")
    print(f"Few-shot:  {few_shot_result}")
    
    if zero_shot_result == few_shot_result:
        print("✅ Both methods agree!")
    else:
        print("⚠️  Methods disagree - Few-shot is typically more reliable")
    
    return zero_shot_result, few_shot_result

def show_prompt_examples(client):
    """
    Show the actual prompts being used for transparency
    """
    print("\n📝 PROMPT EXAMPLES")
    print("=" * 80)
    
    example_question = "What documents are required to file a claim?"
    
    print("\n1️⃣ ZERO-SHOT PROMPT:")
    print("-" * 40)
    zero_prompt = create_zero_shot_prompt(example_question)
    print(zero_prompt)
    
    print("\n2️⃣ FEW-SHOT PROMPT:")
    print("-" * 40)
    few_prompt = create_few_shot_prompt(example_question)
    print(few_prompt)

def test_sample_questions(client):
    """
    Test the classifier with sample insurance questions
    """
    sample_questions = [
        "What documents are required to file a claim?",
        "How much will my premium increase after an accident?",
        "Does my insurance cover rental cars?",
        "When does my policy renew?",
        "How do I report a stolen vehicle?",
        "What's the deductible for my coverage?",
        "Can I change my policy details online?",
        "How long does claim processing take?"
    ]
    
    print("\n🧪 TESTING WITH SAMPLE QUESTIONS")
    print("=" * 80)
    
    results = []
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n--- Test {i} ---")
        zero_shot, few_shot = demonstrate_classification(client, question)
        results.append({
            'question': question,
            'zero_shot': zero_shot,
            'few_shot': few_shot,
            'agreement': zero_shot == few_shot
        })
    
    # Summary
    print(f"\n📊 SUMMARY OF RESULTS")
    print("=" * 80)
    agreements = sum(1 for r in results if r['agreement'])
    print(f"Total questions tested: {len(results)}")
    print(f"Zero-shot and Few-shot agreed: {agreements}/{len(results)} times")
    print(f"Agreement rate: {(agreements/len(results)*100):.1f}%")
    
    return results

def main():
    """
    Main function for insurance question classification
    """
    print("🏥 Assignment 2: Insurance Question Classification")
    print("🎯 Goal: Compare Zero-shot vs Few-shot Prompting")
    print("=" * 80)
    
    # Initialize client
    try:
        client = initialize_client()
        print("✅ Azure OpenAI client initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Show available categories
    print(f"\n📂 CLASSIFICATION CATEGORIES:")
    categories = ["Claim", "Premium", "Coverage", "Policy"]
    for cat in categories:
        descriptions = {
            "Claim": "Questions about filing, processing, or status of insurance claims",
            "Premium": "Questions about payment amounts, billing, or premium calculations", 
            "Coverage": "Questions about what is covered, deductibles, or coverage limits",
            "Policy": "Questions about policy details, renewal, changes, or general policy info"
        }
        print(f"• {cat}: {descriptions[cat]}")
    
    while True:
        print(f"\n{'='*80}")
        print("OPTIONS:")
        print("1. Classify your own question")
        print("2. View prompt examples") 
        print("3. Test with sample questions")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            user_question = input("\nEnter an insurance-related question: ").strip()
            
            if not user_question:
                print("❌ Please enter a valid question.")
                continue
                
            demonstrate_classification(client, user_question)
        
        elif choice == "2":
            show_prompt_examples(client)
        
        elif choice == "3":
            test_sample_questions(client)
        
        elif choice == "4":
            print("👋 Goodbye! Happy learning about prompt engineering!")
            break
        
        else:
            print("❌ Invalid option. Please choose 1-4.")

if __name__ == "__main__":
    main()

