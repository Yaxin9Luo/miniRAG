import pandas as pd
import os
import re

def extract_answer(text):
    """Extract answer option from model output"""
    if pd.isna(text):
        return None
    
    # Common patterns to find the answer
    patterns = [
        r'^([A-D])',  # Single letter at start
        r'(?:the\s+)?answer is:?\s*([A-D])',  # The answer is X
        r'(?:the\s+)?correct answer is:?\s*([A-D])',  # The correct answer is X
        r'option\s*([A-D])\s*is correct',  # Option X is correct
        r'([A-D])\s*is the correct answer',  # X is the correct answer
        r'select\s*(?:option)?\s*([A-D])',  # Select (option) X
        r'choose\s*(?:option)?\s*([A-D])',  # Choose (option) X
        r'answer:\s*([A-D])',  # Answer: X
        r'correct:\s*([A-D])',  # Correct: X
        r'final answer:?\s*([A-D])'  # Final answer: X
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None

def calculate_accuracy(csv_file):
    """Calculate accuracy for a dataset"""
    df = pd.read_csv(csv_file)
    
    # Get dataset name
    dataset_name = os.path.basename(csv_file).replace('_output.csv', '')
    
    # Convert Gold Answer to uppercase
    df['Gold Answer'] = df['Gold Answer'].str.upper()
    
    # Extract predicted answers
    df['predicted_answer'] = df['minirag'].apply(extract_answer)
    
    # Calculate accuracy
    valid_predictions = df['predicted_answer'].notna()
    if valid_predictions.sum() == 0:
        print(f"No valid predictions found in {dataset_name} dataset")
        return 0.0
    
    correct = (df[valid_predictions]['Gold Answer'] == df[valid_predictions]['predicted_answer']).sum()
    total = valid_predictions.sum()
    accuracy = correct / total
    
    print(f"{dataset_name} dataset accuracy: {accuracy:.2%}")
    print(f"Total samples: {len(df)}, Valid predictions: {total}, Correct predictions: {correct}")
    
    # Print answer distribution
    print("\nPredicted answer distribution:")
    answer_dist = df['predicted_answer'].value_counts()
    for option in ['A', 'B', 'C', 'D']:
        count = answer_dist.get(option, 0)
        print(f"Option {option}: {count:4d} times ({count/len(df):6.1%})")
    
    # Print correct answer distribution
    print("\nGold answer distribution:")
    gold_dist = df['Gold Answer'].value_counts()
    for option in ['A', 'B', 'C', 'D']:
        count = gold_dist.get(option, 0)
        print(f"Option {option}: {count:4d} times ({count/len(df):6.1%})")
    
    return accuracy

def main():
    # Set logs directory path
    logs_dir = "./logs"
    
    # Calculate accuracy for all datasets
    datasets = ['medmcqa', 'mmlu', 'arc']
    results = {}
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        print("-" * 50)
        csv_file = os.path.join(logs_dir, f"{dataset}_output.csv")
        if os.path.exists(csv_file):
            accuracy = calculate_accuracy(csv_file)
            results[dataset] = accuracy
        else:
            print(f"Warning: {csv_file} does not exist")
    
    # Print overall results
    print("\nOverall Results Summary:")
    print("-" * 30)
    for dataset, accuracy in results.items():
        print(f"{dataset}: {accuracy:.2%}")

if __name__ == "__main__":
    main() 