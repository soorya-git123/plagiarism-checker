from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def check_plagiarism(file1, file2):
    # Read files
    with open(file1, "r", encoding="utf-8") as f:
        text1 = f.read()
    with open(file2, "r", encoding="utf-8") as f:
        text2 = f.read()

    # Convert texts into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    print(f"Similarity Score: {similarity * 100:.2f}%")
    if similarity > 0.70:  # threshold = 70%
        print("⚠️ The documents are highly similar (Possible Plagiarism).")
    else:
        print("✅ The documents are different enough.")

# Example usage
check_plagiarism("sample1.txt", "sample2.txt")
