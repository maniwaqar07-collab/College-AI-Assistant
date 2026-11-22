"""
=====================================================
  VERY BASIC RAG (Retrieval-Augmented Generation)
=====================================================

RAG = 3 simple steps:
  1. RETRIEVE : find the pieces of text most related to the question
  2. AUGMENT  : put those pieces into the prompt as "context"
  3. GENERATE : ask the LLM to answer using ONLY that context

Install once:
    pip install groq python-dotenv scikit-learn

Run:
    python rag.py
"""

import os
from dotenv import load_dotenv                                # reads the .env file
from pypdf import PdfReader
from groq import Groq                                         # Groq LLM client
from sklearn.feature_extraction.text import TfidfVectorizer   # turns text into numbers
from sklearn.metrics.pairwise import cosine_similarity        # measures how similar two texts are


# -----------------------------------------------------
# STEP 0: Setup - load the API key and create the LLM client
# -----------------------------------------------------
load_dotenv()  # looks for a .env file in this folder (or parent folders)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"  # any Groq chat model works here


# -----------------------------------------------------
# STEP 1: Our "knowledge base" (the documents)
# In real projects these come from PDFs, websites, databases...
# Here we keep it simple: a list of short text chunks.
# -----------------------------------------------------
documents = [
    "Our college library is open from 8 AM to 8 PM, Monday to Saturday. It is closed on Sundays.",
    "Students can borrow up to 3 books at a time for a period of 14 days.",
    "The late fee for a library book is 10 rupees per day.",
    "The Python programming course is taught by Sir Ahmed every Tuesday and Thursday at 11 AM.",
    "The computer lab is on the second floor and has 40 computers with internet access.",
    "Mid-term exams start on 10th March and final exams start on 15th June.",
    "The college cafeteria serves breakfast from 8 AM to 10 AM and lunch from 12 PM to 3 PM.",
    "To apply for a scholarship, students must have at least 80% marks and submit the form before 30th April.",
    "The college bus leaves the main gate at 4:30 PM every day.",
    "Students must maintain 75% attendance to be allowed to sit in the final exams.",
]
# Read text from our PDF
pdf_path = "Attack_on_Titan_RAG_Study_Guide.pdf"

reader = PdfReader(pdf_path)

pdf_documents = []

for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_documents.append(text)

documents.extend(pdf_documents)
print("PDF pages loaded:", len(pdf_documents))

# -----------------------------------------------------
# STEP 2: Convert documents into vectors (numbers)
# Computers can't compare meaning of words directly, so we turn
# each document into a list of numbers using TF-IDF.
# (Advanced RAG uses "embeddings" from a model - same idea.)
# -----------------------------------------------------
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)


# -----------------------------------------------------
# STEP 3: RETRIEVE - find the most relevant documents
# -----------------------------------------------------
def retrieve(question, top_k=2):
    # Turn the question into a vector using the SAME vectorizer
    question_vector = vectorizer.transform([question])

    # Compare the question with every document (score 0 = unrelated, 1 = very similar)
    scores = cosine_similarity(question_vector, doc_vectors)[0]

    # Sort documents by score (highest first) and keep the top_k
    best_indexes = scores.argsort()[::-1][:top_k]

    results = []
    for i in best_indexes:
        results.append((documents[i], round(float(scores[i]), 3)))
    return results


# -----------------------------------------------------
# STEP 4: AUGMENT + GENERATE - send context + question to the LLM
# -----------------------------------------------------
def ask_llm(question, context_docs):
    # Join the retrieved documents into one block of text
    context = "\n".join("- " + doc for doc, score in context_docs)

    # The prompt tells the LLM to use ONLY our context (this reduces made-up answers)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the given information."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # 0 = focused, consistent answers
    )
    return response.choices[0].message.content


# -----------------------------------------------------
# STEP 5: Simple terminal chat loop
# -----------------------------------------------------
if __name__ == "__main__":
    print("=== Basic RAG Demo (type 'exit' to quit) ===")

    while True:
        question = input("\nAsk a question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        # 1. Retrieve
        retrieved = retrieve(question)
        print("\n[Retrieved documents]")
        for doc, score in retrieved:
            print(f"  (score {score}) {doc}")

        # 2 + 3. Augment and Generate
        answer = ask_llm(question, retrieved)
        print("\n[Answer]")
        print(answer)
