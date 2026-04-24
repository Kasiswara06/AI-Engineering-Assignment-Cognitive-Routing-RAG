# phase1_router.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Initialize embedding model (offline)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Improved bot personas (important for better matching)
bot_personas = {
    "Bot_A": "AI, OpenAI, machine learning, crypto, automation, future technology, Elon Musk, innovation, AI replacing jobs",

    "Bot_B": "AI risks, job loss, tech monopolies, capitalism criticism, privacy concerns, social media harm, big tech criticism",

    "Bot_C": "stock market, trading, finance, investments, interest rates, ROI, economic impact, financial analysis"
}

texts = list(bot_personas.values())
ids = list(bot_personas.keys())

# Create FAISS vector store WITH metadata (important)
vector_store = FAISS.from_texts(
    texts,
    embedding_model,
    metadatas=[{"id": i} for i in ids]
)


# Routing function using FAISS similarity search
def route_post_to_bots(post_content: str, threshold: float = 0.3):
    results = vector_store.similarity_search_with_score(post_content, k=3)

    matched_bots = []

    for doc, score in results:
        # Convert distance → similarity (FAISS returns distance)
        similarity = 1 / (1 + score)

        if similarity > threshold:
            matched_bots.append({
                "bot_id": doc.metadata["id"],
                "similarity": round(similarity, 3)
            })

    return matched_bots


# Test run
if __name__ == "__main__":
    test_post = "OpenAI released a new AI model that may replace developers."

    print("\nInput Post:")
    print(test_post)

    print("\nMatched Bots:")
    results = route_post_to_bots(test_post)

    if not results:
        print("No bots matched. Try lowering threshold.")
    else:
        for bot in results:
            print(bot)