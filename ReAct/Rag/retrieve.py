from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from ReAct.API import OpenAI
import numpy as np
collection = MongoClient("mongodb://localhost:27017/")["chatbot_database"]["chat_history"]
embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OpenAI)

def retrieve_similar_docs(query, n=5,collection=collection,embeddings=embeddings):

    docs = list(collection.find({}, {"_id": 0,"content": 1, "emb": 1}))
    doc_vectors = np.array([doc["emb"] for doc in docs])


    query_vector = np.array(embeddings.embed_query(query))
    query_vector = query_vector / np.linalg.norm(query_vector)
    doc_vectors = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)

    similarities = np.dot(doc_vectors, query_vector)

    # Получаем N наиболее похожих документов
    top_indices = similarities.argsort()[-n:][::-1]  # Индексы N лучших

    return [docs[i]["content"] for i in top_indices]

if __name__ == "__main__":
    query = "When Washington was President"
    top_docs = retrieve_similar_docs(query, n=5)
    for i,doc in enumerate(top_docs):
        print(f"number {i}, Content: {doc}")
        print("===========")

