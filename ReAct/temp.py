from history import collection

if __name__ == "__main__":
    def history():
        documents = collection.find(
            {"role": {"$in": ["user", "assistant"]}, "content": {"$exists": True}}
        ).sort("_id", -1).limit(10)
        formatted_output = "Last 5 pairs user request/your answer\n"
        user_query = None
        for doc in reversed(documents.to_list()):
            if doc["role"] == "user":
                user_query = doc["content"]
            elif doc["role"] == "assistant" and user_query:
                formatted_output += f"User query: {user_query}\nYou answer: {doc['content']}\n\n"
                user_query = None
        return formatted_output
    print(history())