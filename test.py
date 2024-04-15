from llm_chains import load_vectordb, create_embeddings
from database_operations import init_db,load_last_k_text_messages, save_text_message, save_image_message, save_audio_message, load_messages, get_all_chat_history_ids, delete_chat_history

if __name__ == "__main__":
    init_db()
    vector_db = load_vectordb(create_embeddings())
    output = vector_db.similarity_search("HoVer dataset")
    print(output)