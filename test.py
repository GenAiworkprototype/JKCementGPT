import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from cementGPT_llm_chat.src.document_ingestion.data_ingestion import ChatIngestor
from cementGPT_llm_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def test_document_ingestion_and_rag():
    try:
        test_file=["/GenAI Hackathon/2.0/data/AgenticRAGRedefiningRetrieval-AugmentedGenerationforAdaptiveIntelligence.pdf"]

        uploaded_files = []

        for file_path in test_file:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File not found: {file_path}")

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

        # Build index using single-module ChatIngestor
        ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
        # Option 1: Use similarity search(default)
        # _=ci.build_index_from_files(uploaded_files, chunk_size=200, overlap=20, k=5)

        # Option 2: Use MMR search(maximal marginal relevance) for more diverse results
        retriever = ci.built_retriver(
            uploaded_files,
            chunk_size=200,
            chunk_overlap=20,
            k=5,                          # Final number of documents to return
            search_type="mmr",            # Use MMR search instead of similarity search
            fetch_k=20,                   # Fetch 20 candidates before MMR filtering
            lambda_mult=0.5,              # 0.5 = balance between diversity and relevance
        )

        # Close file handles
        for f in uploaded_files:
            try:
                f.close()
            except Exception as e:
                pass

        session_id = ci.session_id
        index_dir = os.path.join("faiss_index", session_id)

        # Load RAG with MMR search
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir, 
            k=5, 
            index_name=os.getenv("FAISS_INDEX_NAME", "index"),
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )
        
        # Interactive multi-turn chat loop
        chat_history = []
        print("\nType 'exit' to quit the chat.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("Goodbye!")
                break

            answer = rag.invoke(user_input=user_input, chat_history=chat_history)
            print("Assistant:", answer)

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_document_ingestion_and_rag()