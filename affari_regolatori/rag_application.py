import os
import streamlit as st

import shared_config
import aws_clients
import retrieval
import response_generation
import sidebar

if not shared_config.INFERENCE_PROFILE:
    st.error("Missing Environment Variable: INFERENCE_PROFILE")
    st.stop()

# Inizializza i client e carica l'indice in cache
bedrock_client = aws_clients.get_bedrock_client()
rag_retriever = retrieval.get_retriever(bedrock_client)

# --- Streamlit App UI ---
st.set_page_config(page_title="Edison Legal RAG", layout="wide")
st.title("📄 Document RAG: Affari Regolatori")

# --- Sidebar --- 
# Passiamo bedrock_client invece di embeddings
sidebar.show_sidebar(bedrock_client)

# --- Main Chat Interface ---
st.header(f"💬 Chat with {shared_config.MODEL_NAME}")

# Verifica se l'indice FAISS esiste (ora cerchiamo il file index.faiss nella cartella)
faiss_file_path = os.path.join(shared_config.FAISS_INDEX_PATH, "index.faiss")
index_exists = os.path.exists(faiss_file_path)

if index_exists:
    st.caption("✅ Modalità RAG Ibrida (FAISS + BM25): Le risposte useranno le normative caricate.")
else:
    st.caption("⚪ Modalità Standard: Nessun indice trovato. Il modello risponderà senza RAG.")

# Mostra i messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Mostra le fonti se presenti
        if "sources" in message and isinstance(message["sources"], list):
            with st.expander("Mostra Documenti Recuperati"):
                 for i, doc_text in enumerate(message["sources"]):
                     st.info(f"Riferimento {i+1}:\n\n{doc_text[:500]}...")

# Input dell'utente
if prompt := st.chat_input("Poni un quesito legale o operativo..."):
    # Aggiungi e mostra il messaggio dell'utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Genera la risposta dell'assistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = [] 

        if index_exists:
            # --- RAG MODE ---
            with st.spinner("Espansione query e Ricerca Ibrida (RRF) in corso..."):
                # Recupera il contesto e la query espansa
                sources, search_query = rag_retriever.search(prompt)

            if sources:
                st.toast(f"Trovati {len(sources)} frammenti rilevanti per: '{search_query}'")
                
                # Generazione risposta basata sul contesto
                response_generator = response_generation.generate_rag_response(
                    bedrock_client, 
                    sources, 
                    st.session_state.messages
                )
                full_response = st.write_stream(response_generator)
            else:
                full_response = "Non ho trovato documenti normativi rilevanti per questa domanda."
                message_placeholder.warning(full_response)
        else:
            # --- NO-RAG MODE ---
            response_generator = response_generation.generate_direct_response(
                bedrock_client, 
                st.session_state.messages
            )
            full_response = st.write_stream(response_generator)

    # Salva la risposta e le fonti nello storico della sessione
    assistant_message = {"role": "assistant", "content": full_response}
    if sources: 
        assistant_message["sources"] = sources
    st.session_state.messages.append(assistant_message)

    # Forza il rerun per aggiornare l'UI con gli expander delle fonti
    if sources:
        st.rerun()