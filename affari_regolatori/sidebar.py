import os
import streamlit as st
import shutil

import shared_config
import ingestion
import chat_history
import retrieval

def show_sidebar(bedrock_client):
    with st.sidebar:
        st.header("Configuration")

        # --- Knowledge Base Section ---
        st.subheader("Knowledge Base")
        uploaded_files = st.file_uploader("Carica PDF per RAG", type="pdf", accept_multiple_files=True)

        col1_kb, col2_kb = st.columns(2)
        
        # 1. Definisci i bottoni dentro le colonne
        with col1_kb:
            btn_crea = st.button("Crea Indice", help="Esegue Smart Chunking ed Embedding sui file caricati")
            
        with col2_kb:
            btn_elimina = st.button("Elimina Indice")

        # 2. Esegui la logica FUORI dalle colonne
        if btn_crea:
            if uploaded_files:
                if os.path.exists(shared_config.FAISS_INDEX_PATH):
                    try:
                        shutil.rmtree(shared_config.FAISS_INDEX_PATH)
                        st.info("Vecchio indice rimosso.")
                    except Exception as e:
                        st.error(f"Errore rimozione vecchio indice: {e}")
                
                chunks = ingestion.read_and_split_uploaded_pdfs(uploaded_files, bedrock_client)
                if chunks:
                    ingestion.create_vector_store(chunks, bedrock_client)
                    file_names = [f.name for f in uploaded_files]
                    ingestion.save_index_metadata(file_names)
                    
                    # ---> AGGIUNTA FONDAMENTALE <---
                    # Svuota la cache in modo che il retriever si accorga dei nuovi file
                    retrieval.get_retriever.clear()
                    
                    st.rerun()
            else:
                st.warning("Carica almeno un PDF.")

        if btn_elimina:
            if os.path.exists(shared_config.FAISS_INDEX_PATH):
                try:
                    shutil.rmtree(shared_config.FAISS_INDEX_PATH)
                    
                    # ---> AGGIUNTA FONDAMENTALE <---
                    # Svuota la cache anche quando elimini l'indice
                    retrieval.get_retriever.clear()
                    
                    st.success("Indice eliminato.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Impossibile eliminare l'indice: {e}")

        # --- Display Index Status ---
        st.subheader("Stato RAG")
        
        # Il nuovo controllo sul file fisico
        faiss_file = os.path.join(shared_config.FAISS_INDEX_PATH, "index.faiss")
        
        if os.path.exists(faiss_file):
            st.success("Indice ibrido pronto (RAG Attivo)")
            source_files = ingestion.load_index_metadata() 
            if source_files:
                with st.expander("File sorgente nell'indice:"):
                    for f_name in source_files:
                        st.markdown(f"- {f_name}")
        else:
            st.warning("Nessun indice (RAG Disattivo)")

        st.divider()

        # --- Chat History Section (Invariata) ---
        st.subheader("Chat History")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_chat_file" not in st.session_state:
            st.session_state.current_chat_file = None 
        if "chat_name_input" not in st.session_state:
            st.session_state.chat_name_input = "Nuova Chat" 

        col1_hist, col2_hist = st.columns(2)
        with col1_hist:
            is_new_chat = (st.session_state.current_chat_file is None)
            
            if st.button("New Chat", disabled=is_new_chat):
                st.session_state.messages = []
                st.session_state.current_chat_file = None
                st.session_state.chat_name_input = "Nuova Chat"
                st.rerun() 

        with col2_hist:
            if st.button("Save Chat"):
                if not st.session_state.messages:
                    st.warning("Nessuna conversazione da salvare.")
                else:
                    friendly_name = st.session_state.chat_name_input.strip()

                    if friendly_name == "Nuova Chat" or not friendly_name:
                        st.sidebar.warning("Inserisci un nome valido per la chat.")
                    else:
                        st.session_state.chat_name_input = friendly_name
                        new_filename = chat_history.sanitize_filename(friendly_name)
                        old_filename = st.session_state.current_chat_file

                        saved_file = chat_history.save_chat_history(
                            st.session_state.messages, 
                            filename=new_filename 
                        )
                        
                        if saved_file:
                            is_rename = (old_filename is not None) and (old_filename != new_filename)
                            if is_rename:
                                chat_history.delete_chat_history(old_filename)
                            st.session_state.current_chat_file = new_filename
                            st.rerun() 

        saved_chat_files = chat_history.get_saved_chats()
        
        try:
            current_chat_index = saved_chat_files.index(st.session_state.current_chat_file) if st.session_state.current_chat_file in saved_chat_files else 0
        except ValueError:
            current_chat_index = 0 

        if saved_chat_files:
            chat_options = {chat_history.desanitize_filename(f): f for f in saved_chat_files}
            friendly_name_options = list(chat_options.keys())
            current_friendly_name = chat_history.desanitize_filename(st.session_state.current_chat_file) if st.session_state.current_chat_file else None
            
            try:
                current_chat_index = friendly_name_options.index(current_friendly_name) if current_friendly_name in friendly_name_options else 0
            except ValueError:
                current_chat_index = 0

            selected_friendly_name = st.selectbox(
                "Load a saved chat:",
                options=friendly_name_options, 
                index=current_chat_index, 
                key="chat_selector" 
            )
            
            if st.button("Load Chat"):
                filename_to_load = chat_options[selected_friendly_name]
                st.session_state.messages = chat_history.load_chat_history(filename_to_load)
                st.session_state.current_chat_file = filename_to_load
                st.session_state.chat_name_input = selected_friendly_name 
                st.rerun() 
        else:
            st.write("No saved chats yet.")

        st.text_input("Nome Chat:", key="chat_name_input")