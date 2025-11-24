"""
Enhanced Streamlit web UI for AI Lab IGNFA - Legal RAG System with document management.

This module provides a comprehensive web interface for document processing,
querying, and management using Streamlit.
"""

import sys
import shutil
import tempfile
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from raganything import RAGAnything, get_config
from raganything.smart_processor import SmartProcessor
from raganything.document_tracker import DocumentTracker

# Import prompt functions with fallback
try:
    from raganything.prompt import load_prompt_from_file, get_system_prompt
except ImportError:
    # Fallback: import directly from module
    import raganything.prompt as prompt_module
    load_prompt_from_file = getattr(prompt_module, 'load_prompt_from_file', None)
    get_system_prompt = getattr(prompt_module, 'get_system_prompt', None)
    
    if load_prompt_from_file is None or get_system_prompt is None:
        # Last resort: define minimal versions
        def load_prompt_from_file(file_path: str):
            """Load prompt from file."""
            from pathlib import Path
            prompt_path = Path(file_path)
            if not prompt_path.exists():
                prompt_path = Path.cwd() / file_path
            if not prompt_path.exists():
                prompt_path = Path(__file__).parent.parent.parent / file_path
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8").strip()
            return None
        
        def get_system_prompt(template=None):
            """Get system prompt."""
            return "You are a helpful assistant that answers questions based on provided context from documents."

# Page configuration
st.set_page_config(
    page_title="AI Lab IGNFA - Legal RAG System",
    page_icon="📚",
    layout="wide",
)

# Initialize session state with config defaults
config = get_config()
if "rag" not in st.session_state:
    st.session_state.rag = None
if "processor" not in st.session_state:
    st.session_state.processor = None
if "auto_process" not in st.session_state:
    st.session_state.auto_process = True
if "embedding_model" not in st.session_state:
    config = get_config()
    # Use getattr with default in case Config was initialized before embedding_model was added
    st.session_state.embedding_model = getattr(config, 'embedding_model', 'text-embedding-3-small')
if "llm_model" not in st.session_state:
    st.session_state.llm_model = config.llm_model  # Use config default
if "llm_temperature" not in st.session_state:
    st.session_state.llm_temperature = config.llm_temperature
if "llm_top_p" not in st.session_state:
    st.session_state.llm_top_p = config.llm_top_p
if "llm_max_tokens" not in st.session_state:
    st.session_state.llm_max_tokens = config.llm_max_tokens
if "query_n_results" not in st.session_state:
    st.session_state.query_n_results = config.query_n_results
if "query_max_context_length" not in st.session_state:
    st.session_state.query_max_context_length = config.query_max_context_length


def initialize_components():
    """Initialize AI Lab IGNFA - Legal RAG System and SmartProcessor."""
    if st.session_state.rag is None:
        with st.spinner("Initializing AI Lab IGNFA - Legal RAG System..."):
            st.session_state.rag = RAGAnything(
                llm_model=st.session_state.llm_model,
                llm_temperature=st.session_state.llm_temperature,
                llm_top_p=st.session_state.llm_top_p,
                llm_max_tokens=st.session_state.llm_max_tokens,
            )
            st.session_state.processor = SmartProcessor(
                documents_dir="documents",
                raganything=st.session_state.rag,
            )


def main():
    """Main Streamlit app."""
    st.title("📚 AI Lab IGNFA - Legal RAG System")
    st.markdown("Legal Document Processing and Q&A System")

    # Initialize components
    initialize_components()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # LLM Configuration
        st.subheader("🤖 LLM Configuration")
        llm_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
        # Get default from config (which reads from .env)
        config = get_config()
        default_model = config.llm_model
        current_model_index = (
            llm_models.index(st.session_state.llm_model)
            if st.session_state.llm_model in llm_models
            else (llm_models.index(default_model) if default_model in llm_models else 4)
        )
        llm_model = st.selectbox(
            "LLM Model",
            llm_models,
            index=current_model_index,
            help=f"Select the OpenAI model to use for generating answers (default from .env: {default_model})",
        )
        st.session_state.llm_model = llm_model

        llm_temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            st.session_state.llm_temperature,
            0.1,
            help="Controls randomness: 0 = deterministic, 2 = very creative",
        )
        st.session_state.llm_temperature = llm_temperature

        llm_top_p = st.slider(
            "Top P",
            0.0,
            1.0,
            st.session_state.llm_top_p,
            0.05,
            help="Nucleus sampling: considers tokens with top_p probability mass",
        )
        st.session_state.llm_top_p = llm_top_p

        llm_max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=8000,
            value=st.session_state.llm_max_tokens or 2000,
            step=100,
            help="Maximum tokens in response (None = no limit)",
        )
        st.session_state.llm_max_tokens = llm_max_tokens if llm_max_tokens > 0 else None

        st.divider()
        
        # Query Configuration
        st.subheader("🔍 Query Configuration")
        query_n_results = st.slider(
            "Number of Results",
            1,
            20,
            st.session_state.query_n_results,
            help="Number of context chunks to retrieve",
        )
        st.session_state.query_n_results = query_n_results

        query_max_context_length = st.slider(
            "Max Context Length",
            500,
            8000,
            st.session_state.query_max_context_length,
            100,
            help="Maximum context length in characters",
        )
        st.session_state.query_max_context_length = query_max_context_length

        st.divider()
        
        # Embedding Model Configuration
        st.subheader("🔢 Embedding Model")
        config = get_config()
        embedding_models = [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ]
        # Use getattr with default in case Config was initialized before embedding_model was added
        default_embedding = getattr(config, 'embedding_model', 'text-embedding-3-small')
        current_embedding_index = (
            embedding_models.index(st.session_state.get("embedding_model", default_embedding))
            if st.session_state.get("embedding_model", default_embedding) in embedding_models
            else (embedding_models.index(default_embedding) if default_embedding in embedding_models else 1)
        )
        embedding_model = st.selectbox(
            "Embedding Model",
            embedding_models,
            index=current_embedding_index,
            help=f"Model for generating embeddings (default: {default_embedding})",
        )
        st.session_state.embedding_model = embedding_model
        
        if st.button("📋 List All Embedding Models"):
            st.info("""
**OpenAI Embedding Models:**
- `text-embedding-3-large` - Highest quality, 3072 dimensions
- `text-embedding-3-small` - Balanced, 1536 dimensions (default)
- `text-embedding-ada-002` - Legacy, 1536 dimensions

**Note:** Set `EMBEDDING_MODEL` in `.env` to change the default.
            """)
        
        st.divider()
        
        # Parser Configuration
        st.subheader("📄 Parser Configuration")
        parser = st.selectbox("Parser", ["mineru", "docling"], index=0)
        parse_method = st.selectbox("Parse Method", ["auto", "ocr", "txt"], index=0)
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        
        # Output flag/span files option for MinerU
        output_flag_span = None
        if parser == "mineru":
            config = get_config()
            default_output_flag_span = getattr(config, 'mineru_output_flag_span', True)
            output_flag_span = st.checkbox(
                "Output verification files (span.pdf, layout.pdf, etc.)",
                value=default_output_flag_span,
                help="If checked, MinerU will generate additional files for verification (e.g., span.pdf, layout.pdf). MinerU generates these by default.",
            )
            st.session_state.output_flag_span = output_flag_span

        if st.button("🔄 Reinitialize RAG"):
            with st.spinner("Reinitializing..."):
                # Get output_flag_span setting if using mineru
                parser_kwargs = {}
                if parser == "mineru" and "output_flag_span" in st.session_state:
                    parser_kwargs["output_flag_span"] = st.session_state.output_flag_span
                
                st.session_state.rag = RAGAnything(
                    parser=parser,
                    parse_method=parse_method,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    llm_model=st.session_state.llm_model,
                    llm_temperature=st.session_state.llm_temperature,
                    llm_top_p=st.session_state.llm_top_p,
                    llm_max_tokens=st.session_state.llm_max_tokens,
                    **parser_kwargs
                )
                st.session_state.processor = SmartProcessor(
                    documents_dir="documents",
                    raganything=st.session_state.rag,
                )
            st.success("AI Lab IGNFA - Legal RAG System reinitialized!")
            st.rerun()

        st.divider()
        st.header("📊 Document Status")
        # Ensure processor is initialized before accessing
        if st.session_state.processor is None:
            st.warning("⚠️ Processor not initialized. Please wait...")
            st.stop()
        status = st.session_state.processor.get_document_status()
        st.metric("Total Files", status["total_files"])
        st.metric("Processed", len(status["processed"]))
        st.metric("Unprocessed", len(status["unprocessed"]))

        st.divider()
        st.header("🔄 Auto-Processing")
        auto_process = st.checkbox(
            "Auto-process on changes",
            value=st.session_state.auto_process,
            help="Automatically process new/updated documents when detected",
        )
        st.session_state.auto_process = auto_process

        st.divider()
        st.header("🔄 Process Documents")
        
        # Extract-only option
        extract_only = st.checkbox(
            "Extract text only (skip chunking/embedding)",
            value=False,
            help="Only extract text from documents, skip chunking and embedding. Useful for batch extraction.",
        )
        st.session_state.extract_only = extract_only
        
        force_extract = st.checkbox(
            "Force re-extraction",
            value=False,
            help="Force re-extraction even if extracted file exists",
        )
        st.session_state.force_extract = force_extract
        
        # Force reprocess option
        force_reprocess = st.checkbox(
            "Force reprocess all documents",
            value=False,
            help="If checked, all documents will be reprocessed even if unchanged",
        )
        
        if st.button("🔄 Process All Documents", type="primary"):
            if st.session_state.processor is None:
                st.error("⚠️ Processor not initialized. Please refresh the page.")
                st.stop()
            with st.spinner("Processing all documents..."):
                results = st.session_state.processor.process_all(
                    force_reprocess=force_reprocess,
                    extract_only=st.session_state.get("extract_only", False),
                    force_extract=st.session_state.get("force_extract", False),
                )
                st.success(
                    f"Processed: {len(results['new'])} new, "
                    f"{len(results['updated'])} updated, "
                    f"{len(results['unchanged'])} unchanged, "
                    f"{len(results['removed'])} removed"
                )
                if results.get('errors'):
                    st.warning(f"Errors: {len(results['errors'])} documents had errors")
                st.rerun()
        
        st.divider()
        st.header("🗑️ Reset & Clear")
        
        st.warning("⚠️ This will delete all embeddings and document tracking data!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Reset All Data", type="secondary"):
                if st.session_state.get("confirm_reset", False):
                    # Perform reset
                    if st.session_state.rag is None or st.session_state.processor is None:
                        st.error("⚠️ RAG system or processor not initialized. Please refresh the page.")
                        st.stop()
                    with st.spinner("Clearing all data..."):
                        try:
                            # Clear vector store
                            if hasattr(st.session_state.rag.processor, 'vector_store') and st.session_state.rag.processor.vector_store:
                                st.session_state.rag.processor.vector_store.delete()
                            
                            # Clear document tracker
                            st.session_state.processor.tracker.clear()
                            # Verify tracker file exists after clear
                            tracker_file = st.session_state.processor.tracker.tracker_file
                            if not tracker_file.exists():
                                # Recreate tracker file
                                st.session_state.processor.tracker._save_tracker()
                            
                            st.success("✅ All data cleared! Vector store and document tracker reset.")
                            st.info(f"📄 Tracker file: {tracker_file}")
                            st.session_state.confirm_reset = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during reset: {str(e)}")
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ Click again to confirm reset")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Clear Confirmation"):
                st.session_state.confirm_reset = False
                st.rerun()
        
        if st.session_state.get("confirm_reset", False):
            st.info("💡 Click 'Reset All Data' again to confirm, or 'Clear Confirmation' to cancel")

        st.divider()
        
        # Custom Prompt Configuration
        st.subheader("📝 Custom Prompt")
        config = get_config()
        prompt_file_path = config.prompt_file_path or Path("prompt.md")
        
        # Load current prompt
        current_prompt = None
        if prompt_file_path.exists():
            try:
                current_prompt = prompt_file_path.read_text(encoding="utf-8")
            except Exception as e:
                st.error(f"Error reading prompt file: {e}")
        else:
            # Show default prompt
            current_prompt = get_system_prompt()
            st.info(f"Using default prompt. Create `{prompt_file_path.name}` to customize.")
        
        if current_prompt:
            with st.expander("📄 View/Edit Custom Prompt", expanded=False):
                edited_prompt = st.text_area(
                    "Edit prompt (saved to prompt.md)",
                    value=current_prompt,
                    height=300,
                    help="Edit the system prompt. Save to apply changes.",
                    key="prompt_editor"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Prompt"):
                        try:
                            prompt_file_path.write_text(edited_prompt, encoding="utf-8")
                            st.success(f"Prompt saved to {prompt_file_path.name}")
                            # Reload RAG instance to use new prompt
                            if st.session_state.rag:
                                st.session_state.rag.query_handler.custom_prompt_file = str(prompt_file_path)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving prompt: {e}")
                
                with col2:
                    if st.button("🔄 Reload Prompt"):
                        if prompt_file_path.exists():
                            try:
                                new_prompt = prompt_file_path.read_text(encoding="utf-8")
                                if st.session_state.rag:
                                    st.session_state.rag.query_handler.custom_prompt_file = str(prompt_file_path)
                                st.success("Prompt reloaded!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error reloading prompt: {e}")
                
                st.caption(f"📁 File: {prompt_file_path}")
                if not prompt_file_path.exists():
                    st.warning(f"File `{prompt_file_path.name}` will be created when you save.")

    # Main content area - Chat tab first (default)
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "📊 Status"])

    with tab1:
        st.header("💬 Chat with Documents")

        # Ensure RAG is initialized
        if st.session_state.rag is None:
            st.error("⚠️ RAG system not initialized. Please refresh the page or check configuration.")
            st.stop()

        # Update RAG instance with current LLM settings if changed
        if (
            st.session_state.rag.query_handler.model != st.session_state.llm_model
            or st.session_state.rag.query_handler.temperature != st.session_state.llm_temperature
            or st.session_state.rag.query_handler.top_p != st.session_state.llm_top_p
            or st.session_state.rag.query_handler.max_tokens != st.session_state.llm_max_tokens
        ):
            # Update query handler settings
            st.session_state.rag.query_handler.model = st.session_state.llm_model
            st.session_state.rag.query_handler.temperature = st.session_state.llm_temperature
            st.session_state.rag.query_handler.top_p = st.session_state.llm_top_p
            st.session_state.rag.query_handler.max_tokens = st.session_state.llm_max_tokens

        # Display chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag.query(
                            query=prompt,
                            n_results=st.session_state.query_n_results,
                            max_context_length=st.session_state.query_max_context_length,
                            temperature=st.session_state.llm_temperature,
                            top_p=st.session_state.llm_top_p,
                            max_tokens=st.session_state.llm_max_tokens,
                        )
                        answer = result.get("answer", "I couldn't generate an answer.")
                        sources = result.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander("📚 Sources"):
                                for source in sources:
                                    st.text(f"- {source}")

                        # Add assistant response to history
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": error_msg}
                        )

        # Clear chat button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    with tab2:
        st.header("📄 Document Management")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Upload Document")
            uploaded_file = st.file_uploader(
                "Upload a document to the documents directory",
                type=["pdf", "docx", "pptx", "xlsx", "txt", "md", "jpg", "png", "jpeg"],
                help="Uploaded files will be saved to the documents/ directory",
            )

            if uploaded_file is not None:
                if st.button("💾 Save & Process Document"):
                    documents_dir = Path("documents")
                    documents_dir.mkdir(exist_ok=True)

                    # Save uploaded file
                    file_path = documents_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success(f"File saved to: {file_path}")

                    # Auto-process if enabled
                    if st.session_state.auto_process:
                        if st.session_state.rag is None:
                            st.error("⚠️ RAG system not initialized. Please refresh the page.")
                            st.stop()
                        with st.spinner("Processing document..."):
                            try:
                                result = st.session_state.rag.process_document_complete(
                                    file_path=str(file_path),
                                    doc_id=file_path.stem,
                                    extract_only=st.session_state.get("extract_only", False),
                                    skip_if_extracted_exists=not st.session_state.get("force_extract", False),
                                )

                                st.success(
                                    f"Document processed! Created {result.get('num_chunks', 0)} chunks"
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error processing document: {str(e)}")

        with col2:
            st.subheader("Quick Actions")
            force_reprocess_quick = st.checkbox(
                "Force reprocess",
                value=False,
                key="force_reprocess_quick",
                help="Reprocess all documents even if unchanged",
            )
            if st.button("🔄 Refresh & Process"):
                if st.session_state.processor is None:
                    st.error("⚠️ Processor not initialized. Please refresh the page.")
                    st.stop()
                with st.spinner("Scanning and processing..."):
                    results = st.session_state.processor.process_all(force_reprocess=force_reprocess_quick)
                    st.success(
                        f"Processing complete! "
                        f"New: {len(results['new'])}, "
                        f"Updated: {len(results['updated'])}, "
                        f"Unchanged: {len(results['unchanged'])}"
                    )
                    st.rerun()

            if st.button("📊 Refresh Status"):
                st.rerun()

        st.divider()

        # Document list
        st.subheader("📋 Document List")
        # Ensure processor is initialized
        if st.session_state.processor is None:
            st.error("⚠️ Processor not initialized. Please refresh the page.")
            st.stop()
        status = st.session_state.processor.get_document_status()

        # Processed documents
        if status["processed"]:
            st.markdown("### ✅ Processed Documents")
            for doc in status["processed"]:
                with st.expander(
                    f"📄 {Path(doc['path']).name} - {doc['chunks']} chunks"
                ):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Path:** {doc['path']}")
                        st.write(f"**Document ID:** {doc['doc_id']}")
                        st.write(f"**Chunks:** {doc['chunks']}")
                        st.write(f"**Processed:** {doc.get('processed_time', 'N/A')}")
                    with col2:
                        if st.button("🗑️ Remove", key=f"remove_{doc['path']}"):
                            # Remove file
                            file_path = Path(doc["path"])
                            if file_path.exists():
                                file_path.unlink()
                            # Mark as removed in tracker
                            st.session_state.processor.tracker.mark_removed(doc["path"])
                            st.success("Document removed!")
                            st.rerun()

        # Unprocessed documents
        if status["unprocessed"]:
            st.markdown("### ⏳ Unprocessed Documents")
            for doc in status["unprocessed"]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {Path(doc['path']).name}")
                with col2:
                           if st.button("▶️ Process", key=f"process_{doc['path']}"):
                               if st.session_state.rag is None:
                                   st.error("⚠️ RAG system not initialized. Please refresh the page.")
                                   st.stop()
                               with st.spinner("Processing..."):
                                   try:
                                       # Use options from session state
                                       process_kwargs = {}
                                       if st.session_state.get("output_flag_span") is not None:
                                           process_kwargs["output_flag_span"] = st.session_state.output_flag_span
                                       process_kwargs["extract_only"] = st.session_state.get("extract_only", False)
                                       process_kwargs["skip_if_extracted_exists"] = not st.session_state.get("force_extract", False)
                                       
                                       result = st.session_state.rag.process_document_complete(
                                           file_path=doc["path"],
                                           doc_id=Path(doc["path"]).stem,
                                           **process_kwargs
                                       )
                                       st.success("Processed!")
                                       st.rerun()
                                   except Exception as e:
                                       st.error(f"Error: {str(e)}")

    with tab3:
        st.header("📊 Detailed Status")

        # Ensure processor is initialized
        if st.session_state.processor is None:
            st.error("⚠️ Processor not initialized. Please refresh the page.")
            st.stop()
        status = st.session_state.processor.get_document_status()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", status["total_files"])
        with col2:
            st.metric("Processed", len(status["processed"]))
        with col3:
            st.metric("Unprocessed", len(status["unprocessed"]))
        with col4:
            st.metric("Removed", len(status["removed"]))

        st.divider()

        # Processed documents table
        if status["processed"]:
            st.subheader("✅ Processed Documents")
            import pandas as pd

            processed_data = []
            for doc in status["processed"]:
                processed_data.append(
                    {
                        "File": Path(doc["path"]).name,
                        "Path": doc["path"],
                        "Document ID": doc["doc_id"],
                        "Chunks": doc["chunks"],
                        "Processed": doc.get("processed_time", "N/A")[:19]
                        if doc.get("processed_time")
                        else "N/A",
                    }
                )

            df = pd.DataFrame(processed_data)
            st.dataframe(df, width='stretch')

        # Unprocessed documents
        if status["unprocessed"]:
            st.subheader("⏳ Unprocessed Documents")
            for doc in status["unprocessed"]:
                st.write(f"- {Path(doc['path']).name}")

        # Removed documents
        if status["removed"]:
            st.subheader("🗑️ Removed Documents (tracked but missing)")
            for doc_path in status["removed"]:
                st.write(f"- {Path(doc_path).name}")


if __name__ == "__main__":
    main()
