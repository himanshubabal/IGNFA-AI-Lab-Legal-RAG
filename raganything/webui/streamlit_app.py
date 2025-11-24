"""
Enhanced Streamlit web UI for RAG-Anything with document management.

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
from raganything import RAGAnything
from raganything.smart_processor import SmartProcessor
from raganything.document_tracker import DocumentTracker

# Page configuration
st.set_page_config(
    page_title="RAG-Anything",
    page_icon="📚",
    layout="wide",
)

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "processor" not in st.session_state:
    st.session_state.processor = None
if "auto_process" not in st.session_state:
    st.session_state.auto_process = True


def initialize_components():
    """Initialize RAG-Anything and SmartProcessor."""
    if st.session_state.rag is None:
        with st.spinner("Initializing RAG-Anything..."):
            st.session_state.rag = RAGAnything()
            st.session_state.processor = SmartProcessor(
                documents_dir="documents",
                raganything=st.session_state.rag,
            )


def main():
    """Main Streamlit app."""
    st.title("📚 RAG-Anything")
    st.markdown("All-in-One RAG Framework for Multimodal Document Processing")

    # Initialize components
    initialize_components()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        parser = st.selectbox("Parser", ["mineru", "docling"], index=0)
        parse_method = st.selectbox("Parse Method", ["auto", "ocr", "txt"], index=0)
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)

        if st.button("🔄 Reinitialize RAG"):
            with st.spinner("Reinitializing..."):
                st.session_state.rag = RAGAnything(
                    parser=parser,
                    parse_method=parse_method,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                st.session_state.processor = SmartProcessor(
                    documents_dir="documents",
                    raganything=st.session_state.rag,
                )
            st.success("RAG-Anything reinitialized!")
            st.rerun()

        st.divider()
        st.header("📊 Document Status")
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

        if st.button("🔄 Process All Documents"):
            with st.spinner("Processing all documents..."):
                results = st.session_state.processor.process_all()
                st.success(
                    f"Processed: {len(results['new'])} new, "
                    f"{len(results['updated'])} updated, "
                    f"{len(results['removed'])} removed"
                )
                st.rerun()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 Documents", "💬 Chat", "📊 Status"])

    with tab1:
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
                        with st.spinner("Processing document..."):
                            try:
                                result = st.session_state.rag.process_document_complete(
                                    file_path=str(file_path),
                                    doc_id=file_path.stem,
                                )

                                st.success(
                                    f"Document processed! Created {result.get('num_chunks', 0)} chunks"
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error processing document: {str(e)}")

        with col2:
            st.subheader("Quick Actions")
            if st.button("🔄 Refresh & Process"):
                with st.spinner("Scanning and processing..."):
                    results = st.session_state.processor.process_all()
                    st.success("Processing complete!")
                    st.rerun()

            if st.button("📊 Refresh Status"):
                st.rerun()

        st.divider()

        # Document list
        st.subheader("📋 Document List")
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
                        with st.spinner("Processing..."):
                            try:
                                result = st.session_state.rag.process_document_complete(
                                    file_path=doc["path"],
                                    doc_id=Path(doc["path"]).stem,
                                )
                                st.success("Processed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

    with tab2:
        st.header("💬 Chat with Documents")

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
                        result = st.session_state.rag.query(prompt, n_results=5)
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
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    with tab3:
        st.header("📊 Detailed Status")

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
            st.dataframe(df, use_container_width=True)

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
