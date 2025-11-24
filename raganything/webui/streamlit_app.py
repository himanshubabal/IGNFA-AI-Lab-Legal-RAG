"""
Streamlit web UI for RAG-Anything.

This module provides a simple web interface for document processing
and querying using Streamlit.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from raganything import RAGAnything

# Page configuration
st.set_page_config(
    page_title="RAG-Anything",
    page_icon="📚",
    layout="wide",
)

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def initialize_rag():
    """Initialize RAG-Anything instance."""
    if st.session_state.rag is None:
        with st.spinner("Initializing RAG-Anything..."):
            st.session_state.rag = RAGAnything()


def main():
    """Main Streamlit app."""
    st.title("📚 RAG-Anything")
    st.markdown("All-in-One RAG Framework for Multimodal Document Processing")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        parser = st.selectbox("Parser", ["mineru", "docling"], index=0)
        parse_method = st.selectbox("Parse Method", ["auto", "ocr", "txt"], index=0)
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)

        if st.button("Initialize RAG"):
            with st.spinner("Initializing..."):
                st.session_state.rag = RAGAnything(
                    parser=parser,
                    parse_method=parse_method,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            st.success("RAG-Anything initialized!")

        st.divider()
        st.header("Processed Documents")
        if st.session_state.processed_docs:
            for doc in st.session_state.processed_docs:
                st.text(doc)
        else:
            st.text("No documents processed yet")

    # Main content area
    tab1, tab2 = st.tabs(["📄 Process Documents", "💬 Chat"])

    with tab1:
        st.header("Process Documents")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "docx", "pptx", "xlsx", "txt", "md", "jpg", "png"],
        )

        if uploaded_file is not None:
            if st.button("Process Document"):
                if st.session_state.rag is None:
                    initialize_rag()

                with st.spinner("Processing document..."):
                    # Save uploaded file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    try:
                        result = st.session_state.rag.process_document_complete(
                            file_path=tmp_path,
                            display_stats=True,
                        )

                        st.success(f"Document processed successfully!")
                        st.json({
                            "File": uploaded_file.name,
                            "Chunks": result.get("num_chunks", 0),
                            "Output": result.get("output_file", "N/A"),
                        })

                        if uploaded_file.name not in st.session_state.processed_docs:
                            st.session_state.processed_docs.append(uploaded_file.name)

                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                    finally:
                        # Clean up temp file
                        Path(tmp_path).unlink()

    with tab2:
        st.header("Chat with Documents")

        if st.session_state.rag is None:
            st.info("Please initialize RAG-Anything in the sidebar first, or process a document.")
            initialize_rag()

        # Display chat history
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
                            with st.expander("Sources"):
                                for source in sources:
                                    st.text(f"- {source}")

                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                        })

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()

