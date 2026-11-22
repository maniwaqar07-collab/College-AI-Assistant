
import streamlit as st
import rag

# Page Configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🌙",
    layout="wide"
)


import base64

with open("library.png", "rb") as image_file:
    bg_image = base64.b64encode(image_file.read()).decode()

# Custom Midnight Library Theme
st.markdown("""
<style>

.stApp {
   background-image:
    linear-gradient(rgba(23, 20, 18, 0.78), rgba(23, 20, 18, 0.78)),
    url("data:image/png;base64,IMAGE_PLACEHOLDER");
background-size: cover;
background-position: center;
background-attachment: fixed;
    color: #E8DDC8;
}

/* Main text */
.stApp p, .stApp li {
    color: #E8DDC8;
}

/* Main title */
.main-title {
    background: linear-gradient(135deg, #352A23, #211B19);
    padding: 28px 30px;
    border-radius: 20px;
    border: 1px solid #786451;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
    font-size: 36px;
    font-weight: 700;
    color: #C5A574;
}

/* Subtitle */
.subtitle {
    color: #B6A58D;
    font-size: 16px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #352A23, #211B19);
    border-right: 1px solid #786451;
}

[data-testid="stSidebar"] * {
    color: #E8DDC8;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: rgba(53, 42, 35, 0.85);
    border-radius: 16px;
    padding: 15px;
    border: 1px solid #786451;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}

/* Buttons */
.stButton button {
    background: #C5A574;
    color: #211B19;
    border: 1px solid #C5A574;
    border-radius: 12px;
    transition: 0.3s;
}

.stButton button:hover {
    background: #D6B98C;
    color: #171412;
    border: 1px solid #D6B98C;
}

/* Chat input */
[data-testid="stChatInput"] {
    border: 1px solid #786451;
    border-radius: 15px;
    background: #171412;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.2);
}

/* Chat input focus */
[data-testid="stChatInput"]:focus-within {
    border-color: #C5A574;
    box-shadow: 0 0 0 2px rgba(197, 165, 116, 0.15);
}

/* Expander */
[data-testid="stExpander"] {
    background: #352A23;
    border: 1px solid #786451;
    border-radius: 12px;
}

/* Sidebar spacing */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.5rem 1rem;
}

/* Knowledge base document cards */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(120, 100, 81, 0.15);
    border: 1px solid rgba(197, 165, 116, 0.25);
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #594638;
}

</style>
""".replace("IMAGE_PLACEHOLDER", bg_image), unsafe_allow_html=True)



# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.caption("Your College Knowledge Assistant")

    st.divider()

    st.subheader("📚 Knowledge Base")

    st.write(f"Total Documents: {len(rag.documents)}")

    for i, doc in enumerate(rag.documents, 1):
        with st.expander(f"Document {i}"):
            st.write(doc)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Header
st.markdown(
    '<div class="main-title">College AI Assistant 🌊</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions and get answers from your college knowledge base.</div>',
    unsafe_allow_html=True
)

st.divider()


# Welcome Message
if not st.session_state.messages:
    st.info("👋 Welcome! Ask me anything about your college.")

    st.markdown("### 💡 Try asking:")
    st.write("- What are the library timings?")
    st.write("- How many books can students borrow?")
    st.write("- When do mid-term exams start?")
    st.write("- What is the scholarship eligibility?")


# Display Chat History
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Retrieved Documents"):
                for doc, score in message["sources"]:
                    st.write(f"**Similarity Score:** {score}")
                    st.write(doc)
                    st.divider()


# Chat Input
question = st.chat_input("Ask your question here...")


if question:

    # Display User Message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve Documents
    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            try:
                retrieved = rag.retrieve(question)

                answer = rag.ask_llm(question, retrieved)

                st.markdown(answer)

                with st.expander("📄 Retrieved Documents"):
                    for doc, score in retrieved:
                        st.write(f"**Similarity Score:** {score}")
                        st.write(doc)
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
