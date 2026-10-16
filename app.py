import uuid
from datetime import datetime

import streamlit as st

from config import DATA_DIR, get_config
from services.document_processor import DocumentProcessingError, process_document, validate_upload
from services.embedding_service import AIServiceError
from services.rag_service import (
    answer_question,
    generate_flashcards,
    generate_mcqs,
    generate_summary,
)
from services.vector_store import (
    add_document,
    delete_document,
    get_document_chunks,
    list_documents,
)
from utils import calculate_quiz_score, display_source


st.set_page_config(page_title="SmartStudy AI Pro", page_icon="📘", layout="wide")
DATA_DIR.mkdir(exist_ok=True)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f6f8fc; color: #17233d; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #162342; }
    .hero { padding: 2rem 2.2rem; border-radius: 22px; background: linear-gradient(125deg,#172c5b,#3947a8); color: white; margin-bottom: 1.5rem; }
    .hero h1 { color: white; font-size: clamp(2rem,4vw,3.2rem); margin: .4rem 0; }
    .hero p { color: #e4e9ff; font-size: 1.05rem; max-width: 720px; }
    .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .76rem; font-weight: 700; color: #b8c5ff; }
    .study-card { padding: 1rem 1.2rem; border: 1px solid #e6eaf2; background: white; border-radius: 16px; min-height: 118px; box-shadow: 0 5px 20px #23345b0a; }
    .study-card strong { font-size: 1.1rem; color: #1b2a4d; }
    .study-card p { color: #62708a; margin-bottom: 0; }
    .metric-card { padding: 1.1rem; border-radius: 15px; background: white; border: 1px solid #e6eaf2; }
    .metric-label { color: #68758d; font-size: .9rem; }
    .metric-value { color: #1d2d54; font-family: 'Manrope',sans-serif; font-weight: 800; font-size: 1.8rem; }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e8ebf2; }
    .stButton button { border-radius: 10px; font-weight: 600; }
    @media (max-width: 700px) { .hero { padding: 1.4rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    workspace = st.query_params.get("workspace", "")
    try:
        session_id = str(uuid.UUID(workspace))
    except (ValueError, TypeError, AttributeError):
        session_id = str(uuid.uuid4())
        st.query_params["workspace"] = session_id

    defaults = {
        "session_id": session_id,
        "documents": [],
        "chat_history": [],
        "question_count": 0,
        "quiz_count": 0,
        "quiz_history": [],
        "activities": [],
        "active_quiz": None,
        "quiz_answers": {},
        "quiz_index": 0,
        "quiz_submitted": False,
        "flashcards": [],
        "flashcard_index": 0,
        "flashcard_flipped": False,
        "flashcard_known": set(),
        "revision_cards": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "documents_restored" not in st.session_state:
        st.session_state.documents = list_documents(session_id)
        st.session_state.documents_restored = True


def log_activity(message: str):
    st.session_state.activities.insert(0, {
        "message": message,
        "time": datetime.now().strftime("%b %d, %H:%M"),
    })
    st.session_state.activities = st.session_state.activities[:8]


def document_options():
    return {doc["filename"]: doc for doc in st.session_state.documents}


def document_text(doc: dict) -> str:
    return "\n\n".join(get_document_chunks(st.session_state.session_id, doc["hash"]))


def warn_if_prompt_truncated(text: str):
    if len(text) > 24_000:
        st.warning(
            "This study tool sends the first 24,000 characters of the selected document to the AI provider. "
            "The full document remains available for retrieval in AI Study Chat."
        )


def missing_setup():
    config = get_config()
    required = {}
    if config.missing_embedding_key():
        required.setdefault(f"{config.embedding_provider.upper()}_API_KEY", []).append(
            "document indexing and search"
        )
    if config.missing_llm_key():
        required.setdefault(f"{config.llm_provider.upper()}_API_KEY", []).append(
            "AI responses and study tools"
        )
    missing = [f"{key} ({', '.join(purposes)})" for key, purposes in required.items()]
    return config, missing


def setup_notice(missing: list[str]):
    if missing:
        st.info(
            "Finish AI setup by adding **" + " and ".join(missing)
            + "** to your `.env` file. See the README for provider setup."
        )


def render_dashboard():
    docs = st.session_state.documents
    st.markdown(
        """<div class="hero"><span class="eyebrow">Your personal learning workspace</span>
        <h1>Upload Your Notes. Ask Anything. Learn Smarter.</h1>
        <p>Turn class notes and course materials into clear answers, revision summaries, quizzes, and flashcards.</p></div>""",
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    metrics = [
        ("Documents", len(docs)),
        ("Study chunks", sum(doc["chunks"] for doc in docs)),
        ("Questions asked", st.session_state.question_count),
        ("Quizzes completed", st.session_state.quiz_count),
    ]
    for column, (label, value) in zip(columns, metrics):
        column.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### A smarter way to study")
    feature_columns = st.columns(3)
    cards = [
        ("📄  Add your materials", "Upload PDFs, DOCX, TXT, or images. Text is extracted and organized into searchable chunks."),
        ("🔎  Ask with context", "RAG finds relevant passages first, then builds an answer grounded in your own notes."),
        ("🧠  Practice & remember", "Create focused summaries, quizzes, and flashcards from the documents you select."),
    ]
    for col, (title, detail) in zip(feature_columns, cards):
        col.markdown(f'<div class="study-card"><strong>{title}</strong><p>{detail}</p></div>', unsafe_allow_html=True)
    st.markdown("### Recent activity")
    if st.session_state.activities:
        for activity in st.session_state.activities[:5]:
            st.write(f"• {activity['message']}  ·  {activity['time']}")
    else:
        st.caption("Your recent study activity will appear here.")
    if st.session_state.quiz_history:
        st.markdown("### Recent quiz performance")
        for result in st.session_state.quiz_history[-5:][::-1]:
            st.write(f"• {result['score']}/{result['total']} correct · {result['percent']:.0f}%")
    st.caption("Bookmark this workspace URL to reopen its saved documents. Keep the URL private.")
    st.caption("RAG means retrieval-augmented generation: SmartStudy searches your notes for relevant passages before asking the AI to answer.")


def render_documents(config):
    st.title("My Documents")
    st.write("Upload study materials to extract text and add them to your private study search.")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="PDF, DOCX, TXT, JPG, JPEG, PNG · maximum 20 MB each",
    )
    if uploaded_files and st.button("Process documents", type="primary"):
        for uploaded in uploaded_files:
            try:
                content = uploaded.getvalue()
                validate_upload(uploaded.name, content)
                processed = process_document(uploaded.name, content)
                with st.spinner(f"Indexing {uploaded.name}..."):
                    digest, chunks, duplicate = add_document(
                        st.session_state.session_id,
                        uploaded.name,
                        content,
                        processed.text,
                        processed.pages,
                        config,
                        ocr_used=processed.ocr_used,
                    )
                if not any(doc["hash"] == digest for doc in st.session_state.documents):
                    st.session_state.documents.append({
                        "filename": uploaded.name,
                        "hash": digest,
                        "size": len(content),
                        "pages": processed.pages,
                        "chunks": chunks,
                        "ocr": processed.ocr_used,
                    })
                if duplicate:
                    st.info(f"{uploaded.name} is already in your study library.")
                else:
                    st.success(f"Added {uploaded.name}: {processed.pages} page(s), {chunks} searchable chunk(s).")
                    log_activity(f"Added {uploaded.name}")
                if processed.ocr_warning:
                    st.warning(f"{uploaded.name}: {processed.ocr_warning}")
                elif processed.ocr_used:
                    st.caption(f"OCR was used to read {uploaded.name}.")
            except (DocumentProcessingError, AIServiceError, ValueError) as exc:
                st.error(f"{uploaded.name}: {exc}")
    st.markdown("### Your library")
    if not st.session_state.documents:
        st.info("No documents yet. Upload your first set of notes above.")
    for doc in st.session_state.documents:
        with st.container(border=True):
            left, middle, right = st.columns([5, 3, 1])
            left.markdown(f"**{doc['filename']}**")
            left.caption(f"{doc['size'] / (1024 * 1024):.2f} MB · SHA-256 deduplicated")
            middle.caption(f"{doc['pages']} page(s) · {doc['chunks']} chunks" + (" · OCR" if doc["ocr"] else ""))
            if right.button("Remove", key=f"remove_{doc['hash']}"):
                delete_document(st.session_state.session_id, doc["hash"])
                st.session_state.documents = [item for item in st.session_state.documents if item["hash"] != doc["hash"]]
                st.rerun()


def render_chat(config, missing):
    st.title("AI Study Chat")
    setup_notice(missing)
    if not st.session_state.documents:
        st.info("Upload and process at least one document to start asking questions.")
        return
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            for source in item.get("sources", []):
                st.caption("Source: " + display_source(source))
    question = st.chat_input("Ask a question about your study materials...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        try:
            with st.spinner("Searching your notes and preparing an answer..."):
                answer, sources = answer_question(
                    st.session_state.session_id,
                    question,
                    st.session_state.chat_history[:-1],
                    config,
                )
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
            st.session_state.question_count += 1
            log_activity("Asked a question in Study Chat")
            st.rerun()
        except AIServiceError as exc:
            st.session_state.chat_history.pop()
            st.error(str(exc))


def render_summaries(config, missing):
    st.title("Smart Summaries")
    setup_notice(missing)
    docs = document_options()
    if not docs:
        st.info("Upload a document before creating a summary.")
        return
    selected = st.selectbox("Choose a document", list(docs), key="summary_doc")
    style = st.selectbox("Summary style", ["Short summary", "Detailed chapter summary", "Revision notes", "Exam-oriented key points"])
    text = document_text(docs[selected])
    warn_if_prompt_truncated(text)
    if st.button("Generate summary", type="primary"):
        try:
            with st.spinner("Reading your selected document..."):
                summary = generate_summary(text, config, style)
            st.session_state.summary_result = summary
            log_activity(f"Created {style.lower()} for {selected}")
        except (AIServiceError, ValueError) as exc:
            st.error(str(exc))
    if st.session_state.get("summary_result"):
        st.markdown(st.session_state.summary_result)


def render_quiz_generator(config, missing):
    st.title("MCQ Generator")
    setup_notice(missing)
    docs = document_options()
    if not docs:
        st.info("Upload a document before generating practice questions.")
        return
    selected = st.selectbox("Source document", list(docs), key="quiz_doc")
    col1, col2 = st.columns(2)
    count = col1.slider("Number of questions", min_value=3, max_value=10, value=5)
    difficulty = col2.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    topic = st.text_input("Focus topic (optional)", placeholder="e.g. cell structure")
    text = document_text(docs[selected])
    warn_if_prompt_truncated(text)
    if st.button("Generate quiz", type="primary"):
        try:
            with st.spinner("Creating questions from your notes..."):
                questions = generate_mcqs(text, config, count, difficulty, topic)
            st.session_state.active_quiz = questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_index = 0
            st.session_state.quiz_submitted = False
            st.session_state.quiz_submit_confirmed = False
            log_activity(f"Generated a {difficulty.lower()} quiz from {selected}")
            st.success("Quiz ready. Your answer key will stay hidden until you submit.")
        except (AIServiceError, ValueError) as exc:
            st.error(str(exc))
    if st.session_state.active_quiz:
        st.markdown(f"**{len(st.session_state.active_quiz)} questions ready** — open Practice Quiz to begin.")


def render_practice_quiz():
    st.title("Practice Quiz")
    questions = st.session_state.active_quiz
    if not questions:
        st.info("Generate a quiz first in MCQ Generator.")
        return
    if st.session_state.quiz_submitted:
        correct, percent = calculate_quiz_score(questions, st.session_state.quiz_answers)
        st.success(f"Score: {correct}/{len(questions)} · {percent:.0f}%")
        for index, question in enumerate(questions):
            selected = st.session_state.quiz_answers.get(index, "Not answered")
            is_correct = selected == question["answer"]
            st.markdown(f"**{index + 1}. {question['question']}**")
            st.write(f"Your answer: {selected}")
            st.write(f"{'✅ Correct' if is_correct else '❌ Correct answer: ' + question['answer']}")
            st.caption(question["explanation"])
        if st.button("Start a new quiz"):
            st.session_state.active_quiz = None
            st.session_state.quiz_submitted = False
            st.rerun()
        return

    index = st.session_state.quiz_index
    question = questions[index]
    st.progress((index + 1) / len(questions), text=f"Question {index + 1} of {len(questions)}")
    st.subheader(question["question"])
    prior = st.session_state.quiz_answers.get(index)
    selected_index = question["options"].index(prior) if prior in question["options"] else None
    choice = st.radio("Select one answer", question["options"], index=selected_index, key=f"quiz_choice_{index}")
    if choice:
        st.session_state.quiz_answers[index] = choice
    prev_col, next_col, submit_col = st.columns([1, 1, 2])
    if prev_col.button("← Previous", disabled=index == 0):
        st.session_state.quiz_index -= 1
        st.rerun()
    if next_col.button("Next →", disabled=index == len(questions) - 1):
        st.session_state.quiz_index += 1
        st.rerun()
    unanswered = len(questions) - len(st.session_state.quiz_answers)
    if submit_col.button("Submit quiz", type="primary"):
        if unanswered:
            st.warning(f"{unanswered} question(s) are unanswered. Submit again to finish anyway.")
            st.session_state.quiz_submit_confirmed = True
        else:
            st.session_state.quiz_submitted = True
            st.session_state.quiz_count += 1
            correct, percent = calculate_quiz_score(questions, st.session_state.quiz_answers)
            st.session_state.quiz_history.append({"score": correct, "total": len(questions), "percent": percent})
            st.session_state.quiz_submit_confirmed = False
            log_activity(f"Completed a quiz with {percent:.0f}%")
            st.rerun()
    if unanswered and st.session_state.get("quiz_submit_confirmed"):
        if st.button("Confirm submit with unanswered questions"):
            st.session_state.quiz_submitted = True
            st.session_state.quiz_count += 1
            correct, percent = calculate_quiz_score(questions, st.session_state.quiz_answers)
            st.session_state.quiz_history.append({"score": correct, "total": len(questions), "percent": percent})
            st.session_state.quiz_submit_confirmed = False
            log_activity(f"Completed a quiz with {percent:.0f}%")
            st.rerun()


def render_flashcards(config, missing):
    st.title("Flashcards")
    setup_notice(missing)
    docs = document_options()
    if not docs:
        st.info("Upload a document before creating flashcards.")
        return
    selected = st.selectbox("Source document", list(docs), key="flashcard_doc")
    count = st.slider("Number of cards", min_value=3, max_value=20, value=8)
    text = document_text(docs[selected])
    warn_if_prompt_truncated(text)
    if st.button("Create flashcards", type="primary"):
        try:
            with st.spinner("Creating your flashcards..."):
                cards = generate_flashcards(text, config, count)
            st.session_state.flashcards = cards
            st.session_state.flashcard_index = 0
            st.session_state.flashcard_flipped = False
            st.session_state.flashcard_known = set()
            st.session_state.revision_cards = []
            log_activity(f"Created flashcards from {selected}")
        except (AIServiceError, ValueError) as exc:
            st.error(str(exc))
    cards = st.session_state.flashcards
    if not cards:
        return
    index = st.session_state.flashcard_index
    card = cards[index]
    st.progress(len(st.session_state.flashcard_known) / len(cards), text=f"{len(st.session_state.flashcard_known)} of {len(cards)} marked known")
    st.markdown(f"### Card {index + 1} of {len(cards)}")
    with st.container(border=True):
        st.markdown("**Answer**" if st.session_state.flashcard_flipped else "**Question**")
        st.write(card["answer"] if st.session_state.flashcard_flipped else card["question"])
    if st.button("Flip card"):
        st.session_state.flashcard_flipped = not st.session_state.flashcard_flipped
        st.rerun()
    prev_col, next_col, known_col, revision_col = st.columns(4)
    if prev_col.button("← Previous", disabled=index == 0):
        st.session_state.flashcard_index -= 1
        st.session_state.flashcard_flipped = False
        st.rerun()
    if next_col.button("Next →", disabled=index == len(cards) - 1):
        st.session_state.flashcard_index += 1
        st.session_state.flashcard_flipped = False
        st.rerun()
    if known_col.button("✓ Known"):
        st.session_state.flashcard_known.add(index)
        st.session_state.revision_cards = [i for i in st.session_state.revision_cards if i != index]
        st.rerun()
    if revision_col.button("↻ Need revision"):
        st.session_state.revision_cards = sorted(set(st.session_state.revision_cards + [index]))
        st.session_state.flashcard_known.discard(index)
        st.rerun()
    if st.button("Shuffle cards"):
        import random
        random.shuffle(st.session_state.flashcards)
        st.session_state.flashcard_index = 0
        st.session_state.flashcard_flipped = False
        st.rerun()
    if st.session_state.revision_cards:
        st.caption(f"Need revision: {len(st.session_state.revision_cards)} card(s).")


def render_settings(config, missing):
    st.title("Settings & Privacy")
    setup_notice(missing)
    st.markdown(f"**Language model:** `{config.llm_provider}`  \n**Embedding provider:** `{config.embedding_provider}`")
    st.markdown(
        """
        - Add credentials in a local `.env` file; never paste API keys into chat or commit them.
        - Documents are extracted in this app and their embeddings are stored locally in `data/chroma_db/`.
        - When you use AI features, relevant document text or selected-document chunks are sent to the configured AI provider.
        - Study activity and chat history are held in the current Streamlit session; uploaded documents can be restored using the workspace ID in the page URL.
        - The workspace ID acts like a private access link. Do not share the URL; this demo does not provide user authentication.
        - Use **Remove** in My Documents to delete a document's stored vectors for this workspace.
        """
    )
    st.caption("Change `LLM_PROVIDER` or `EMBEDDING_PROVIDER` to `gemini` or `openai` in `.env`, then restart the app.")


def main():
    init_state()
    try:
        config, missing = missing_setup()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    with st.sidebar:
        st.markdown("## 📘 SmartStudy")
        st.caption("Your AI-powered study companion")
        page = st.radio(
            "Navigate",
            ["Dashboard", "My Documents", "AI Study Chat", "Smart Summaries", "MCQ Generator", "Practice Quiz", "Flashcards", "Settings"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"{len(st.session_state.documents)} document(s) in this session")

    if page == "Dashboard":
        render_dashboard()
    elif page == "My Documents":
        render_documents(config)
    elif page == "AI Study Chat":
        render_chat(config, missing)
    elif page == "Smart Summaries":
        render_summaries(config, missing)
    elif page == "MCQ Generator":
        render_quiz_generator(config, missing)
    elif page == "Practice Quiz":
        render_practice_quiz()
    elif page == "Flashcards":
        render_flashcards(config, missing)
    elif page == "Settings":
        render_settings(config, missing)


if __name__ == "__main__":
    main()
