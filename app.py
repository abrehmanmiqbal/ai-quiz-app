import streamlit as st
from quiz_generator import generate_mcqs
from ann_model import load_ann_model
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import base64
import io

# Load ANN model and vectorizer
model, vectorizer, label_encoder = load_ann_model()

# Session State
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "topic" not in st.session_state:
    st.session_state.topic = ""

# Classify difficulty of questions using ANN
def classify_difficulty(questions):
    if not questions:
        return []
    X = vectorizer.transform(questions).toarray()
    preds = model.predict(X)
    return label_encoder.inverse_transform(preds)

# Generate PDF
def create_score_pdf(score, total, topic):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Quiz Results", ln=1, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Topic: {topic}", ln=1)
    pdf.cell(200, 10, txt=f"Score: {score} / {total}", ln=2)
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return io.BytesIO(pdf_bytes)

def get_csv_download_link(score_data):
    csv = score_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="quiz_score.csv">Download CSV</a>'

# UI
st.set_page_config(page_title="AI Quiz App", layout="centered")
st.title("🤖 AI-Powered Quiz App")

with st.sidebar:
    st.markdown("### 🧠 Features")
    st.markdown("- Advanced MCQ generation via OpenRouter API")
    st.markdown("- Difficulty classification (ANN)")
    st.markdown("- YouTube & W3Schools help links")
    st.markdown("- Quiz retry + Export results")

page = st.sidebar.radio("Navigation", ["🏠 Home", "📝 Take Quiz", "📊 Results", "📤 Export"])

# ------------------ HOME ------------------
if page == "🏠 Home":
    st.header("Welcome!")
    topic = st.text_input("Enter a topic (e.g., Python functions, Neural Networks)", key="topic_input")

    if st.button("Generate Quiz"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            st.session_state.questions = []
            st.session_state.score = 0
            st.session_state.user_answers = []
            st.session_state.current_question = 0
            st.session_state.topic = topic

            # Generate MCQs from OpenRouter API
            raw_mcqs = generate_mcqs(topic, num_questions=10)
            if not raw_mcqs:
                st.error("Failed to generate questions. Try again.")
            else:
                difficulties = classify_difficulty([q["question"] for q in raw_mcqs])

                for mcq, diff in zip(raw_mcqs, difficulties):
                    st.session_state.questions.append({
                        "question": mcq["question"],
                        "options": mcq["options"],
                        "answer": mcq["answer"],
                        "difficulty": diff
                    })

                st.success("✅ Quiz generated! Go to '📝 Take Quiz' to begin.")

# ------------------ QUIZ ------------------
elif page == "📝 Take Quiz":
    if not st.session_state.questions:
        st.warning("Please generate a quiz first from the Home page.")
    else:
        q_idx = st.session_state.current_question
        if q_idx < len(st.session_state.questions):
            question = st.session_state.questions[q_idx]

            st.markdown(f"### Question {q_idx+1} of {len(st.session_state.questions)}")
            st.markdown(f"**Difficulty**: `{question['difficulty']}`")
            st.write(question["question"])

            selected = st.radio("Choose an answer:", question["options"], key=f"q{q_idx}")

            if st.button("Submit Answer"):
                st.session_state.user_answers.append(selected)
                if selected == question["answer"]:
                    st.session_state.score += 1

                st.session_state.current_question += 1

                if st.session_state.current_question >= len(st.session_state.questions):
                    st.success("🎉 Quiz completed! Check results in the 📊 Results tab.")
                else:
                    st.rerun()


# ------------------ RESULTS ------------------
elif page == "📊 Results":
    if not st.session_state.user_answers:
        st.info("Take the quiz to see results.")
    else:
        total = len(st.session_state.questions)
        score = st.session_state.score
        st.subheader("Your Score:")
        st.success(f"{score} / {total}")

        wrong_qs = []
        for i, (q, user_ans) in enumerate(zip(st.session_state.questions, st.session_state.user_answers)):
            if user_ans != q["answer"]:
                wrong_qs.append(q)

        if wrong_qs:
            st.markdown("### 🔍 Suggestions:")
            for q in wrong_qs:
                st.write(f"- **{q['question']}**")
                topic = st.session_state.topic or "the topic"
                st.markdown(f"[📺 YouTube: {topic}](https://www.youtube.com/results?search_query={topic})")
                st.markdown(f"[📘 W3Schools](https://www.w3schools.com/python/)")

        if st.button("🔁 Retry Quiz"):
            st.session_state.score = 0
            st.session_state.current_question = 0
            st.session_state.user_answers = []
            random.shuffle(st.session_state.questions)
            st.success("Shuffled questions! Go to 📝 Take Quiz.")

# ------------------ EXPORT ------------------
elif page == "📤 Export":
    if not st.session_state.user_answers:
        st.warning("Complete the quiz to export results.")
    else:
        topic = st.session_state.topic or "Quiz"
        pdf_buf = create_score_pdf(st.session_state.score, len(st.session_state.questions), topic)
        st.download_button("📄 Download PDF", data=pdf_buf, file_name="quiz_score.pdf", mime="application/pdf")

        df = pd.DataFrame({
            "Question": [q["question"] for q in st.session_state.questions],
            "Your Answer": st.session_state.user_answers,
            "Correct Answer": [q["answer"] for q in st.session_state.questions],
            "Difficulty": [q["difficulty"] for q in st.session_state.questions]
        })

        st.markdown(get_csv_download_link(df), unsafe_allow_html=True)
