from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from moviepy.editor import VideoFileClip
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
import assemblyai as aai

# Load environment variables
load_dotenv()

api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Set the AssemblyAI API key
aai.settings.api_key = api_key

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the Groq client
groq_client = Groq()

# Function to extract audio from video
def extract_audio_from_video(video_file, audio_file):
    try:
        video = VideoFileClip(video_file)
        video.audio.write_audiofile(audio_file)
        return audio_file
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

# Function to transcribe audio using AssemblyAI
def transcribe_audio(filename):
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(filename)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"Error during transcription: {transcript.error}")
            return None
        else:
            return transcript.text
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

# Function to setup FAISS vector store
def get_vectorstore(documents, embeddings):
    texts = [doc.page_content for doc in documents]
    return FAISS.from_texts(texts, embeddings)

# Initialize embedding model
model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# Setup text splitter
def text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
    )

# Setup language model
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)

# Function to answer question with FAISS vector store
def answer_question(question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    # Prepend system prompt to the user question
    system_prompt = (
        "You are a generative AI assistant specialized in summarizing video content, such as meeting discussions, "
        "presentations, or other relevant events. Your primary task is to extract the key points, topics discussed, "
        "decisions made, action items, and other important information from the video's content. Ensure the summary is "
        "concise yet comprehensive, accurately reflecting the main ideas and context of the video. Focus on presenting "
        "the information in a structured and easy-to-understand manner, highlighting the core themes and outcomes."
    )
    full_prompt = f"{system_prompt}\n\nQuestion: {question}"

    # Pass the full prompt as the query
    result = qa.invoke({"query": full_prompt})
    return result['result']

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "video" not in request.files:
        return redirect(url_for("index"))

    video_file = request.files["video"]
    if video_file.filename == "":
        return redirect(url_for("index"))

    # # Save video file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Extract audio from video
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_audio.mp3")
    extracted_audio_path = extract_audio_from_video(video_path, audio_path)

    if extracted_audio_path:
        transcription_text = transcribe_audio(extracted_audio_path)
        if transcription_text:
            documents = [Document(page_content=transcription_text)]
            global vectorstore
            vectorstore = get_vectorstore(documents, embeddings)
            return render_template("index.html", success=True, transcription=transcription_text)
    return render_template("index.html", error="Failed to process the video.")

@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.form.get("question")
    if not question or not vectorstore:
        return render_template("index.html", error="Please upload a video and enter a question.")

    response = answer_question(question, vectorstore)
    return render_template("index.html", response=response, question=question)

if __name__ == "__main__":
    app.run(debug=True)
