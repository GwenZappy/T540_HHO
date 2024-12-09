import os
import wave
from playsound import playsound
import sounddevice as sd
import streamlit as st
import openai

# API Key
from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Audio settings
SAMPLE_RATE = 16000  # 16 kHz
CHANNELS = 1         # Mono
CHUNK_DURATION = 2   # Process audio in 5-second chunks
OUTPUT_FILE = "recording.wav"

def play_audio(file_path):
    """Play a pre-recorded .wav audio file."""
    if os.path.exists(file_path):
        playsound(file_path)
    else:
        st.error(f"Audio file not found: {file_path}")

def transcribe_with_whisper(audio_file):
    """Transcribe audio using OpenAI Whisper."""
    try:
        with open(audio_file, "rb") as f:
            response = openai.Audio.transcribe("whisper-1", f)
        return response["text"]
    except Exception as e:
        st.error(f"Whisper Transcription Error: {e}")
        return ""

def analyze_sentiment_with_chatgpt(text):
    """Analyze sentiment and primary emotion."""
    prompt = f"""
    Analyze the sentiment of the following text based on the circumplex model of emotions. Identify the primary and secondary (if any) emotion (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and its intensity (Low, Medium, High):\n\n{text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def extract_entities_with_emotions(text):
    """Extract entities and emotions with associations."""
    prompt = f"""
    Extract the following categories of entities from the text below. For each category, list the relevant details:

    1. **People**: Names of people mentioned in the text.
    2. **Locations**: Specific locations mentioned (e.g., park, apartment, city, bookstore).
    3. **Events**: Key actions or activities described in the text (e.g., walking, kissing, watching a movie).
    4. **Environment Conditions**: The surroundings or environment (e.g., rainy, noisy, cold).
    5. **Emotions**: Identify emotions based on the circumplex model of emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and their intensity (Low, Medium, High).
    6. **Associations**: For each emotion, provide:
       - People
       - Locations
       - Events
       - Environment Conditions

    Text:
    {text}

    Please provide the output in a structured format.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def record_and_transcribe():
    """Record audio in chunks and stop when 'that's it' is detected."""
    st.info("Listening... Say 'that's it' to stop.")
    full_transcription = []
    stop_detected = False

    def save_chunk_to_wav(filename, chunk):
        """Save audio chunk to a .wav file."""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # Assuming 16-bit audio
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(chunk))

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        while not stop_detected:
            audio_chunk = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))[0]
            chunk_file = "chunk.wav"
            save_chunk_to_wav(chunk_file, audio_chunk)

            # Transcribe the chunk
            chunk_transcription = transcribe_with_whisper(chunk_file)
            st.write(f"Partial Transcription: {chunk_transcription}")
            full_transcription.append(chunk_transcription)

            if "that's it" in chunk_transcription.lower():
                stop_detected = True
                st.success("Stopping transcription...")
                play_audio("thanks for sharing.wav")  # Play the closing message

    # Combine all transcriptions
    return " ".join(full_transcription)


def process_transcription_with_chatgpt(transcription):
    """Process transcription text with sentiment analysis and entity extraction."""
    st.write("### Transcription")
    st.write(transcription)

    # Sentiment Analysis
    try:
        sentiment = analyze_sentiment_with_chatgpt(transcription)
        st.write("### Sentiment Analysis")
        st.write(sentiment)
    except Exception as e:
        st.error(f"Sentiment Analysis Error: {e}")

    # Entity Extraction with Emotions
    try:
        entities = extract_entities_with_emotions(transcription)
        st.write("### Extracted Entities and Emotions")
        st.write(entities)
    except Exception as e:
        st.error(f"Entity Extraction Error: {e}")

def main():
    """Main Streamlit App."""
    st.title("Neuropy HomeHub")
    st.write("Tell me about your day!")

    if st.button("Start Conversation"):
        try:
            play_audio("tell me about your day.wav")  # Play the opening message
            transcription = record_and_transcribe()
            process_transcription_with_chatgpt(transcription)
            #play_audio("thanks for sharing.wav")  # Play the closing message
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()