from flask import Flask, render_template, send_file
import assemblyai as aai
import requests
from io import BytesIO
import os

app = Flask(__name__)

# Replace with your AssemblyAI API key
aai.settings.api_key = "817cc900f5ec4b44be6559a62905a600"

# URL of the audio file to transcribe
AUDIO_FILE_URL = "https://drive.google.com/uc?export=download&id=13jYN1wN_MBUu3a7XKTCna1CpGjr4Nbbv"
# File path for the video file to play
VIDEO_FILE_PATH = "static/20230607_me_canadian_wildfires.mp3"

# Ensure the directory exists
os.makedirs(os.path.dirname(VIDEO_FILE_PATH), exist_ok=True)

# Download the file if it doesn't exist
if not os.path.exists(VIDEO_FILE_PATH):
    print("Downloading the video file...")
    response = requests.get(AUDIO_FILE_URL)
    with open(VIDEO_FILE_PATH, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

# Transcribe the audio file
config = aai.TranscriptionConfig(speaker_labels=True, sentiment_analysis=True, summarization=True,
                                 summary_model=aai.SummarizationModel.informative, summary_type=aai.SummarizationType.bullets)
transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(AUDIO_FILE_URL)

def format_timestamp(milliseconds):
    seconds = milliseconds / 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

@app.route('/')
def index():
    result_data = []

    if transcript and transcript.utterances:
        for utterance, sentiment_result in zip(transcript.utterances, transcript.sentiment_analysis):
            result_data.append({
                'speaker': utterance.speaker,
                'start': format_timestamp(utterance.start),
                'end': format_timestamp(utterance.end),
                'transcription': utterance.text,
                'sentiment': sentiment_result.sentiment,
                'confidence': sentiment_result.confidence
            })

    summary = str(transcript.summary) if transcript and transcript.summary else None

    return render_template('index3.html', result_data=result_data, summary=summary, video_file=VIDEO_FILE_PATH)

@app.route('/video')
def video():
    return send_file(VIDEO_FILE_PATH, mimetype='audio/mp3')

if __name__ == '__main__':
    app.run(debug=True)
