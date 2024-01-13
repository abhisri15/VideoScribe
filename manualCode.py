from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import tempfile
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Set your Hugging Face API token
auth_token = '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'

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def convert_audio_to_wav(input_file):
    try:
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("audio.wav"))
        file_extension = os.path.splitext(input_file)[1].lower()

        if file_extension == ".mp3":
            audio = AudioSegment.from_mp3(input_file)
        elif file_extension == ".m4a":
            audio = AudioSegment.from_file(input_file, "m4a")
        elif file_extension == ".mp4":
            video = VideoFileClip(input_file)
            video.audio.write_audiofile(output_file, codec="pcm_s16le", verbose=False)
            print(f"Conversion successful! WAV file saved as: {output_file}")
            return output_file
        else:
            raise ValueError("Unsupported audio format")

        audio.export(output_file, format="wav")
        print(f"Conversion successful! WAV file saved as: {output_file}")
        return output_file
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        return None


def transcribe_chunk(chunk_audio, language='hi-IN'):
    # Initialize a recognizer
    recognizer = sr.Recognizer()

    # Export the chunk audio as a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        chunk_audio.export(temp_wav.name, format="wav", codec="pcm_s16le")

        # Load the WAV file using the speech recognition library
        with sr.AudioFile(temp_wav.name) as source:
            chunk_data = recognizer.record(source)

    # Recognize the speech in the chunk with Hindi language and show all information
    try:
        recognition_result = recognizer.recognize_google(chunk_data, language=language, show_all=True)
        alternatives = recognition_result.get("alternative", [])
        transcriptions = [alt["transcript"] for alt in alternatives]
    except sr.UnknownValueError:
        transcriptions = ["Speech recognition could not understand the audio"]
    except sr.RequestError as e:
        transcriptions = ["Error with the speech recognition service; {0}".format(e)]
    finally:
        # Clean up the temporary WAV file
        os.remove(temp_wav.name)

    # Additional processing steps to enhance the transcriptions
    # ...

    return transcriptions


def process_chunk(args):
    audio, start_time, end_time, language = args
    chunk_audio = audio[start_time:end_time]
    return transcribe_chunk(chunk_audio, language)


def select_most_likely_transcription(transcription_results):
    # You can implement your logic here to select the most likely transcription
    # For example, you can choose the transcription with the highest confidence score
    return max(set(transcription_results), key=transcription_results.count)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)

            # Convert audio to WAV
            saved_wav_file_path = convert_audio_to_wav(file_path)
            if saved_wav_file_path:
                # Load diarization model
                diarization_model = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=auth_token)

                # Transcribe and diarize
                result = diarization_model({'uri': 'test', 'audio': saved_wav_file_path})

                # Extract diarization information
                diarization = result

                # Combine diarization information with transcription
                result_data = []
                audio = AudioSegment.from_wav(saved_wav_file_path)

                # Using ThreadPoolExecutor for multi-threading
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        start_time = int(turn.start * 1000)
                        end_time = int(turn.end * 1000)
                        future = executor.submit(process_chunk, (audio, start_time, end_time, 'hi-IN'))
                        futures.append((future, start_time, end_time, speaker))

                    # Wait for all threads to complete
                    for future, start_time, end_time, speaker in futures:
                        transcription_results = future.result()

                        if "Speech recognition could not understand the audio" in transcription_results:
                            continue

                        most_likely_transcription = select_most_likely_transcription(transcription_results)

                        print('Transcription for speaker {}: {}'.format(speaker, most_likely_transcription))

                        result_data.append({
                            "speaker": speaker,
                            "start": start_time / 1000,
                            "end": end_time / 1000,
                            "transcription": most_likely_transcription
                        })

                # Clean up temporary audio files
                for temp_file in os.listdir(app.config['UPLOAD_FOLDER']):
                    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_file)
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

                return render_template('result.html', result=result_data)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
