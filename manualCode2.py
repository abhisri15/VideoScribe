# app.py
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import tempfile

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


def transcribe_audio(audio_segment, language='hi-IN'):
    # Initialize a recognizer
    recognizer = sr.Recognizer()

    # Export the segment as a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        audio_segment.export(temp_wav.name, format="wav", codec="pcm_s16le")

        # Load the WAV file using the speech recognition library
        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)

    # Recognize the speech in the segment with Hindi language and show all information
    try:
        recognition_result = recognizer.recognize_google(audio_data, language=language, show_all=True)
        alternatives = recognition_result.get("alternative", [])
        transcriptions = [alt["transcript"] for alt in alternatives]
    except sr.UnknownValueError:
        transcriptions = ['Speech recognition could not understand the audio']
    except sr.RequestError as e:
        transcriptions = ['Error with the speech recognition service; {0}'.format(e)]
    finally:
        # Clean up the temporary WAV file
        os.remove(temp_wav.name)

    return transcriptions


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

                # Initialize recognizer class (for recognizing the speech)
                r = sr.Recognizer()

                # Combine diarization information with transcription
                result_data = []
                audio = AudioSegment.from_wav(saved_wav_file_path)

                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    # Adjust the start and end times to match the current speaker's turn
                    start_time = int(turn.start * 1000)  # Convert to milliseconds
                    end_time = int(turn.end * 1000)  # Convert to milliseconds

                    # Extract the portion of audio corresponding to the current speaker's turn
                    speaker_audio = audio[start_time:end_time]

                    # Transcribe the speaker audio using the updated function
                    transcription_result = transcribe_audio(speaker_audio, language='hi-IN')

                    # Skip entries where transcription is the error message
                    if "Speech recognition could not understand the audio" in transcription_result:
                        continue

                    print('Transcription for speaker {}: {}'.format(speaker, transcription_result))

                    result_data.append({
                        "speaker": speaker,
                        "start": turn.start,
                        "end": turn.end,
                        "transcription": transcription_result
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
