from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
import speech_recognition as sr
from google.cloud import storage
from moviepy.editor import VideoFileClip

app = Flask(__name__)

# Initialize a client
storage_client = storage.Client()
bucket_name = 'video_scribe'
# Set your Hugging Face API token
auth_token = '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'
# Define the upload folder
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def convert_audio_to_wav(input_file):
    try:
        file_extension = os.path.splitext(input_file)[1].lower()

        if file_extension == ".mp3":
            audio = AudioSegment.from_mp3(input_file)
        elif file_extension == ".m4a":
            audio = AudioSegment.from_file(input_file, "m4a")
        elif file_extension == ".mp4":
            video = VideoFileClip(input_file)
            audio = video.audio
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("audio.wav"))
            audio.write_audiofile(output_file, codec="pcm_s16le", verbose=False)
            print(f"Conversion successful! WAV file saved as: {output_file}")
            return output_file
        else:
            raise ValueError("Unsupported audio format")

        output_file = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("audio.wav"))
        audio.export(output_file, format="wav")

        print(f"Conversion successful! WAV file saved as: {output_file}")
        return output_file

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Use Flask's temporary directory
            file_path = os.path.join('/tmp', secure_filename(file.filename))
            file.save(file_path)

            # Convert audio to WAV
            saved_wav_file_path = convert_audio_to_wav(file_path)
            if saved_wav_file_path:
                # Load diarization model
                diarization_model = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                             use_auth_token=auth_token)

                # Transcribe and diarize
                result = diarization_model({'audio': saved_wav_file_path})

                # Extract diarization information
                diarization = result

                # Initialize recognizer class (for recognizing the speech)
                r = sr.Recognizer()

                # Reading Audio file as source
                # listening to the audio file and store in audio_text variable
                with sr.AudioFile(saved_wav_file_path) as source:
                    audio_text = r.listen(source)

                try:
                    # using Google Speech Recognition
                    transcription = r.recognize_google(audio_text)
                    print('Transcription: ', transcription)

                except:
                    transcription = 'Sorry.. run again...'

                # Combine diarization information with transcription
                result_data = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    result_data.append({
                        "speaker": speaker,
                        "start": turn.start,
                        "end": turn.end,
                        "transcription": transcription
                    })

                # No need to upload to Google Cloud Storage in this example

                return render_template('result.html', result=result_data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
