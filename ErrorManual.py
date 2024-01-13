from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip

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


# manualCode.py

# ... (previous code)

# manualCode.py

# ... (previous code)

# manualCode.py

# ... (previous code)

def process_chunk(chunk_path, chunk, result, r, start_time):
    result_data = []

    for window, _, speaker in result.itertracks(yield_label=True):
        adjusted_start = start_time + window.start
        adjusted_end = min(start_time + window.end, chunk.end)

        # Use 'chunk.audio.subclip' to extract the relevant portion of audio
        speaker_audio = chunk.audio.subclip(window.start, window.end)

        # Check if the extracted audio is non-empty before processing
        if speaker_audio.duration > 0:
            speaker_audio_file = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f"{speaker}_audio.wav"))
            speaker_audio.write_audiofile(speaker_audio_file, codec="pcm_s16le", verbose=False)

            try:
                with sr.AudioFile(speaker_audio_file) as source:
                    transcription = r.record(source)
                    transcription_result = r.recognize_google(transcription)
                    print('Transcription for speaker {}: {}'.format(speaker, transcription_result))
            except sr.UnknownValueError:
                transcription_result = 'Speech recognition could not understand the audio'
            except sr.RequestError as e:
                transcription_result = 'Error with the speech recognition service; {0}'.format(e)

            result_data.append({
                "speaker": speaker,
                "start": adjusted_start,
                "end": adjusted_end,
                "transcription": transcription_result
            })

            # Clean up temporary audio files
            if os.path.exists(speaker_audio_file):
                os.remove(speaker_audio_file)

    return result_data

# ... (remaining code)


# ... (remaining code)


# ... (remaining code)



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
                diarization_model = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                             use_auth_token=auth_token)

                # Transcribe and diarize
                result = diarization_model({'uri': 'test', 'audio': saved_wav_file_path})

                # Extract diarization information
                diarization = result

                # Initialize recognizer class (for recognizing the speech)
                r = sr.Recognizer()

                # Split the video into chunks
                video = VideoFileClip(file_path)
                chunk_size = 3  # seconds
                chunks = [video.subclip(i, i + chunk_size) for i in range(0, int(video.duration), chunk_size)]

                result_data = []

                with ThreadPoolExecutor() as executor:
                    futures = []

                    for i, chunk in enumerate(chunks):
                        chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"chunk_{i}.wav")
                        chunk.audio.write_audiofile(chunk_path, codec="pcm_s16le", verbose=False)

                        start_time = i * chunk_size
                        future = executor.submit(process_chunk, chunk_path, chunk, diarization, r, start_time)
                        futures.append(future)

                    for future in futures:
                        result_data.extend(future.result())

                return render_template('result.html', result=result_data)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
