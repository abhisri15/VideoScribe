import subprocess
import wave
import contextlib
import datetime
import numpy as np
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import torch
import whisper
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Load the pretrained speaker embedding model
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cpu"))

# This is just to convert to encode to UTF 8
import locale
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# Replace this with the path to your audio file
path = "Vid2.mp4"

# Now select the speaker language and model
num_speakers = 2
language = 'English'
model_size = 'large'

model_name = model_size
if language == 'English' and model_size != 'large':
    model_name += '.en'

# This is a check whether an audio is wav or not
if path[-3:] != 'wav':
    subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
    path = 'audio.wav'

# Loading the model
model = whisper.load_model(model_size)

# Generate the segment of it
result = model.transcribe(path)
segments = result["segments"]

# Skip language detection

# This helps to get the frames rate and duration
with contextlib.closing(wave.open(path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

# This is pyannote built-in function
audio = Audio()

# This helps to split the audios into waveform
def segment_embedding(segment):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)

    # Convert waveform to single channel
    waveform = waveform.mean(dim=0, keepdim=True)

    return embedding_model(waveform.unsqueeze(0))

# Here we create the embeddings
embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

# Here we are using agglomerative clustering to cluster same channels as 1 speaker
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

def time(secs):
    return datetime.timedelta(seconds=round(secs))

# Replace this with the desired path for the transcript file
transcript_path = "trans.txt"
with open(transcript_path, "w", encoding="UTF-8") as f:
    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')

print(open(transcript_path, 'r', encoding="UTF-8").read())
