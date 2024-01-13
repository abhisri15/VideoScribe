import assemblyai as aai

aai.settings.api_key = "817cc900f5ec4b44be6559a62905a600"

audio_url = "https://github.com/abhisri15/None/raw/main/vid2.mp4"

# create a new TranscriptionConfig
config = aai.TranscriptionConfig(language_code="hi")

# set the configuration
transcriber = aai.Transcriber(config=config)

transcript = transcriber.transcribe(audio_url)

print(transcript.text)
