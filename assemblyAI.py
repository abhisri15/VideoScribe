#https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3
#https://github.com/abhisri15/None/raw/main/vid.mp3
#https://github.com/abhisri15/None/raw/main/vid2.mp4
#https://github.com/abhisri15/None/raw/main/videoplayback.mp4

from flask import Flask, render_template
import assemblyai as aai

# Replace with your AssemblyAI API key
aai.settings.api_key = "817cc900f5ec4b44be6559a62905a600"

# URL of the file to transcribe
FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"
# Transcribe the audio file
config = aai.TranscriptionConfig(speaker_labels=True, sentiment_analysis=True, summarization=True, summary_model=aai.SummarizationModel.informative, summary_type=aai.SummarizationType.bullets)
transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(FILE_URL)

def format_timestamp(milliseconds):
    seconds = milliseconds / 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

result = []

if transcript and transcript.utterances:
    for utterance in transcript.utterances:
        result.append({
            'speaker': utterance.speaker,
            'start': format_timestamp(utterance.start),
            'end': format_timestamp(utterance.end),
            'transcription': utterance.text
        })

    # Writing 'results' to a text file
    with open('transcription_results.txt', 'w') as file:
        for utterance in transcript.utterances:
            file.write(f"Speaker {utterance.speaker} ({format_timestamp(utterance.start)} - {format_timestamp(utterance.end)}): {utterance.text}\n")

if transcript and transcript.summary:
    # Writing summary to 'transcription_results.txt'
    with open('transcription_results.txt', 'a') as file:
        file.write("\nSummary:\n")
        file.write(str(transcript.summary))

sentiment_results = []

if transcript and transcript.sentiment_analysis:
    for sentiment_result in transcript.sentiment_analysis:
        sentiment_results.append({
            'text': sentiment_result.text,
            'sentiment': sentiment_result.sentiment,
            'confidence': sentiment_result.confidence,
            'timestamp': f"{format_timestamp(sentiment_result.start)} - {format_timestamp(sentiment_result.end)}"
        })

    # Writing 'sentiment_results' to a text file
    with open('sentiment_results.txt', 'w') as file:
        for sentiment_result in transcript.sentiment_analysis:
            file.write(f"Text: {sentiment_result.text}\nSentiment: {sentiment_result.sentiment}\nConfidence: {sentiment_result.confidence}\nTimestamp: {format_timestamp(sentiment_result.start)} - {format_timestamp(sentiment_result.end)}\n\n")
