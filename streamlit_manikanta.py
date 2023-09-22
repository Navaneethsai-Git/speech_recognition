import streamlit as st
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

def main():
    st.title("Voice Sentiment Analysis")

    # Create a button to start recording
    if st.button("Record Voice"):
        record_and_analyze_audio()

def record_and_analyze_audio():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Define audio recording parameters
    sample_rate = 44100  # You can adjust this based on your requirements
    duration = 5  # Duration of audio recording in seconds

    st.write('Clearing background noise...')
    with sd.InputStream(callback=None, channels=1, samplerate=sample_rate):
        sd.sleep(int(duration * 1000))  # Sleep to allow audio input
        st.write('Recording...')
        recorded_audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
        sd.wait()

    st.write('Done recording..')

    # Convert the recorded audio to an AudioData object
    audio_data = sr.AudioData(recorded_audio.tobytes(), sample_rate=sample_rate, sample_width=2)

    try:
        st.write('Printing the message..')
        text = recognizer.recognize_google(audio_data, language='en-US')
        st.write('Your message: {}'.format(text))

        # Sentiment analysis
        Sentence = [str(text)]
        analyzer = SentimentIntensityAnalyzer()

        for sentence in Sentence:
            sentiment_scores = analyzer.polarity_scores(sentence)
            st.write('Sentiment scores for "{}": {}'.format(sentence, sentiment_scores))
    except sr.UnknownValueError:
        st.write("Speech recognition could not understand audio")
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == "__main__":
    main()