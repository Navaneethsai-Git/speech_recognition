from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Provide the path to your audio file
audio_file_path = 'harvard.wav'  
# Replace with the path to your audio file

# Load the audio file
with sr.AudioFile(audio_file_path) as source:
    print('Processing audio file...')
    audio = recognizer.record(source)

try:
    # Transcribe the audio
    text = recognizer.recognize_google(audio, language='en-US')
    print('Transcribed text: {}'.format(text))
except Exception as ex:
    print('Error transcribing audio:', ex)

# Sentiment analysis
sentences = []
analyzer = SentimentIntensityAnalyzer()

for sentence in sentences:
    sentiment_scores = analyzer.polarity_scores(sentence)
    print(sentiment_scores)
