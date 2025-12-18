import pyttsx3

engine = pyttsx3.init()

def text_to_speech(text, output_file="output.wav"):
    engine.save_to_file(text, output_file)
    engine.runAndWait()
    return output_file
