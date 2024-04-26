import gradio as gr
from pydub.playback import play
from mido import MidiFile, MidiTrack, Message
import openai
import subprocess
from pydub import AudioSegment
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

openai.api_key = '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'

rock_music_prompt = """
Generate a piece of rock music that embodies the characteristics of rock music. 
The music should have energetic electric guitars, a driving rhythm section, and powerful vocals. 
It should feature a moderate to fast tempo, guitar distortion, and dynamic shifts between quiet and loud sections. 
The lyrics should address themes commonly found in rock music, such as rebellion, love, and personal experiences.
"""

soundfont_file = "sountrack.sf2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.to(device)

def convert_music_and_text(details, input_type, audio=None):
    output_text = ""
    if input_type == "Text":
        prompt = rock_music_prompt + f"\nUser input: {details}"
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.2,
            max_tokens=500 
        )
        music_text = response.choices[0].text.strip()

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        for note_char in music_text:
            if note_char.isdigit():
                note = int(note_char)
                track.append(Message("note_on", note=note, velocity=64, time=0))
                track.append(Message("note_off", note=note, velocity=64, time=480))

        generated_music_file = "generated_music.mid"
        mid.save(generated_music_file)

        # wav_output_file = "generated_music.wav"
        # subprocess.run(["fluidsynth", "-F", wav_output_file, soundfont_file, generated_music_file])

        # # Load the generated WAV file
        # generated_wav = AudioSegment.from_wav(wav_output_file)

        # # Save the generated WAV file as MP3
        # generated_mp3_file = "generated_music.mp3"
        # generated_wav.export(generated_mp3_file, format="mp3")
        
        output_text = f"Generated music saved as: {generated_music_file}"
    else:
       
        audio = AudioSegment.from_mp3(audio)
        wav_audio_file = 'user_audio.wav'
        audio.export(wav_audio_file, format="wav")

        input_audio, sr = librosa.load(wav_audio_file, sr=16000)

        input_values = tokenizer(input_audio, return_tensors="pt", padding="longest").input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)

        output_text = f"Transcribed text: {transcription[0]}"

    return output_text

input_text = gr.Textbox(label="Enter Written Details", type="text")
input_audio = gr.Audio(label="Upload an Audio", type="filepath")
input_type = gr.Radio(["Text", "Audio"], label="Input Type")
output =gr.File(label="Generated Music", type="file")

gr.Interface(
    fn=convert_music_and_text,
    inputs=[input_text, input_type, input_audio],
    outputs=output
).launch(share=True)