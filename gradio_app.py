from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

system_prompt_with_image = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
            What's in this image? Do you find anything wrong with it medically?
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

system_prompt_no_image = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
            Respond only based on what the patient says, no image is available.
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

def process_inputs(audio_filepath, image_filepath):
    if not audio_filepath:
        warning = "Please finish recording and click Stop before submitting."
        return "", warning, None

    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3",
    )

    has_image = bool(image_filepath)
    prompt = (
        system_prompt_with_image if has_image else system_prompt_no_image
    ) + speech_to_text_output
    encoded_image = encode_image(image_filepath) if has_image else None

    doctor_response = analyze_image_with_query(
        query=prompt,
        encoded_image=encoded_image,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    voice_of_doctor = text_to_speech_with_gtts(
        input_text=doctor_response, output_filepath="final.wav"
    )
    return speech_to_text_output, doctor_response, voice_of_doctor

iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath", label="Upload Image (optional)"),
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio("Temp.mp3"),
    ],
    title="AI Doctor with Vision and Voice",
)

iface.launch(
    debug=True,
    server_name=os.environ.get("HOST", "127.0.0.1"),
    server_port=int(os.environ.get("PORT", 7860)),
)