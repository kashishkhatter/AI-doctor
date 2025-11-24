from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

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

def process_inputs(audio_filepath, image_filepath, history):
    conversation_history = history or []
    if not audio_filepath:
        warning = "Please finish recording and click Stop before submitting."
        return "", warning, None, conversation_history, gr.update(visible=False)

    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3",
    )

    has_image = bool(image_filepath)
    prompt_text = (
        system_prompt_with_image if has_image else system_prompt_no_image
    ) + speech_to_text_output

    user_message_content = [{"type": "text", "text": prompt_text}]
    if has_image:
        user_message_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_filepath)}",
                },
            }
        )

    conversation_with_user = conversation_history + [
        {"role": "user", "content": user_message_content}
    ]

    doctor_response = analyze_image_with_query(
        messages=conversation_with_user,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    updated_conversation = conversation_with_user + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": doctor_response,
                }
            ],
        }
    ]

    voice_of_doctor = text_to_speech_with_gtts(
        input_text=doctor_response, output_filepath="final.wav"
    )

    return (
        speech_to_text_output,
        doctor_response,
        voice_of_doctor,
        updated_conversation,
        gr.update(visible=True),
    )


def reset_conversation():
    return "", "", None, [], gr.update(visible=False), None, None


def prepare_followup():
    """Clear audio and image inputs to prepare for follow-up question"""
    return None, None


with gr.Blocks(title="AI Doctor with Vision and Voice") as demo:
    gr.Markdown("# AI Doctor with Vision and Voice")

    convo_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record or Upload Audio",
            )
            image_input = gr.Image(
                type="filepath",
                label="Upload Image (optional)",
            )
            followup_btn = gr.Button(
                "Ask Follow-up Question",
                variant="secondary",
                visible=False,
            )
            submit_btn = gr.Button("Submit Question", variant="primary")
            clear_btn = gr.Button("Clear Conversation")
        with gr.Column(scale=1):
            speech_out = gr.Textbox(label="Speech to Text")
            doctor_out = gr.Textbox(label="Doctor's Response")
            doctor_audio = gr.Audio(label="Doctor's Voice", interactive=False)

    submit_btn.click(
        process_inputs,
        inputs=[audio_input, image_input, convo_state],
        outputs=[speech_out, doctor_out, doctor_audio, convo_state, followup_btn],
    )

    clear_btn.click(
        reset_conversation,
        inputs=None,
        outputs=[
            speech_out,
            doctor_out,
            doctor_audio,
            convo_state,
            followup_btn,
            audio_input,
            image_input,
        ],
    )

    followup_btn.click(
        prepare_followup,
        inputs=None,
        outputs=[audio_input, image_input],
    )

demo.launch(
    debug=True,
    server_name=os.environ.get("HOST", "127.0.0.1"),
    server_port=int(os.environ.get("PORT", 7860)),
)