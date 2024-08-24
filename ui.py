import gradio as gr
import agent_utils

agent_utils.set_logging_handlers()
agent = agent_utils.get_agent()

def chat(message, history):
    response = agent.chat(message)
    return response.response

sample_questions = [
    "what is the prize",
    "Describe the winning scene",
    "What happened in the first 3 minutes of the video?"
]

demo = gr.ChatInterface(fn=chat, examples=sample_questions, title="Video chatbot")
demo.launch(share=True)