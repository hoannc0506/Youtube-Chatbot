import gradio as gr
import agent_utils
import app

agent_utils.set_logging_handlers()
chatbot = None

def init_chatbot(video_url):
    print(video_url)
    global chatbot
    chatbot = app.video_chatbot(video_url)
    return gr.update(visible=False), gr.update(visible=True)
    
def chat(message, chat_history):
    response = chatbot.chat(message)
    
    chat_history.append((message, response.response))
    
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("# Video Analyst Chatbot")

    with gr.Row(visible=True) as row1:
        video_url = gr.Textbox("Youtube URL", label="Video url", scale=3)
        # slider = gr.Slider(10, 100, render=False)
        analyze_button = gr.Button("Analyze", scale=1)

    # gr.ChatInterface(fn=chat, examples=sample_questions, title="Video chatbot")
    with gr.Row(visible=False) as row2:
        with gr.Column():
            chatbot = gr.Chatbot()
            
            msg = gr.Textbox(show_label=False, placeholder="Type a message...")
            msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
            
    # Button click will hide Row 1 and show Row 2 (chatbot)
    analyze_button.click(init_chatbot, inputs=[video_url], outputs=[row1, row2])

# demo.launch(inline=True)
demo.launch(share=True)