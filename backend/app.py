import gradio as gr
from server import app as fastapi_app

# Gradio acts as a free host wrapper for our FastAPI Deep Learning Backend
# This allows deploying on Hugging Face Spaces without a credit card (Gradio tier)

def dummy_ui():
    return "WinGo Deep Learning Backend is running in the background!"

with gr.Blocks() as demo:
    gr.Markdown("# Deep Learning AI Server")
    gr.Markdown("The Python API is actively running at `/api/state`.")

# Mount the FastAPI app onto the Gradio server
gr.mount_gradio_app(fastapi_app, demo, path="/")
