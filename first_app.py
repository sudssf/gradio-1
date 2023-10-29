
# import gradio as gr

# def multiply(a, b):
#     return a * b
# iface = gr.Interface(fn=multiply, inputs=["number", "number"], outputs="number")


# iface.launch()
import gradio as gr

def process_files(workspace, list_of_files):
    # Handle the uploaded files and selected workspace here.
    # For demo, let's just return how many files received and what workspace was selected.
    return f"Received {len(list_of_files)} files for workspace {workspace}."

workspaces = ["Workspace 1", "Workspace 2", "Workspace 3"]

iface = gr.Interface(
    process_files,
    inputs=[
        gr.inputs.Dropdown(choices=workspaces, label="Select workspace"),
        gr.inputs.File(type="file", label="Upload Files")],
    outputs="text",
    layout="vertical")

iface.launch()