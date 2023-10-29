import gradio as gr

workspaces = ["Workspace 1", "Workspace 2", "Workspace 3"]

def add_workspace(workspace_name: str):
    global workspaces
    if workspace_name and workspace_name not in workspaces:
        workspaces.append(workspace_name)
        return f"{workspace_name} added! Current workspaces are: {workspaces}"
    else:
        return f"Current workspaces are: {workspaces}"
def select_workspace(workspace_name: str):
    return f"You selected: {workspace_name}"

iface = gr.Interface(
    add_workspace,
    ["text", gr.inputs.Dropdown(choices=workspaces, label="Select workspace")],
    ["text", "text"]
)


iface.launch()