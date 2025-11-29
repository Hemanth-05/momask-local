import os
import glob
import uuid
import subprocess

import gradio as gr

# Optional: set this if you want GPU on Spaces
USE_GPU = True  # Hugging Face: choose GPU hardware in Space settings


def generate_motion(prompt: str):
    """
    Runs MoMask's gen_t2m.py with a given text prompt and returns the path to the generated mp4.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a non-empty text prompt.")

    # Unique id for this run
    run_id = str(uuid.uuid4())[:8]

    # Build command
    cmd = [
        "python",
        "gen_t2m.py",
        "--gpu_id",
        "0" if USE_GPU else "-1",
        "--ext",
        run_id,
        "--text_prompt",
        prompt,
    ]

    # Run the generation
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Generation failed. See logs. (Error: {e})")

    # Find the generated mp4
    anim_dir = os.path.join("generation", run_id, "animation")
    mp4_files = glob.glob(os.path.join(anim_dir, "*.mp4"))

    if not mp4_files:
        raise gr.Error("No MP4 file was generated. Please check server logs.")

    # Return the first video file (Gradio will display it)
    return mp4_files[0]


# Build the Gradio interface
demo = gr.Interface(
    fn=generate_motion,
    inputs=gr.Textbox(lines=2, label="Text prompt", placeholder="A person is walking forward."),
    outputs=gr.Video(label="Generated motion"),
    title="MoMask: Text-to-Motion Generation",
    description=(
        "Enter a natural language description (e.g., 'A person is walking forward.') "
        "and generate a 3D motion animation using MoMask."
    ),
)


if __name__ == "__main__":
    # For local testing; on Hugging Face Spaces they call `python app.py`
    demo.launch()
