#!/usr/bin/env python3
"""
📧 Email Writing Assistant — Gradio Demo App
==============================================

A dual-mode email assistant fine-tuned on the Enron corporate email corpus:
  • Compose:  Generate a professional email from a subject/topic
  • Reply:    Generate a professional reply to a received email

Inference backends:
  1. llama-cpp-python with GGUF model (for free HF Spaces / CPU)
  2. HuggingFace Inference API (set HF_API_TOKEN env var)
  3. Demo/mock mode (no model loaded — returns placeholder)

Usage:
    python app/app.py
    MODEL_PATH=./model.gguf python app/app.py
"""

import os
import logging

import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")

SYSTEM_PROMPT_COMPOSE = (
    "You are a professional email writing assistant. "
    "Write clear, concise, and professional emails based on the given subject and context."
)

SYSTEM_PROMPT_REPLY = (
    "You are a professional email reply assistant. "
    "Write thoughtful and professional replies to the given email."
)


# ---------------------------------------------------------------------------
# Inference backend
# ---------------------------------------------------------------------------


class EmailGenerator:
    """Handles email generation with multiple backend support."""

    def __init__(self):
        self.backend = "demo"
        self.llm = None

        # Try llama-cpp-python first
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=2048,
                    n_threads=os.cpu_count() or 2,
                    verbose=False,
                )
                self.backend = "llama_cpp"
                logger.info("Loaded GGUF model: %s", MODEL_PATH)
            except Exception as e:
                logger.warning("Failed to load GGUF model: %s", e)

        # Try downloading from HF Hub
        elif HF_MODEL_REPO:
            try:
                from huggingface_hub import hf_hub_download
                from llama_cpp import Llama

                local_path = hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    filename="model-q4_k_m.gguf",
                    token=HF_API_TOKEN or None,
                )
                self.llm = Llama(
                    model_path=local_path,
                    n_ctx=2048,
                    n_threads=os.cpu_count() or 2,
                    verbose=False,
                )
                self.backend = "llama_cpp"
                logger.info("Downloaded & loaded GGUF model from: %s", HF_MODEL_REPO)
            except Exception as e:
                logger.warning("Failed to download/load GGUF model: %s", e)

        logger.info("Inference backend: %s", self.backend)

    def _build_prompt(self, system: str, user: str) -> str:
        """Build a Llama 3.1 chat prompt."""
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response using the available backend."""
        if self.backend == "llama_cpp" and self.llm is not None:
            prompt = self._build_prompt(system_prompt, user_prompt)
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()

        else:
            # Demo mode — return a polished placeholder
            return (
                "⚠️ No model loaded — running in demo mode.\n\n"
                "To enable inference, set one of these environment variables:\n"
                "  • MODEL_PATH=/path/to/model.gguf\n"
                "  • HF_MODEL_REPO=your-username/your-model-gguf\n\n"
                "This app supports:\n"
                "  1. Local GGUF inference via llama-cpp-python\n"
                "  2. HuggingFace Hub model download\n\n"
                f"Your prompt was: {user_prompt[:200]}"
            )


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

generator = EmailGenerator()


def compose_email(
    subject: str,
    tone: str,
    context: str,
    max_length: int,
    temperature: float,
) -> str:
    """Handle the compose tab: subject → email."""
    if not subject.strip():
        return "⚠️ Please enter a subject line."

    tone_instruction = f" Use a {tone.lower()} tone." if tone != "Professional" else ""
    context_part = f"\n\nAdditional context: {context}" if context.strip() else ""

    user_prompt = (
        f"Write a professional email with the subject: {subject}"
        f"{tone_instruction}{context_part}"
    )

    return generator.generate(
        system_prompt=SYSTEM_PROMPT_COMPOSE,
        user_prompt=user_prompt,
        max_tokens=max_length,
        temperature=temperature,
    )


def reply_to_email(
    original_email: str,
    tone: str,
    max_length: int,
    temperature: float,
) -> str:
    """Handle the reply tab: received email → reply."""
    if not original_email.strip():
        return "⚠️ Please paste the email you want to reply to."

    tone_instruction = f" Use a {tone.lower()} tone." if tone != "Professional" else ""

    user_prompt = (
        f"Reply to this email:{tone_instruction}\n\n{original_email}"
    )

    return generator.generate(
        system_prompt=SYSTEM_PROMPT_REPLY,
        user_prompt=user_prompt,
        max_tokens=max_length,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    background: #0a0a1a !important;
}

.main-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #1a1a3e 0%, #0f0f23 50%, #1a1a3e 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.1);
}

.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.main-header p {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.25rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    background: rgba(99, 102, 241, 0.1);
}

footer {
    text-align: center;
    padding: 1rem;
    color: #475569;
    font-size: 0.8rem;
}

.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

.generate-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
}

.generate-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}
"""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0a0a1a",
            body_text_color="#e2e8f0",
            block_background_fill="#111827",
            block_border_color="rgba(99, 102, 241, 0.15)",
            block_label_text_color="#94a3b8",
            input_background_fill="#1e293b",
            input_border_color="rgba(99, 102, 241, 0.2)",
            button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
            button_primary_text_color="#ffffff",
        ),
        title="📧 Email Writing Assistant",
    ) as demo:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>📧 Email Writing Assistant</h1>
            <p>AI-powered email generation fine-tuned on corporate email communications</p>
            <div style="margin-top: 0.75rem;">
                <span class="badge">Llama 3.1 8B</span>
                <span class="badge">QLoRA Fine-Tuned</span>
                <span class="badge">Enron Corpus</span>
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:
            # ── Tab 1: Compose ──────────────────────────────────────
            with gr.Tab("✍️ Compose Email", id="compose"):
                gr.Markdown("*Generate a professional email from a subject line and optional context.*")

                with gr.Row():
                    with gr.Column(scale=2):
                        compose_subject = gr.Textbox(
                            label="Subject / Topic",
                            placeholder="e.g. Quarterly budget review meeting request",
                            lines=1,
                            elem_id="compose-subject",
                        )
                        compose_context = gr.Textbox(
                            label="Additional Context (optional)",
                            placeholder="e.g. The meeting is for the finance team, scheduled next Tuesday at 2 PM...",
                            lines=3,
                            elem_id="compose-context",
                        )
                    with gr.Column(scale=1):
                        compose_tone = gr.Dropdown(
                            choices=["Professional", "Formal", "Casual", "Friendly"],
                            value="Professional",
                            label="Tone",
                            elem_id="compose-tone",
                        )
                        compose_length = gr.Slider(
                            minimum=64, maximum=512, value=256, step=32,
                            label="Max Length (tokens)",
                            elem_id="compose-length",
                        )
                        compose_temp = gr.Slider(
                            minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                            label="Temperature",
                            elem_id="compose-temp",
                        )

                compose_btn = gr.Button(
                    "✨ Generate Email",
                    variant="primary",
                    elem_classes=["generate-btn"],
                    elem_id="compose-btn",
                )
                compose_output = gr.Textbox(
                    label="Generated Email",
                    lines=12,
                    show_copy_button=True,
                    elem_id="compose-output",
                )

                compose_btn.click(
                    fn=compose_email,
                    inputs=[compose_subject, compose_tone, compose_context, compose_length, compose_temp],
                    outputs=compose_output,
                )

                gr.Examples(
                    examples=[
                        ["Quarterly budget review meeting — next Tuesday 2 PM", "Professional", "Finance team, conference room B", 256, 0.7],
                        ["Follow-up on contract negotiation with Acme Corp", "Formal", "", 256, 0.7],
                        ["Team lunch this Friday", "Casual", "New Italian place downtown", 192, 0.8],
                        ["Project milestone update — Phase 2 complete", "Professional", "Software development project, ahead of schedule", 256, 0.7],
                    ],
                    inputs=[compose_subject, compose_tone, compose_context, compose_length, compose_temp],
                    label="Example Prompts",
                )

            # ── Tab 2: Reply ────────────────────────────────────────
            with gr.Tab("💬 Reply to Email", id="reply"):
                gr.Markdown("*Generate a professional reply to a received email.*")

                with gr.Row():
                    with gr.Column(scale=2):
                        reply_original = gr.Textbox(
                            label="Original Email",
                            placeholder="Paste the email you want to reply to...",
                            lines=8,
                            elem_id="reply-original",
                        )
                    with gr.Column(scale=1):
                        reply_tone = gr.Dropdown(
                            choices=["Professional", "Formal", "Casual", "Friendly"],
                            value="Professional",
                            label="Tone",
                            elem_id="reply-tone",
                        )
                        reply_length = gr.Slider(
                            minimum=64, maximum=512, value=256, step=32,
                            label="Max Length (tokens)",
                            elem_id="reply-length",
                        )
                        reply_temp = gr.Slider(
                            minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                            label="Temperature",
                            elem_id="reply-temp",
                        )

                reply_btn = gr.Button(
                    "✨ Generate Reply",
                    variant="primary",
                    elem_classes=["generate-btn"],
                    elem_id="reply-btn",
                )
                reply_output = gr.Textbox(
                    label="Generated Reply",
                    lines=12,
                    show_copy_button=True,
                    elem_id="reply-output",
                )

                reply_btn.click(
                    fn=reply_to_email,
                    inputs=[reply_original, reply_tone, reply_length, reply_temp],
                    outputs=reply_output,
                )

                gr.Examples(
                    examples=[
                        [
                            "Hi Team,\n\nI wanted to follow up on the deliverables for the Q3 project. "
                            "Could you please share an update on the current status? We need to present "
                            "to the board next week.\n\nBest regards,\nSarah",
                            "Professional", 256, 0.7,
                        ],
                        [
                            "Dear Marketing Team,\n\nWe've decided to move the product launch from "
                            "October to November. Please adjust your campaigns accordingly and send "
                            "me the revised timeline by EOD Friday.\n\nThanks,\nMike",
                            "Formal", 256, 0.7,
                        ],
                    ],
                    inputs=[reply_original, reply_tone, reply_length, reply_temp],
                    label="Example Emails",
                )

        # Footer
        gr.HTML("""
        <footer>
            <p>Built with Llama 3.1 8B Instruct · QLoRA Fine-Tuned on Enron Email Corpus · Gradio</p>
        </footer>
        """)

    return demo


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
