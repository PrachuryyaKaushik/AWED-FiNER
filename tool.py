from gradio_client import Client
from smolagents import Tool

class AWEDFiNERTool(Tool):
    name = "awed_finer_tool"
    description = (
        "High-precision Fine-grained NER for 36 global languages (6.6B people). "
        "Includes high-resource languages and vulnerable/low-resource languages "
        "(Bodo, Manipuri, Bishnupriya, Mizo). Use this tool to extract entities "
        "like Person, Org, or Location when standard LLMs lack regional accuracy."
    )
    inputs = {
        "text": {"type": "string", "description": "The sentence to analyze."},
        "language": {"type": "string", "description": "The language name (e.g., 'Assamese', 'Bodo', 'Ukrainian', 'French')."}
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Directly connecting to your specific HF Space
        self.client = Client("prachuryyaIITG/AWED-FiNER")

    def forward(self, text: str, language: str) -> str:
        try:
            # Calls your AWED-FiNER Space via its API
            result = self.client.predict(
                text=text,
                language=language,
                api_name="/predict"
            )
            return str(result)
        except Exception as e:
            return f"Error connecting to AWED-FiNER Service: {str(e)}"
