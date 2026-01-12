from gradio_client import Client
from smolagents import Tool

class AWEDFiNERTool(Tool):
    name = "awed_finer_tool"
    description = (
        "A specialized Fine-grained NER tool for 36 global languages (6.6B people). "
        "Supports high-resource and vulnerable languages (Bodo, Manipuri, Bishnupriya, Mizo). "
        "Use this for precise entity extraction (Person, Location, Org, etc.)."
    )
    inputs = {
        "text": {"type": "string", "description": "The text to analyze."},
        "language": {"type": "string", "description": "The language name (e.g., 'Bishnupriya', 'Bodo')."}
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = Client("prachuryyaIITG/AWED-FiNER")

    def forward(self, text: str, language: str) -> str:
        try:
            result = self.client.predict(
                text=text,
                language=language,
                api_name="/predict"
            )
            # Handling the Gradio HighlightedText list output
            if isinstance(result, list):
                return str(result)
            if isinstance(result, dict):
                return str(result.get('entities', result))
            return str(result)
        except Exception as e:
            return f"Error connecting to AWED-FiNER Service: {str(e)}"
