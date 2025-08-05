"""
Prompt engineering and template management.
"""
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape


class PromptManager:
    """Manages loading and rendering of prompt templates."""

    def __init__(self, template_dir: str):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Renders a prompt template with the given context."""
        template = self.env.get_template(template_name)
        return template.render(context)
