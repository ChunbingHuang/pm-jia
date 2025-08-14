"""
Prompt templates for the project.
"""

from typing import Dict

from src.pm_jia.config import GeneralConfig
from src.pm_jia.utils import load_yaml


class PromptTemplates:
    def __init__(self):
        self.prompt_template_path = GeneralConfig.prompt_template_path
        self.prompt_templates = self._get_prompt_templates()

    def _get_prompt_templates(self) -> Dict[str, str]:
        return load_yaml(self.prompt_template_path)

    def get_template(self, name: str) -> str:
        return self.prompt_templates[name]


if __name__ == "__main__":
    from src.pm_jia.logger import setup_logger

    logger = setup_logger(__name__)

    prompt_templates = PromptTemplates()
    safety_agent_prompt = prompt_templates.get_template("safety_agent_prompt")
    logger.info(safety_agent_prompt)
    logger.info("end")
