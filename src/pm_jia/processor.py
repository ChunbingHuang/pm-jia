"""
Material processor for handling various input types. (under construction)
"""

import io
import json
import os
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import pytesseract
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pdf2image import convert_from_path
from PIL import Image

from src.pm_jia.config import ProcessorConfig
from src.pm_jia.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)


class MaterialProcessor:
    def __init__(self):
        self.config = ProcessorConfig()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", self.config.gemini_model)

        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            self.use_gemini = True
        else:
            logger.info("GEMINI_API_KEY is not set, using OCR fallback")
            self.use_gemini = False

    async def process_materials(self, materials: Dict) -> Dict:
        """
        Process various types of input materials.

        Args:
            materials: Dict containing file paths or binary data

        Returns:
            Dict containing processed materials in structured format
        """
        processed = {}

        for key, material in materials.items():
            if isinstance(material, (str, Path)):
                path = Path(material)
                if not path.exists():
                    logger.error(f"File does not exist: {path}")
                    continue

                if not self._validate_file_size(path):
                    logger.error(
                        f"File size exceeds {self.config.max_file_size_mb}MB limit: {path}"
                    )
                    processed[key] = {
                        "type": "error",
                        "message": f"File size exceeds {self.config.max_file_size_mb}MB limit",
                        "metadata": {"file_path": str(path)},
                    }
                    continue

                ext = path.suffix.lower()[1:]  # Remove the dot

                if ext in self.config.supported_image_types:
                    processed[key] = await self.process_image(path)
                elif ext in self.config.supported_table_types:
                    processed[key] = await self.process_table(path)
                elif ext in self.config.supported_text_types:
                    processed[key] = await self.process_text(path)
                elif ext == "pdf":
                    processed[key] = await self.process_pdf(path)
                else:
                    logger.error(f"Unsupported file type: {ext}")
                    processed[key] = {
                        "type": "unsupported",
                        "message": f"Unsupported file type: {ext}",
                        "metadata": {"file_path": str(path)},
                    }

        return processed

    async def process_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Process image files using Google Gemini or OCR fallback.

        Args:
            image_path: Path to image file

        Returns:
            Dict containing extracted text, content understanding, and metadata
        """
        image = Image.open(image_path)
        if self.use_gemini:
            try:
                image_file = self.gemini_client.files.upload(file=image_path)
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=[
                        image_file,
                        "Extract all text from this image and provide a detailed description of the content, including any charts, diagrams, or visual elements. Format the response as: TEXT: [extracted text] DESCRIPTION: [content description]",
                    ],
                )
                vision_output = response.text

                text_part = ""
                description_part = ""

                if "TEXT:" in vision_output and "DESCRIPTION:" in vision_output:
                    parts = vision_output.split("DESCRIPTION:")
                    text_part = parts[0].replace("TEXT:", "").strip()
                    description_part = parts[1].strip()
                else:
                    text_part = vision_output
                    description_part = "Content analysis not available"

                return {
                    "type": "image",
                    "text": text_part,
                    "content_description": description_part,
                    "processing_method": "gemini_vision",
                    "metadata": {
                        "size": image.size,
                        "format": image.format,
                        "mode": image.mode,
                        "file_path": str(image_path),
                    },
                }
            except Exception as e:
                logger.error(f"Failed to process image with Gemini: {e}, falling back to OCR")

        try:
            text = pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Failed to process image with OCR: {e}")
            text = "Failed to process image with OCR"

        return {
            "type": "image",
            "text": text,
            "content_description": "OCR text extraction only",
            "processing_method": "ocr",
            "metadata": {
                "size": image.size,
                "format": image.format,
                "mode": image.mode,
                "file_path": str(image_path),
            },
        }

    async def process_table(
        self, table_path: Union[str, Path], output_format: str = "json"
    ) -> Dict:
        """
        Process table files (CSV, Excel) with flexible output format.

        Args:
            table_path: Path to table file
            output_format: Either 'json' or 'text' for output format

        Returns:
            Dict containing structured table data in requested format
        """
        ext = Path(table_path).suffix.lower()[1:]

        try:
            if ext == "csv":
                df = pd.read_csv(table_path)
            else:
                df = pd.read_excel(table_path)
        except Exception as e:
            logger.error(f"Failed to read table file: {str(e)}")
            return {
                "type": "error",
                "message": f"Failed to read table file: {str(e)}",
                "metadata": {"file_path": str(table_path)},
            }

        base_result = {
            "type": "table",
            "metadata": {
                "columns": list(df.columns),
                "rows": len(df),
                "file_path": str(table_path),
                "format": ext,
            },
        }

        if output_format.lower() == "json":
            base_result["data"] = df.to_dict(orient="records")
            base_result["data_format"] = "json"
        else:
            base_result["data"] = df.to_string(index=False)
            base_result["data_format"] = "text"

        return base_result

    async def process_text(self, text_path: Union[str, Path]) -> Dict:
        """
        Process text files with enhanced JSON handling.

        Args:
            text_path: Path to text file

        Returns:
            Dict containing text content and metadata with proper JSON parsing
        """
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file: {str(e)}")
            return {
                "type": "error",
                "message": f"Failed to read text file: {str(e)}",
                "metadata": {"file_path": str(text_path)},
            }

        file_ext = Path(text_path).suffix.lower()[1:]

        result = {
            "type": "text",
            "content": content,
            "metadata": {"size": len(content), "format": file_ext, "file_path": str(text_path)},
        }

        if file_ext == "json":
            try:
                parsed_json = json.loads(content)
                result["parsed_data"] = parsed_json
                result["data_type"] = "structured_json"
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file: {str(e)}")
                result["data_type"] = "invalid_json"
                result["json_error"] = str(e)
        else:
            result["data_type"] = "plain_text"

        return result

    async def process_pdf(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Process PDF files with enhanced text extraction using Google Gemini vision.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict containing extracted text and metadata
        """
        try:
            images = convert_from_path(pdf_path)
        except Exception as e:
            logger.error(f"Failed to convert PDF: {str(e)}")
            return {
                "type": "error",
                "message": f"Failed to convert PDF: {str(e)}",
                "metadata": {"file_path": str(pdf_path)},
            }

        image_parts = [self._pil_to_part(p, "PNG") for p in images]

        pages = []
        page_details = []

        for i, img in enumerate(image_parts):
            if self.use_gemini:
                try:
                    image = types.Part.from_bytes(
                        mime_type=img["mime_type"],
                        data=img["data"],
                    )
                    response = self.gemini_client.models.generate_content(
                        model=self.gemini_model,
                        contents=[
                            image,
                            "Extract all text from this PDF page. Maintain structure and formatting as much as possible.",
                        ],
                    )
                    text = response.text
                    method = "gemini_vision"
                except Exception as e:
                    logger.error(f"Failed to process PDF with Gemini: {e}, falling back to OCR")
                    text = pytesseract.image_to_string(img)
                    method = "ocr"
            else:
                try:
                    text = pytesseract.image_to_string(img)
                    method = "ocr"
                except Exception as e:
                    logger.error(f"Failed to process PDF with OCR: {e}")
                    text = "Failed to process PDF with OCR"
                    method = "ocr"

            pages.append(text)
            page_details.append(
                {"page_number": i + 1, "text_length": len(text), "extraction_method": method}
            )

        combined_text = "\n\n--- Page Break ---\n\n".join(pages)

        return {
            "type": "pdf",
            "text": combined_text,
            "pages": pages,
            "metadata": {
                "num_pages": len(pages),
                "file_path": str(pdf_path),
                "page_details": page_details,
                "total_text_length": len(combined_text),
            },
        }

    def _validate_file_size(self, path: Union[str, Path]) -> bool:
        """Check if file size is within limits."""
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        return size_mb <= self.config.max_file_size_mb

    def _pil_to_part(self, image: Image.Image, fmt: str = "PNG") -> Dict:
        """Convert PIL image to bytes."""
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return {"mime_type": f"image/{fmt.lower()}", "data": buf.getvalue()}
