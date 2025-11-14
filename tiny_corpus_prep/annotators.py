"""
Custom annotators for adding metadata columns to text data.
Provides base class and example implementations (e.g., Gemini API).
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Callable
from abc import ABC, abstractmethod
import polars as pl
from tqdm import tqdm


class BaseAnnotator(ABC):
    """
    Base class for custom annotators.
    
    Subclass this to create custom annotation logic.
    """
    
    @abstractmethod
    def annotate(self, text: str) -> Dict[str, Any]:
        """
        Annotate a single text string.
        
        Args:
            text: Input text to annotate
            
        Returns:
            Dictionary of column_name -> value pairs
        """
        pass
    
    def annotate_dataframe(
        self, 
        df: pl.DataFrame, 
        text_column: str = "text",
        show_progress: bool = True
    ) -> pl.DataFrame:
        """
        Annotate all rows in a DataFrame.
        
        Args:
            df: Input Polars DataFrame
            text_column: Name of the text column
            show_progress: Show progress bar
            
        Returns:
            DataFrame with additional annotation columns
        """
        results = []
        
        texts = df[text_column].to_list()
        iterator = tqdm(texts, desc="Annotating") if show_progress else texts
        
        for text in iterator:
            annotation = self.annotate(text)
            results.append(annotation)
        
        # Create DataFrame from results
        annotation_df = pl.DataFrame(results)
        
        # Concatenate with original DataFrame
        return pl.concat([df, annotation_df], how="horizontal")


class GeminiAnnotator(BaseAnnotator):
    """
    Annotator using Google Gemini API for text classification.
    
    Classifies text into topics and education levels.
    """
    
    ALLOWED_TOPICS = [
        "Arts & Humanities",
        "History & Archaeology",
        "Social Sciences",
        "Mathematics",
        "Physical Sciences",
        "Children entertainment",    
        "Computer Science",
        "Engineering & Technology",
        "Life Sciences",
        "Health & Medicine",
        "Education Studies",
        "Business & Finance",
        "Law & Legal Studies",
        "Environmental Science & Sustainability",
        "Languages & Linguistics",
        "Daily Routines & Home Management",
        "Family & Interpersonal Relationships",
        "Hobbies, Leisure & Entertainment",
        "Personal Health, Wellness & Lifestyle", 
        "Work Life & Career", 
        "Consumer Experiences & Personal Finance",
        "Personal Journeys & Life Events",
        "Food & Culinary"    
    ]
    
    EDUCATION_LEVELS = [
        "primary school", 
        "middle school", 
        "high school", 
        "university degree", 
        "PhD degree"
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        max_text_length: int = 15000,
        temperature: float = 0.1,
        max_output_tokens: int = 50
    ):
        """
        Initialize Gemini annotator.
        
        Args:
            api_key: Google API key (if None, loads from environment)
            model_name: Gemini model name
            max_text_length: Maximum text length to send to API
            temperature: Model temperature for generation
            max_output_tokens: Maximum tokens in response
        """
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
            import os
        except ImportError:
            raise ImportError(
                "Gemini annotator requires google-generativeai and python-dotenv. "
                "Install with: uv pip install -e '.[annotators]'"
            )
        
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("MY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API Key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        genai.configure(api_key=self.api_key)
        self.genai = genai
        self.model_name = model_name
        self.max_text_length = max_text_length
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
    
    def annotate(self, text: str) -> Dict[str, Any]:
        """
        Classify text using Gemini API.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with 'topic' and 'education' keys
        """
        import json
        import time
        
        if not text or text.isspace():
            return {"topic": None, "education": None}
        
        # Truncate if needed
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."
        
        prompt = f"""
Analyze the following text and determine its primary topic and the educational level typically required to understand it.

Text:
"{text}"

Instructions:
1. Choose the *single best* topic from this list: {self.ALLOWED_TOPICS}
2. Choose the *single most appropriate* education level one would need to properly understand the text: {self.EDUCATION_LEVELS}
3. Provide your answer ONLY in the following JSON format:
   {{"topic": "SELECTED_TOPIC", "education": "SELECTED_EDUCATION"}}

Example Response:
{{"topic": "Science", "education": "high school"}}

Output ONLY the JSON object.
"""
        
        try:
            model = self.genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature
                )
            )
            
            raw_response_text = response.text.strip()
            
            # Clean up response
            if raw_response_text.startswith("```json"):
                raw_response_text = raw_response_text[7:]
            if raw_response_text.endswith("```"):
                raw_response_text = raw_response_text[:-3]
            raw_response_text = raw_response_text.strip()
            
            # Parse JSON
            result = json.loads(raw_response_text)
            topic = result.get("topic")
            education = result.get("education")
            
            # Validate response
            if topic not in self.ALLOWED_TOPICS:
                print(f"Warning: Received invalid topic '{topic}'. Setting to None.")
                topic = None
            if education not in self.EDUCATION_LEVELS:
                print(f"Warning: Received invalid education '{education}'. Setting to None.")
                education = None
            
            return {"topic": topic, "education": education}
        
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON response: {raw_response_text}")
            return {"topic": "Error: JSON Decode", "education": "Error: JSON Decode"}
        except ValueError as ve:
            print(f"Error: Gemini API ValueError. Details: {ve}")
            return {"topic": "Error: API Value", "education": "Error: API Value"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"topic": "Error: API Call", "education": "Error: API Call"}


class CustomFunctionAnnotator(BaseAnnotator):
    """
    Annotator that wraps a custom function.
    
    Useful for simple annotation logic without subclassing.
    """
    
    def __init__(self, func: Callable[[str], Dict[str, Any]]):
        """
        Initialize with a custom function.
        
        Args:
            func: Function that takes text string and returns dict of annotations
        """
        self.func = func
    
    def annotate(self, text: str) -> Dict[str, Any]:
        """Apply the custom function."""
        return self.func(text)

