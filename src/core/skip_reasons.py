"""
Skip reason enumeration for text classification.

Provides detailed categorization of why texts are skipped during classification,
enabling better operator triage without opening logs.
"""

from enum import Enum
from typing import Optional


class SkipReason(str, Enum):
    """Enumeration of reasons why a text might be skipped during classification."""
    
    NONE = "none"  # Text was processed successfully
    EMPTY = "empty"  # Text is empty string
    WHITESPACE = "whitespace"  # Text contains only whitespace
    NAN = "nan"  # Text is pandas NaN or string "nan"
    NON_TEXT = "non_text"  # Text is not a string type (number, bool, etc.)
    
    def __str__(self) -> str:
        """Return string value for display."""
        return self.value
    
    @classmethod
    def from_text(cls, text: any) -> 'SkipReason':
        """Determine skip reason for a given text.
        
        Args:
            text: The input text to check
            
        Returns:
            SkipReason enum value indicating why text should be skipped or NONE
        """
        # Check if text is None
        if text is None:
            return cls.EMPTY
        
        # Check if text is not a string (number, bool, etc.)
        if not isinstance(text, str):
            # Special case for pandas NaN (float)
            text_str = str(text)
            if text_str.lower() == 'nan':
                return cls.NAN
            return cls.NON_TEXT
        
        # Check if empty
        if not text:
            return cls.EMPTY
        
        # Check if only whitespace
        if not text.strip():
            return cls.WHITESPACE
        
        # Check for string "nan"
        if text.lower() == 'nan':
            return cls.NAN
        
        # Text is valid (no length restriction)
        return cls.NONE
    
    def should_skip(self) -> bool:
        """Return True if this reason indicates the text should be skipped."""
        return self != SkipReason.NONE
