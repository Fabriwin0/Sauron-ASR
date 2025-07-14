import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import json
import logging

class ConversationAgent:
    """
    A conversational agent using SmolLM-360M that processes transcripts
    and generates contextual responses.
    """
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM-360M-Instruct"):
        """
        Initialize the conversation agent with SmolLM model.
        
        Args:
            model_name: The model identifier for SmolLM
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.conversation_history = []
        self.max_history_length = 5  # Keep last 5 exchanges
        
        # Load model and tokenizer
        self._load_model()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self):
        """Load the SmolLM model and tokenizer."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_transcript(self, transcript: str) -> Dict:
        """
        Analyze the transcript to extract key information.
        
        Args:
            transcript: The transcribed text from Whisper
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "text": transcript.strip(),
            "word_count": len(transcript.split()),
            "contains_question": "?" in transcript,
            "sentiment": self._basic_sentiment_analysis(transcript),
            "key_topics": self._extract_key_topics(transcript)
        }
        
        return analysis
    
    def _basic_sentiment_analysis(self, text: str) -> str:
        """Basic sentiment analysis using keyword matching."""
        positive_words = ["good", "great", "excellent", "happy", "love", "like", "wonderful", "amazing"]
        negative_words = ["bad", "terrible", "hate", "awful", "horrible", "sad", "angry", "frustrated"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics using simple keyword extraction."""
        # This is a simplified approach - in production, you might use more sophisticated NLP
        common_topics = {
            "weather": ["weather", "rain", "sunny", "cold", "hot", "temperature"],
            "technology": ["computer", "phone", "software", "app", "internet", "AI"],
            "food": ["eat", "food", "restaurant", "cook", "meal", "hungry"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel", "visit"],
            "work": ["work", "job", "office", "meeting", "project", "business"],
            "health": ["health", "doctor", "medicine", "sick", "hospital", "exercise"]
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in common_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def generate_response(self, transcript: str, context: Optional[str] = None) -> str:
        """
        Generate a conversational response based on the transcript.
        
        Args:
            transcript: The transcribed speech
            context: Optional additional context
            
        Returns:
            Generated response text
        """
        # Analyze the transcript
        analysis = self.analyze_transcript(transcript)
        
        # Build conversation context
        conversation_context = self._build_conversation_context(analysis, context)
        
        # Generate response using SmolLM
        response = self._generate_with_model(conversation_context)
        
        # Update conversation history
        self._update_history(transcript, response)
        
        return response
    
    def _build_conversation_context(self, analysis: Dict, context: Optional[str] = None) -> str:
        """Build the conversation context for the model."""
        # Start with system prompt
        system_prompt = """You are a helpful and friendly conversational assistant. 
Respond naturally and appropriately to the user's input. Keep responses concise but engaging.
Be helpful and maintain a positive tone."""
        
        # Add conversation history
        history_context = ""
        if self.conversation_history:
            history_context = "\n\nPrevious conversation:\n"
            for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                history_context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
        
        # Add current input analysis
        current_input = f"\n\nCurrent user input: {analysis['text']}"
        
        # Add sentiment and topic context
        if analysis['sentiment'] != 'neutral':
            current_input += f"\n(User seems {analysis['sentiment']})"
        
        if analysis['key_topics']:
            current_input += f"\n(Topics: {', '.join(analysis['key_topics'])})"
        
        # Add additional context if provided
        if context:
            current_input += f"\nAdditional context: {context}"
        
        full_context = system_prompt + history_context + current_input + "\n\nAssistant:"
        
        return full_context
    
    def _generate_with_model(self, context: str) -> str:
        """Generate response using the SmolLM model."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(context, return_tensors="pt", truncate=True, max_length=512)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            response = full_response[len(context):].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing that right now. Could you please try again?"
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response."""
        # Remove any unwanted prefixes
        response = response.replace("Assistant:", "").strip()
        
        # Ensure response doesn't end mid-sentence
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Limit response length
        if len(response) > 200:
            sentences = response.split('.')
            truncated = []
            current_length = 0
            for sentence in sentences:
                if current_length + len(sentence) < 180:
                    truncated.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            response = '.'.join(truncated)
            if not response.endswith('.'):
                response += '.'
        
        return response
    
    def _update_history(self, user_input: str, assistant_response: str):
        """Update conversation history."""
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation."""
        if not self.conversation_history:
            return {"status": "No conversation history"}
        
        total_exchanges = len(self.conversation_history)
        recent_topics = []
        
        for exchange in self.conversation_history[-3:]:
            analysis = self.analyze_transcript(exchange["user"])
            recent_topics.extend(analysis["key_topics"])
        
        return {
            "total_exchanges": total_exchanges,
            "recent_topics": list(set(recent_topics)),
            "last_user_input": self.conversation_history[-1]["user"] if self.conversation_history else None,
            "last_response": self.conversation_history[-1]["assistant"] if self.conversation_history else None
        }