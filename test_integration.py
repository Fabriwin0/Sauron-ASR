#!/usr/bin/env python3
"""
Test script for the integrated ASR + Conversational AI system.
"""

import os
import sys
import torch
from conversation_agent import ConversationAgent
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_conversation_agent():
    """Test the conversation agent functionality."""
    logger.info("Testing ConversationAgent...")
    
    try:
        # Initialize agent
        agent = ConversationAgent()
        logger.info("✅ Agent initialized successfully")
        
        # Test basic response generation
        test_inputs = [
            "Hello, how are you today?",
            "What's the weather like?",
            "I'm feeling a bit sad today.",
            "Can you help me with my computer?",
            "Thank you for your help!"
        ]
        
        logger.info("Testing conversation flow...")
        for i, input_text in enumerate(test_inputs, 1):
            logger.info(f"\n--- Test {i} ---")
            logger.info(f"Input: {input_text}")
            
            # Analyze transcript
            analysis = agent.analyze_transcript(input_text)
            logger.info(f"Analysis: {analysis}")
            
            # Generate response
            response = agent.generate_response(input_text)
            logger.info(f"Response: {response}")
            
            # Get conversation summary
            summary = agent.get_conversation_summary()
            logger.info(f"Summary: {summary}")
        
        logger.info("✅ All conversation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversation agent test failed: {e}")
        return False

def test_model_loading():
    """Test if the SmolLM model can be loaded."""
    logger.info("Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
        logger.info(f"Loading model: {model_name}")
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✅ Tokenizer loaded successfully")
        
        # Test model loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
        )
        logger.info("✅ Model loaded successfully")
        
        # Test basic generation
        test_text = "Hello, how are you?"
        inputs = tokenizer.encode(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test generation: {response}")
        logger.info("✅ Model generation test passed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False

def test_audio_processing():
    """Test if audio processing dependencies work."""
    logger.info("Testing audio processing...")
    
    try:
        import torchaudio
        import numpy as np
        
        # Create a dummy audio signal
        sample_rate = 16000
        duration = 1  # 1 second
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_signal = np.sin(2 * np.pi * frequency * t)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio_signal, dtype=torch.float32).unsqueeze(0)
        
        logger.info(f"Created test audio: shape={audio_tensor.shape}, sample_rate={sample_rate}")
        logger.info("✅ Audio processing test passed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Audio processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting integration tests...")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Audio Processing", test_audio_processing),
        ("Conversation Agent", test_conversation_agent),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        return 0
    else:
        logger.error("⚠️ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())