#!/usr/bin/env python3
"""
Main entry point for the integrated ASR + Conversational AI system.
This script provides both CLI and web interface options.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_web_interface():
    """Launch the Gradio web interface."""
    logger.info("Starting web interface...")
    try:
        from integrated_asr_chat import main
        main()
    except ImportError as e:
        logger.error(f"Failed to import web interface: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        sys.exit(1)

def run_cli_mode(audio_file: str, task: str = "transcribe", language: str = "english"):
    """Run in CLI mode for single audio file processing."""
    logger.info(f"Processing audio file: {audio_file}")
    
    try:
        from integrated_asr_chat import process_audio_and_respond
        
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            sys.exit(1)
        
        # Process the audio
        result = process_audio_and_respond(audio_file, task, language, include_analysis=True)
        
        # Display results
        print("\n" + "="*60)
        print("PROCESSING RESULTS")
        print("="*60)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)
        
        print(f"🎤 Transcript: {result['transcript']}")
        print(f"🤖 Response: {result['response']}")
        
        if "analysis" in result:
            analysis = result["analysis"]
            print(f"\n📊 Analysis:")
            print(f"   - Word count: {analysis['word_count']}")
            print(f"   - Sentiment: {analysis['sentiment']}")
            print(f"   - Contains question: {analysis['contains_question']}")
            print(f"   - Topics: {', '.join(analysis['key_topics']) if analysis['key_topics'] else 'None detected'}")
        
        summary = result.get('conversation_summary', {})
        print(f"\n💬 Conversation Summary:")
        print(f"   - Total exchanges: {summary.get('total_exchanges', 0)}")
        print(f"   - Recent topics: {', '.join(summary.get('recent_topics', [])) or 'None'}")
        
        print("\n" + "="*60)
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process audio: {e}")
        sys.exit(1)

def run_tests():
    """Run system tests."""
    logger.info("Running system tests...")
    try:
        from test_integration import main as test_main
        return test_main()
    except ImportError as e:
        logger.error(f"Failed to import test module: {e}")
        sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        "torch",
        "transformers",
        "torchaudio",
        "gradio",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are available!")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Integrated ASR + Conversational AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web interface (default)
  python run_integrated_system.py
  
  # Process single audio file
  python run_integrated_system.py --cli --audio examples/sample.wav
  
  # Process with translation
  python run_integrated_system.py --cli --audio examples/sample.wav --task translate --language spanish
  
  # Run tests
  python run_integrated_system.py --test
  
  # Check dependencies
  python run_integrated_system.py --check-deps
        """
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of web interface"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        help="Audio file to process (required for CLI mode)"
    )
    
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="ASR task type (default: transcribe)"
    )
    
    parser.add_argument(
        "--language",
        choices=["Russian", "French", "Italian", "Portuguese", "english", "spanish"],
        default="english",
        help="Language for ASR (default: english)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run system tests"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are installed"
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    elif args.test:
        sys.exit(run_tests())
    
    elif args.cli:
        if not args.audio:
            logger.error("--audio is required for CLI mode")
            parser.print_help()
            sys.exit(1)
        
        run_cli_mode(args.audio, args.task, args.language)
    
    else:
        # Default: web interface
        if not check_dependencies():
            sys.exit(1)
        run_web_interface()

if __name__ == "__main__":
    main()