import logging
from typing import Any, Dict, List, Optional

from .audio_processor import AudioTranscript
from .clients.llm_client import LLMClient
from .frame import Frame
from .frame_selection import FrameContextItem
from .prompt import PromptLoader

logger = logging.getLogger(__name__)

MAX_PREVIOUS_FRAME_CONTEXT = 3
MAX_PREVIOUS_ANALYSIS_CHARS = 700
MAX_OCR_EVIDENCE_CHARS = 2000
MAX_CONTEXT_OCR_CHARS = 500

class VideoAnalyzer:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        prompt_loader: PromptLoader,
        temperature: float,
        user_prompt: str = "",
        frame_num_predict: int = 300,
        frame_no_think: bool = False,
    ):
        """Initialize the VideoAnalyzer.
        
        Args:
            client: LLM client for making API calls
            model: Name of the model to use
            prompt_loader: Loader for prompt templates
            user_prompt: Optional user question about the video that will be injected into frame analysis
                        and video description prompts using the {prompt} token
        """
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.user_prompt = user_prompt  # Store user's question about the video
        self.frame_num_predict = frame_num_predict
        self.frame_no_think = frame_no_think
        self._load_prompts()
        self.previous_analyses = []
        
    def _format_user_prompt(self) -> str:
        """Format the user's prompt by adding prefix if not empty."""
        if self.user_prompt:
            return f"I want to know {self.user_prompt}"
        return ""
        
    def _load_prompts(self):
        """Load prompts from files."""
        self.frame_prompt = self.prompt_loader.get_by_index(0)  # Frame Analysis prompt
        self.video_prompt = self.prompt_loader.get_by_index(1)  # Video Reconstruction prompt

    def _format_previous_analyses(self) -> str:
        """Format previous frame analyses for inclusion in prompt."""
        if not self.previous_analyses:
            return ""
            
        formatted_analyses = []
        start_index = max(0, len(self.previous_analyses) - MAX_PREVIOUS_FRAME_CONTEXT)
        for i, analysis in enumerate(self.previous_analyses[start_index:], start=start_index):
            text = analysis.get('response', 'No analysis available')
            if len(text) > MAX_PREVIOUS_ANALYSIS_CHARS:
                text = text[:MAX_PREVIOUS_ANALYSIS_CHARS] + "\n[truncated]"
            formatted_analysis = (
                f"Frame {i}\n"
                f"{text}\n"
            )
            formatted_analyses.append(formatted_analysis)
            
        return "\n".join(formatted_analyses)

    def analyze_frame(
        self,
        frame: Frame,
        ocr_text: str = "",
        context_window: Optional[List[FrameContextItem]] = None,
        context_ocr_texts: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """Analyze a single frame using the LLM."""
        # Replace {PREVIOUS_FRAMES} token with formatted previous analyses
        # Replace tokens in the prompt template
        previous_context = "" if context_window else self._format_previous_analyses()
        prompt = self.frame_prompt.replace("{PREVIOUS_FRAMES}", previous_context)
        prompt = prompt.replace("{prompt}", self._format_user_prompt())
        prompt = f"{prompt}\nThis is frame {frame.number} captured at {frame.timestamp:.2f} seconds."
        image_paths = None
        if context_window:
            prompt = f"{prompt}\n\n{self._format_frame_context_window(frame, context_window, context_ocr_texts or {})}"
            image_paths = [str(item.frame.path) for item in context_window]
        if self.frame_no_think:
            prompt = f"/no_think\n{prompt}"
        if ocr_text:
            if len(ocr_text) > MAX_OCR_EVIDENCE_CHARS:
                ocr_text = ocr_text[:MAX_OCR_EVIDENCE_CHARS] + "\n[truncated]"
            prompt = (
                f"{prompt}\n\nOCR evidence from this frame follows. Treat it as hard evidence "
                f"for visible UI text, commands, labels, and filenames:\n{ocr_text}"
            )
        
        try:
            response = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                image_paths=image_paths,
                model=self.model,
                temperature=self.temperature,
                num_predict=self.frame_num_predict
            )
            logger.debug(f"Successfully analyzed frame {frame.number}")
            
            # Store the analysis for future frames
            analysis_result = {k: v for k, v in response.items() if k != "context"}
            if not context_window:
                self.previous_analyses.append(analysis_result)
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error analyzing frame {frame.number}: {e}")
            error_result = {"response": f"Error analyzing frame {frame.number}: {str(e)}"}
            if not context_window:
                self.previous_analyses.append(error_result)
            return error_result

    def _format_frame_context_window(
        self,
        current_frame: Frame,
        context_window: List[FrameContextItem],
        context_ocr_texts: Dict[int, str],
    ) -> str:
        lines = [
            "Multi-image temporal context:",
            "The attached images are ordered exactly as listed below. Analyze the CURRENT frame as the main target; use previous/next frames only to understand continuity and before/after effects.",
        ]
        for index, item in enumerate(context_window, start=1):
            ocr_text = " ".join((context_ocr_texts.get(item.frame.number) or "").split())
            if len(ocr_text) > MAX_CONTEXT_OCR_CHARS:
                ocr_text = ocr_text[:MAX_CONTEXT_OCR_CHARS] + "..."
            role = "CURRENT" if item.frame.number == current_frame.number else item.role.upper()
            lines.append(
                f"[Image {index}] {role} frame {item.frame.number} at {item.frame.timestamp:.2f}s "
                f"(delta {item.gap_seconds:.2f}s)."
            )
            if ocr_text:
                lines.append(f"OCR context for image {index}: {ocr_text}")
        return "\n".join(lines)

    def reconstruct_video(self, frame_analyses: List[Dict[str, Any]], frames: List[Frame], 
                         transcript: Optional[AudioTranscript] = None) -> Dict[str, Any]:
        """Reconstruct video description from frame analyses and transcript."""
        frame_notes = []
        for i, (frame, analysis) in enumerate(zip(frames, frame_analyses)):
            frame_note = (
                f"Frame {i} ({frame.timestamp:.2f}s):\n"
                f"{analysis.get('response', 'No analysis available')}"
            )
            frame_notes.append(frame_note)
        
        analysis_text = "\n\n".join(frame_notes)
        
        # Get first frame analysis
        first_frame_text = ""
        if frame_analyses and len(frame_analyses) > 0:
            first_frame_text = frame_analyses[0].get('response', '')
        
        # Include transcript information if available
        transcript_text = ""
        if transcript and transcript.text.strip():
            transcript_text = transcript.text
        
        # Replace tokens in the prompt template
        prompt = self.video_prompt.replace("{prompt}", self._format_user_prompt())
        prompt = prompt.replace("{FRAME_NOTES}", analysis_text)
        prompt = prompt.replace("{FIRST_FRAME}", first_frame_text)
        prompt = prompt.replace("{TRANSCRIPT}", transcript_text)
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                num_predict=1000
            )
            logger.info("Successfully reconstructed video description")
            return {k: v for k, v in response.items() if k != "context"}
        except Exception as e:
            logger.error(f"Error reconstructing video: {e}")
            return {"response": f"Error reconstructing video: {str(e)}"}
