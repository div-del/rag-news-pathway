"""
YouTube Video Analyzer.
Fetches YouTube videos related to news, downloads audio, transcribes with Whisper,
and analyzes with OpenRouter LLM.
"""

import asyncio
import logging
import tempfile
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import httpx

from config import Config

logger = logging.getLogger(__name__)

# Cache directory for downloaded audio
CACHE_DIR = Path(__file__).parent.parent / "cache" / "youtube"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class YouTubeSearchResult:
    """Represents a YouTube video search result"""
    video_id: str
    url: str
    title: str
    channel: str
    description: str
    thumbnail: Optional[str]
    duration: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAnalysisResult:
    """Result of video analysis"""
    video_url: str
    video_title: str
    transcript: str
    analysis: str
    key_points: List[str]
    sentiment: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class YouTubeAnalyzer:
    """
    Analyzes YouTube videos related to news articles.
    Pipeline: Serper Search → yt-dlp Download → Whisper Transcribe → OpenRouter Analyze
    """
    
    def __init__(
        self,
        api_key: str = None,
        openrouter_key: str = None,
        whisper_model: str = "base"
    ):
        self.api_key = api_key or Config.SERPER_API_KEY
        self.openrouter_key = openrouter_key or Config.OPENROUTER_API_KEY
        self.whisper_model = whisper_model
        self.base_url = "https://google.serper.dev/videos"
        
        # Lazy load whisper to avoid import time
        self._whisper = None
        self._whisper_model = None
        
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is required")
        if not self.openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is required")
    
    def _get_whisper(self):
        """Lazy load whisper model"""
        if self._whisper is None:
            import whisper
            self._whisper = whisper
            logger.info(f"Loading Whisper model: {self.whisper_model}")
            self._whisper_model = whisper.load_model(self.whisper_model)
        return self._whisper_model
    
    async def search_youtube_videos(
        self,
        query: str,
        num_results: int = 5
    ) -> List[YouTubeSearchResult]:
        """
        Search for YouTube videos using Serper API.
        
        Args:
            query: Search query (e.g., article title or topic)
            num_results: Number of results to return
        
        Returns:
            List of YouTubeSearchResult objects
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": f"{query} youtube",
            "num": num_results
        }
        
        results = []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                videos = data.get("videos", [])
                logger.info(f"Found {len(videos)} YouTube videos for query: {query}")
                
                # Fallback strategy if no videos found
                if not videos and len(query.split()) > 5:
                    logger.info("No videos found, trying simplified query...")
                    # Try with first 6 words
                    simplified_query = " ".join(query.split()[:6])
                    payload["q"] = f"{simplified_query} youtube"
                    
                    response = await client.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        videos = response.json().get("videos", [])
                        logger.info(f"Found {len(videos)} videos with simplified query: {simplified_query}")
                
                # Second fallback: very short query
                if not videos and len(query.split()) > 3:
                    logger.info("No videos found, trying keyword query...")
                    # Try with first 3 words
                    keyword_query = " ".join(query.split()[:3])
                    payload["q"] = f"{keyword_query} news youtube"
                    
                    response = await client.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        videos = response.json().get("videos", [])
                        logger.info(f"Found {len(videos)} videos with keyword query: {keyword_query}")

                for video in videos[:num_results]:
                    # Extract video ID from link
                    link = video.get("link", "")
                    video_id = ""
                    if "watch?v=" in link:
                        video_id = link.split("watch?v=")[1].split("&")[0]
                    elif "youtu.be/" in link:
                        video_id = link.split("youtu.be/")[1].split("?")[0]
                    
                    result = YouTubeSearchResult(
                        video_id=video_id,
                        url=link,
                        title=video.get("title", ""),
                        channel=video.get("channel", ""),
                        description=video.get("snippet", ""),
                        thumbnail=video.get("imageUrl"),
                        duration=video.get("duration")
                    )
                    results.append(result)
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error searching YouTube for '{query}': {e}")
        except Exception as e:
            logger.error(f"Error searching YouTube for '{query}': {e}")
        
        return results
    
    async def download_audio(self, youtube_url: str) -> Optional[str]:
        """
        Download audio from YouTube video using pytubefix (more reliable than yt-dlp for 403s).
        
        Args:
            youtube_url: YouTube video URL
        
        Returns:
            Path to downloaded MP3 file, or None if failed
        """
        try:
            from pytubefix import YouTube
            import hashlib
            import subprocess
            import os
            
            # Create unique filename based on video URL
            url_hash = hashlib.md5(youtube_url.encode()).hexdigest()[:12]
            final_mp3 = CACHE_DIR / f"{url_hash}.mp3"
            
            # Check if already cached
            if final_mp3.exists():
                logger.info(f"Using cached audio: {final_mp3}")
                return str(final_mp3)
            
            logger.info(f"Downloading audio from: {youtube_url} using pytubefix")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _download_and_convert():
                errors = []
                # Try different clients, some might work while others are blocked
                clients = ['WEB', 'ANDROID', 'IOS']
                
                for client in clients:
                    try:
                        logger.info(f"Trying download with client: {client}")
                        # Initialize YouTube object
                        yt = YouTube(youtube_url, client=client)
                        
                        # Check availability
                        try:
                            yt.check_availability()
                        except Exception as e:
                            logger.warning(f"Video unavailable with client {client}: {e}")
                            errors.append(f"{client}: {e}")
                            continue

                        stream = yt.streams.get_audio_only()
                        
                        if not stream:
                            logger.warning(f"No audio stream with client {client}")
                            errors.append(f"{client}: No audio stream")
                            continue
                            
                        # Download raw audio (usually m4a or webm)
                        raw_filename = f"{url_hash}_raw"
                        downloaded_path = stream.download(output_path=str(CACHE_DIR), filename=raw_filename)
                        
                        logger.info(f"Downloaded raw audio to: {downloaded_path}")
                        
                        # Convert to MP3 using ffmpeg
                        cmd = [
                            'ffmpeg',
                            '-i', downloaded_path,
                            '-vn',           # No video
                            '-acodec', 'libmp3lame', # MP3 encoder
                            '-q:a', '2',     # Good quality (V2)
                            '-y',            # Overwrite output
                            str(final_mp3)
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            logger.error(f"FFmpeg conversion failed: {result.stderr}")
                            # If conversion fails, try to return the raw file if it's audio
                            if downloaded_path.endswith('.m4a') or downloaded_path.endswith('.webm'):
                                logger.warning("Returning raw audio file due to conversion failure")
                                return downloaded_path
                            return None
                            
                        # Cleanup raw file
                        try:
                            os.remove(downloaded_path)
                        except:
                            pass
                            
                        return str(final_mp3)
                        
                    except Exception as e:
                        logger.warning(f"Error with client {client}: {e}")
                        errors.append(f"{client}: {e}")
                        continue
                
                logger.error(f"All pytubefix clients failed. Errors: {errors}")
                return None

            result_path = await loop.run_in_executor(None, _download_and_convert)
            
            if result_path and os.path.exists(result_path):
                logger.info(f"Audio ready (pytubefix): {result_path}")
                return result_path
            
            # FALLBACK: Try yt-dlp if pytubefix failed
            logger.info("pytubefix failed, falling back to yt-dlp...")
            
            import yt_dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': str(CACHE_DIR / f"{url_hash}.%(ext)s"),
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_color': True,
                'geo_bypass': True,
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'referer': 'https://www.google.com/',
            }
            
            def _download_ytdlp():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            
            await loop.run_in_executor(None, _download_ytdlp)
            
            if final_mp3.exists():
                logger.info(f"Audio downloaded (yt-dlp): {final_mp3}")
                return str(final_mp3)
                
            return None
            
        except ImportError as e:
            logger.error(f"Dependency missing: {e}. Ensure pytubefix and yt-dlp are installed.")
            return None
        except Exception as e:
            logger.error(f"Error in download_audio: {e}")
            return None
    
    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using OpenAI Whisper.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcribed text, or None if failed
        """
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            # Run whisper in executor to not block
            loop = asyncio.get_event_loop()
            
            def _transcribe():
                model = self._get_whisper()
                result = model.transcribe(audio_path)
                return result["text"]
            
            transcript = await loop.run_in_executor(None, _transcribe)
            logger.info(f"Transcription complete: {len(transcript)} characters")
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    async def analyze_transcript(
        self,
        transcript: str,
        article_title: str,
        article_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze transcript using OpenRouter LLM.
        
        Args:
            transcript: Video transcript
            article_title: Related news article title
            article_content: Optional article content for context
        
        Returns:
            Analysis dictionary with insights
        """
        logger.info("Analyzing transcript with OpenRouter")
        
        context = f"News Article: {article_title}"
        if article_content:
            context += f"\n\nArticle Summary: {article_content[:500]}..."
        
        prompt = f"""You are an expert news analyst. Analyze this YouTube video transcript in relation to the news article.

{context}

Video Transcript:
{transcript[:4000]}  # Limit to avoid token limits

Please provide:
1. **Summary**: A concise summary of the video's main points (2-3 sentences)
2. **Video Perspective**: What is the video creator's opinion/stance on this news?
3. **Key Points**: List 3-5 key insights from the video
4. **Sentiment**: Is the video positive, negative, or neutral about the news topic?
5. **Additional Context**: Any unique information or perspective the video adds that isn't in the news article

Format your response clearly with these sections."""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{Config.LLM_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "News AI Platform"
                    },
                    json={
                        "model": Config.LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a news and media analyst expert."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1500
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                
                analysis = data["choices"][0]["message"]["content"]
                
                # Parse key points (simple extraction)
                key_points = []
                for line in analysis.split("\n"):
                    if line.strip().startswith(("- ", "• ", "* ", "1.", "2.", "3.", "4.", "5.")):
                        point = line.strip().lstrip("-•* 0123456789.").strip()
                        if point and len(key_points) < 5:
                            key_points.append(point)
                
                # Detect sentiment
                sentiment = "neutral"
                lower_analysis = analysis.lower()
                if any(word in lower_analysis for word in ["positive", "optimistic", "bullish", "favorable"]):
                    sentiment = "positive"
                elif any(word in lower_analysis for word in ["negative", "pessimistic", "bearish", "critical"]):
                    sentiment = "negative"
                
                return {
                    "analysis": analysis,
                    "key_points": key_points,
                    "sentiment": sentiment
                }
                
        except Exception as e:
            logger.error(f"Error analyzing transcript: {e}")
            return {
                "analysis": "Analysis failed. Please try again.",
                "key_points": [],
                "sentiment": "unknown"
            }
    
    async def analyze_video(
        self,
        youtube_url: str,
        article_title: str,
        article_content: Optional[str] = None
    ) -> Optional[VideoAnalysisResult]:
        """
        Complete pipeline: Download → Transcribe → Analyze.
        
        Args:
            youtube_url: YouTube video URL
            article_title: Related news article title
            article_content: Optional article content
        
        Returns:
            VideoAnalysisResult or None if failed
        """
        logger.info(f"Starting full video analysis for: {youtube_url}")
        
        # Step 1: Download audio
        audio_path = await self.download_audio(youtube_url)
        if not audio_path:
            return None
        
        # Step 2: Transcribe
        transcript = await self.transcribe_audio(audio_path)
        if not transcript:
            return None
        
        # Step 3: Analyze
        analysis_result = await self.analyze_transcript(
            transcript,
            article_title,
            article_content
        )
        
        # Get video title from search results (if available)
        video_title = article_title  # Fallback
        
        return VideoAnalysisResult(
            video_url=youtube_url,
            video_title=video_title,
            transcript=transcript,
            analysis=analysis_result["analysis"],
            key_points=analysis_result["key_points"],
            sentiment=analysis_result["sentiment"]
        )
    
    def clear_cache(self):
        """Clear downloaded audio cache"""
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("YouTube audio cache cleared")


# Singleton instance
_youtube_analyzer: Optional[YouTubeAnalyzer] = None


def get_youtube_analyzer() -> YouTubeAnalyzer:
    """Get or create YouTubeAnalyzer singleton"""
    global _youtube_analyzer
    if _youtube_analyzer is None:
        _youtube_analyzer = YouTubeAnalyzer()
    return _youtube_analyzer


# Test function
async def test_analyzer():
    """Test the YouTube analyzer"""
    analyzer = get_youtube_analyzer()
    
    print("Testing YouTube Video Search...")
    results = await analyzer.search_youtube_videos("Tesla stock news", num_results=3)
    
    print(f"\nFound {len(results)} videos:")
    for r in results:
        print(f"  - {r.title} ({r.channel})")
        print(f"    URL: {r.url}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_analyzer())
