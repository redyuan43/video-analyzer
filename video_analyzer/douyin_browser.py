"""Browser-session Douyin downloader fallback.

Douyin's web detail API requires browser-generated request parameters that
yt-dlp can fail to reproduce. This module uses a copied Chrome profile to let
the page generate those parameters, then downloads the media URLs observed from
the page's own playback requests.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

DOUYIN_ID_RE = re.compile(r"/video/(?P<id>\d+)")
DOUYIN_HOST_MARKERS = ("douyin.com", "douyinvod.com")
VIDEO_URL_MARKER = "media-video-avc1"
AUDIO_URL_MARKER = "media-audio-und-mp4a"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)


class DouyinBrowserDownloadError(RuntimeError):
    """Raised when the browser-session fallback cannot materialize media."""


@dataclass
class DouyinBrowserResult:
    info: dict[str, Any]
    video_dir: Path
    video_path: Path


def is_douyin_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "douyin.com" in host or host == "v.douyin.com"


def infer_douyin_id(url: str) -> str:
    match = DOUYIN_ID_RE.search(urlparse(url).path)
    return match.group("id") if match else ""


def download_douyin_with_browser(url: str, output_root: Path, args: Any) -> DouyinBrowserResult:
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ModuleNotFoundError as exc:
        raise DouyinBrowserDownloadError("Douyin browser fallback requires `playwright` in the project environment") from exc
    return asyncio.run(_download_douyin_with_browser(url, output_root, args))


async def _download_douyin_with_browser(url: str, output_root: Path, args: Any) -> DouyinBrowserResult:
    from playwright.async_api import async_playwright

    chrome_path = shutil.which("google-chrome") or shutil.which("google-chrome-stable") or shutil.which("chromium")
    if not chrome_path:
        raise DouyinBrowserDownloadError("Chrome/Chromium is required for Douyin browser fallback")

    browser_name, profile_name = parse_browser_profile(getattr(args, "cookies_from_browser", "") or "chrome")
    source_profile = browser_profile_root(browser_name)
    if not source_profile.exists():
        raise DouyinBrowserDownloadError(f"Chrome profile root does not exist: {source_profile}")

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="video-analyzer-douyin-profile.") as profile_tmp:
        profile_root = Path(profile_tmp)
        copy_minimal_chrome_profile(source_profile, profile_root, profile_name)

        capture: dict[str, Any] = {
            "detail": None,
            "final_url": "",
            "title": "",
            "video_url": "",
            "audio_url": "",
            "errors": [],
        }

        async with async_playwright() as playwright:
            context = await launch_chrome_context(playwright, chrome_path, profile_root)
            page = context.pages[0] if context.pages else await context.new_page()

            page.on("request", lambda request: capture_media_request(capture, request.url))

            async def on_response(response: Any) -> None:
                response_url = response.url
                if "aweme/v1/web/aweme/detail" not in response_url:
                    return
                try:
                    payload = await response.json()
                except Exception as exc:  # pragma: no cover - depends on remote response shape
                    capture["errors"].append(f"failed to parse Douyin detail response: {exc}")
                    return
                detail = payload.get("aweme_detail")
                if isinstance(detail, dict):
                    capture["detail"] = detail

            page.on("response", lambda response: asyncio.create_task(on_response(response)))

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(4000)
                await page.mouse.click(640, 420)
                await page.evaluate(
                    """
                    () => Array.from(document.querySelectorAll('video')).forEach(v => {
                        try {
                            v.muted = true;
                            const promise = v.play();
                            if (promise && promise.catch) promise.catch(() => {});
                        } catch (e) {}
                    })
                    """
                )
                await wait_for_media_capture(capture, timeout_seconds=25)
                capture["title"] = await page.title()
                capture["final_url"] = page.url
            finally:
                await context.close()

        video_id = infer_douyin_id(capture["final_url"]) or infer_douyin_id(url) or str(int(time.time()))
        video_dir = output_root / safe_slug(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "video.mp4"
        materialize_captured_media(capture, video_path)
        info = aweme_detail_to_info(capture.get("detail") or {}, url, capture)
        if not info.get("id"):
            info["id"] = video_id
        (video_dir / "browser_download.json").write_text(
            json.dumps(
                {
                    "extractor": "douyin_browser",
                    "id": info.get("id"),
                    "title": info.get("title"),
                    "final_url": capture.get("final_url"),
                    "video_bytes": video_path.stat().st_size,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return DouyinBrowserResult(info=info, video_dir=video_dir, video_path=video_path)


async def launch_chrome_context(playwright: Any, chrome_path: str, profile_root: Path) -> Any:
    headless = not ensure_display_env()
    try:
        return await playwright.chromium.launch_persistent_context(
            str(profile_root),
            executable_path=chrome_path,
            headless=headless,
            viewport={"width": 1280, "height": 900},
            locale="zh-CN",
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-blink-features=AutomationControlled",
            ],
            ignore_default_args=["--enable-automation"],
        )
    except Exception:
        if headless:
            raise
        return await playwright.chromium.launch_persistent_context(
            str(profile_root),
            executable_path=chrome_path,
            headless=True,
            viewport={"width": 1280, "height": 900},
            locale="zh-CN",
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-blink-features=AutomationControlled",
            ],
            ignore_default_args=["--enable-automation"],
        )


def ensure_display_env() -> bool:
    if os.environ.get("DISPLAY"):
        return True
    if Path("/tmp/.X11-unix/X1").exists():
        os.environ["DISPLAY"] = ":1"
        return True
    return False


async def wait_for_media_capture(capture: dict[str, Any], timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if capture.get("video_url") and capture.get("audio_url"):
            return
        await asyncio.sleep(0.25)
    if not capture.get("video_url"):
        raise DouyinBrowserDownloadError("Douyin browser fallback did not observe a video media request")


def capture_media_request(capture: dict[str, Any], url: str) -> None:
    if VIDEO_URL_MARKER in url and not capture.get("video_url"):
        capture["video_url"] = url
    elif AUDIO_URL_MARKER in url and not capture.get("audio_url"):
        capture["audio_url"] = url


def materialize_captured_media(capture: dict[str, Any], video_path: Path) -> None:
    video_url = capture.get("video_url")
    audio_url = capture.get("audio_url")
    if not video_url:
        raise DouyinBrowserDownloadError("Douyin browser fallback did not capture a video URL")

    with tempfile.TemporaryDirectory(prefix="video-analyzer-douyin-media.") as tmp:
        tmp_dir = Path(tmp)
        video_part = tmp_dir / "video.mp4"
        audio_part = tmp_dir / "audio.mp4"
        download_signed_media(video_url, video_part)
        if audio_url:
            download_signed_media(audio_url, audio_part)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video_part),
                    "-i",
                    str(audio_part),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(video_path),
                ],
                check=True,
            )
        else:
            shutil.copy2(video_part, video_path)
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise DouyinBrowserDownloadError("Douyin browser fallback produced an empty media file")


def download_signed_media(url: str, output_path: Path) -> None:
    headers = {"Referer": "https://www.douyin.com/", "User-Agent": DEFAULT_USER_AGENT}
    with requests.get(url, headers=headers, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if output_path.stat().st_size <= 0:
        raise DouyinBrowserDownloadError(f"empty media response for {output_path.name}")


def aweme_detail_to_info(detail: dict[str, Any], original_url: str, capture: dict[str, Any]) -> dict[str, Any]:
    author = detail.get("author") if isinstance(detail.get("author"), dict) else {}
    video = detail.get("video") if isinstance(detail.get("video"), dict) else {}
    desc = str(detail.get("desc") or capture.get("title") or "Douyin video").strip()
    duration_ms = video.get("duration") or 0
    tags = []
    for item in detail.get("text_extra") or []:
        if isinstance(item, dict) and item.get("hashtag_name"):
            tags.append(str(item["hashtag_name"]))
    return {
        "id": str(detail.get("aweme_id") or infer_douyin_id(capture.get("final_url") or original_url) or ""),
        "display_id": str(detail.get("aweme_id") or ""),
        "title": desc,
        "description": desc,
        "duration": round(float(duration_ms) / 1000, 3) if duration_ms else None,
        "webpage_url": capture.get("final_url") or original_url,
        "extractor": "douyin_browser",
        "uploader": author.get("nickname") or "",
        "uploader_id": author.get("uid") or "",
        "channel_id": author.get("sec_uid") or "",
        "tags": tags,
        "comments": [],
    }


def parse_browser_profile(value: str) -> tuple[str, str]:
    browser = (value or "chrome").strip()
    browser = browser.split("+", 1)[0]
    if ":" in browser:
        name, profile = browser.split(":", 1)
    else:
        name, profile = browser, "Default"
    name = name.strip().lower() or "chrome"
    profile = profile.strip() or "Default"
    return name, profile


def browser_profile_root(browser_name: str) -> Path:
    home = Path.home()
    if browser_name == "chromium":
        return home / ".config" / "chromium"
    return home / ".config" / "google-chrome"


def copy_minimal_chrome_profile(source_root: Path, target_root: Path, profile_name: str) -> None:
    target_profile = target_root / profile_name
    target_profile.mkdir(parents=True, exist_ok=True)
    copy_if_exists(source_root / "Local State", target_root / "Local State")
    source_profile = source_root / profile_name
    if not source_profile.exists():
        raise DouyinBrowserDownloadError(f"Chrome profile does not exist: {source_profile}")
    for name in (
        "Cookies",
        "Network Persistent State",
        "Local Storage",
        "Session Storage",
        "IndexedDB",
        "WebStorage",
        "Preferences",
        "Secure Preferences",
    ):
        copy_if_exists(source_profile / name, target_profile / name)


def copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return slug or "douyin-video"
