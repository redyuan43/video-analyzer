#!/usr/bin/env python3
import argparse
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
from http import HTTPStatus
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response, redirect
from werkzeug.utils import secure_filename

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.video_link_status_server import (  # noqa: E402
    DEFAULT_JOBS_DIR,
    REPO_ROOT as VIDEO_LINK_REPO_ROOT,
    BridgeError,
    STAGE_ORDER,
    VideoLinkStatusServer,
)
from video_analyzer_ui.runtime_identity import RuntimeIdentity  # noqa: E402
from video_analyzer_ui.audio_tenants import AudioTenantRegistry  # noqa: E402
from web_debug_console import WebDebugConsole  # noqa: E402

# Initialize logger
logger = logging.getLogger(__name__)

class VideoAnalyzerUI:
    def __init__(
        self,
        host='localhost',
        port=5000,
        dev_mode=False,
        jobs_dir=DEFAULT_JOBS_DIR,
        video_link_auto_resume=None,
        debug_console_enabled=True,
    ):
        package_dir = Path(__file__).resolve().parent
        self.app = Flask(
            __name__,
            template_folder=str(package_dir / 'templates'),
            static_folder=str(package_dir / 'static'),
        )
        self.host = host
        self.port = port
        self.dev_mode = dev_mode
        self.sessions = {}
        self.runtime_identity = RuntimeIdentity(VIDEO_LINK_REPO_ROOT)
        self.audio_tenants = AudioTenantRegistry()
        resolved_jobs_dir = Path(jobs_dir).resolve()
        if video_link_auto_resume is None:
            video_link_auto_resume = resolved_jobs_dir == DEFAULT_JOBS_DIR.resolve()
        self.video_link = VideoLinkStatusServer(
            resolved_jobs_dir,
            VIDEO_LINK_REPO_ROOT,
            auto_resume=bool(video_link_auto_resume),
        )
        self.debug_console = WebDebugConsole(
            self.app,
            VIDEO_LINK_REPO_ROOT,
            context_provider=self.debug_console_context,
            enabled=debug_console_enabled,
        )
        
        # Ensure tmp directories exist
        self.tmp_root = Path(tempfile.gettempdir()) / 'video-analyzer-ui'
        self.uploads_dir = self.tmp_root / 'uploads'
        self.results_dir = self.tmp_root / 'results'
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_routes()
        
    def setup_routes(self):
        def require_mobile_audio_tenant():
            supplied = request.headers.get('X-Audio-Pipeline-Token', '').strip()
            try:
                tenant_id = self.audio_tenants.resolve(supplied)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.error("audio tenant registry is invalid: %s", exc)
                return None, (
                    jsonify({'error': 'audio tenant registry is unavailable'}),
                    int(HTTPStatus.SERVICE_UNAVAILABLE),
                )
            if tenant_id is None:
                return None, (
                    jsonify({'error': 'audio pipeline token is invalid'}),
                    int(HTTPStatus.UNAUTHORIZED),
                )
            return tenant_id, None

        def has_video_link_admin_token():
            expected = os.environ.get('VIDEO_ANALYZER_ADMIN_TOKEN', '').strip()
            if not expected:
                return False
            supplied = request.headers.get('X-Video-Analyzer-Admin-Token', '').strip()
            authorization = request.headers.get('Authorization', '').strip()
            if authorization.lower().startswith('bearer '):
                supplied = authorization[7:].strip()
            return hmac.compare_digest(supplied, expected)

        @self.app.before_request
        def protect_mobile_audio_from_generic_routes():
            if not request.path.startswith('/api/video-link/jobs/'):
                return None
            match = re.match(r"^/api/video-link/jobs/([a-f0-9]{32})(?:/|$)", request.path)
            if not match:
                return None
            try:
                job = self.video_link.load_job(match.group(1))
            except Exception:
                return None
            if not self.video_link.is_tenant_mobile_audio_job(job):
                return None
            if request.method == 'GET' and not request.headers.get(
                'X-Audio-Pipeline-Token', ''
            ).strip():
                return None
            if has_video_link_admin_token():
                return None
            return jsonify({'error': 'job not found'}), int(HTTPStatus.NOT_FOUND)

        @self.app.before_request
        def reject_mutations_from_stale_runtime():
            if request.method not in {'POST', 'PUT', 'PATCH', 'DELETE'}:
                return None
            if not request.path.startswith(
                (
                    '/api/video-link/',
                    '/api/skills',
                    '/api/skill-projects',
                    '/api/settings',
                    '/api/mobile/audio-jobs',
                    '/api/mobile/audio-transcriptions',
                )
            ):
                return None
            if request.path.endswith('/stop'):
                return None
            runtime = self.runtime_identity.payload()
            if not runtime['source_stale']:
                return None
            return jsonify(
                {
                    'error': 'status service source code changed; waiting for supervised restart',
                    'runtime': runtime,
                }
            ), int(HTTPStatus.SERVICE_UNAVAILABLE)

        @self.app.route('/')
        def index():
            static_root = Path(self.app.static_folder or '')
            try:
                assets = [
                    static_root / 'js' / 'main.js',
                    static_root / 'css' / 'styles.css',
                    static_root / 'data' / 'audio_prompt_templates.json',
                    static_root / 'vendor' / 'markdown-it' / 'markdown-it.min.js',
                    static_root / 'vendor' / 'dompurify' / 'purify.min.js',
                    static_root / 'vendor' / 'katex' / 'katex.min.css',
                    static_root / 'vendor' / 'katex' / 'katex.min.js',
                    static_root / 'vendor' / 'katex' / 'contrib' / 'auto-render.min.js',
                    REPO_ROOT / 'web_debug_console' / 'static' / 'debug-console.js',
                    REPO_ROOT / 'web_debug_console' / 'static' / 'debug-console.css',
                ]
                static_version = int(max(path.stat().st_mtime for path in assets))
            except OSError:
                static_version = 1
            return render_template(
                'index.html',
                static_version=static_version,
                debug_console_token=self.debug_console.token,
            )

        @self.app.route('/video-link')
        def video_link_home():
            return redirect('/', code=int(HTTPStatus.FOUND))

        @self.app.route('/video-link/jobs/<job_id>')
        def video_link_job_redirect(job_id):
            return redirect(f'/?job={job_id}', code=int(HTTPStatus.FOUND))

        @self.app.route('/api/video-link/health')
        def video_link_health():
            runtime = self.runtime_identity.payload()
            return jsonify(
                {
                    'ok': not runtime['source_stale'],
                    'stages': STAGE_ORDER,
                    'runtime': runtime,
                    'activity': self.video_link.runtime_activity(),
                    'background_workers': self.video_link.background_worker_status(),
                }
            )

        @self.app.route('/api/video-link/options')
        def video_link_options():
            return jsonify(self.video_link.options())

        @self.app.route('/api/settings')
        def settings_get():
            return jsonify(self.video_link.settings())

        @self.app.route('/api/settings/models', methods=['POST'])
        def settings_create_model():
            payload = request.get_json(silent=True) or {}
            model_id = str(payload.get('id') or '').strip()
            try:
                return jsonify(self.video_link.save_model_setting(model_id, payload)), int(HTTPStatus.CREATED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/models/<model_id>', methods=['PUT'])
        def settings_update_model(model_id):
            try:
                return jsonify(self.video_link.save_model_setting(model_id, request.get_json(silent=True) or {}))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/models/<model_id>', methods=['DELETE'])
        def settings_delete_model(model_id):
            try:
                return jsonify(self.video_link.delete_model_setting(model_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/models/<model_id>/test', methods=['POST'])
        def settings_test_model(model_id):
            try:
                return jsonify(
                    self.video_link.test_model_setting(
                        model_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/models/<model_id>/tts-preview', methods=['POST'])
        def settings_tts_preview(model_id):
            try:
                audio, metadata = self.video_link.preview_tts_setting(
                    model_id,
                    request.get_json(silent=True) or {},
                )
                return Response(
                    audio,
                    mimetype='audio/wav',
                    headers={
                        'Content-Disposition': 'inline; filename="tts-preview.wav"',
                        'X-TTS-Voice': str(metadata.get('voice') or ''),
                        'X-TTS-Text-Chars': str(metadata.get('text_chars') or 0),
                    },
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/profile-test', methods=['POST'])
        def settings_test_profile():
            try:
                return jsonify(
                    self.video_link.test_profile_setting(
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/profiles', methods=['POST'])
        def settings_create_profile():
            payload = request.get_json(silent=True) or {}
            profile_name = str(payload.get('name') or '').strip()
            try:
                return jsonify(self.video_link.save_profile_setting(profile_name, payload)), int(HTTPStatus.CREATED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/profiles/<profile_name>', methods=['PUT'])
        def settings_update_profile(profile_name):
            try:
                return jsonify(self.video_link.save_profile_setting(profile_name, request.get_json(silent=True) or {}))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/profiles/<profile_name>', methods=['DELETE'])
        def settings_delete_profile(profile_name):
            try:
                return jsonify(self.video_link.delete_profile_setting(profile_name))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/settings/profiles/<profile_name>/activate', methods=['POST'])
        def settings_activate_profile(profile_name):
            try:
                return jsonify(self.video_link.activate_profile_setting(profile_name))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs')
        def video_link_jobs():
            limit = request.args.get('limit', default=50, type=int)
            return jsonify(
                self.video_link.list_jobs(limit, include_mobile_audio=False)
            )

        @self.app.route('/api/operator/audio-tenants')
        def operator_audio_tenants():
            tenant_ids = set(self.audio_tenants.tenant_ids())
            for path in self.video_link.jobs_dir.glob('*/job.json'):
                try:
                    job = self.video_link.load_job(path.parent.name)
                except Exception:
                    continue
                if self.video_link.is_tenant_mobile_audio_job(job):
                    tenant_ids.add(self.video_link.mobile_audio_tenant_id(job))
            return jsonify(
                {
                    'tenants': [
                        {'id': tenant_id, 'label': tenant_id.upper()}
                        for tenant_id in sorted(tenant_ids)
                    ]
                }
            )

        @self.app.route('/api/operator/audio-jobs')
        def operator_audio_jobs():
            limit = request.args.get('limit', default=50, type=int)
            tenant_id = str(request.args.get('tenant_id') or '').strip().lower()
            return jsonify(
                self.video_link.list_operator_audio_jobs(
                    limit,
                    tenant_id or None,
                )
            )

        @self.app.route('/api/mobile/audio-jobs')
        def mobile_audio_jobs():
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            limit = request.args.get('limit', default=50, type=int)
            return jsonify(self.video_link.list_mobile_audio_jobs(limit, tenant_id))

        @self.app.route('/api/mobile/audio-jobs/by-attempt/<external_attempt_id>')
        def mobile_audio_job_by_attempt(external_attempt_id):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                return jsonify(
                    self.video_link.get_mobile_audio_job_by_attempt(
                        external_attempt_id,
                        tenant_id,
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/mobile/audio-jobs/<job_id>')
        def mobile_audio_job(job_id):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                return jsonify(self.video_link.get_mobile_audio_job(job_id, tenant_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/mobile/audio-templates')
        def mobile_audio_templates():
            _tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                return jsonify(self.video_link.mobile_audio_templates())
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/mobile/audio-jobs/upload', methods=['POST'])
        def mobile_audio_upload_job():
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            media = request.files.get('media')
            if not media or not media.filename:
                return jsonify({'error': 'media file is required'}), int(HTTPStatus.BAD_REQUEST)
            temp_dir = self.tmp_root / 'mobile-audio-uploads'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{uuid.uuid4().hex}-{secure_filename(media.filename)}"
            try:
                media.save(temp_path)
                payload = dict(request.form.items())
                return (
                    jsonify(
                        self.video_link.create_mobile_audio_job(
                            payload,
                            temp_path,
                            media.filename,
                            tenant_id=tenant_id,
                        )
                    ),
                    int(HTTPStatus.CREATED),
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)
            finally:
                temp_path.unlink(missing_ok=True)

        @self.app.route('/api/mobile/audio-jobs/from-transcript', methods=['POST'])
        def mobile_audio_job_from_transcript():
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            transcript = request.files.get('transcript') or request.files.get('transcript_json')
            if not transcript or not transcript.filename:
                return jsonify({'error': 'transcript JSON file is required'}), int(HTTPStatus.BAD_REQUEST)
            temp_dir = self.tmp_root / 'mobile-transcript-uploads'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{uuid.uuid4().hex}-{secure_filename(transcript.filename)}"
            try:
                transcript.save(temp_path)
                payload = dict(request.form.items())
                return (
                    jsonify(
                        self.video_link.create_mobile_transcript_job(
                            payload,
                            temp_path,
                            transcript.filename,
                            tenant_id=tenant_id,
                        )
                    ),
                    int(HTTPStatus.CREATED),
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)
            finally:
                temp_path.unlink(missing_ok=True)

        @self.app.route('/api/mobile/audio-jobs/<job_id>/ack', methods=['POST'])
        def mobile_audio_ack(job_id):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                return jsonify(
                    self.video_link.acknowledge_mobile_audio_job(job_id, tenant_id)
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route(
            '/api/mobile/audio-jobs/<job_id>/background-tasks/tts/ack',
            methods=['POST'],
        )
        def mobile_audio_tts_ack(job_id):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                return jsonify(
                    self.video_link.acknowledge_mobile_audio_tts(
                        job_id,
                        tenant_id,
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/mobile/audio-jobs/<job_id>/resources/<path:resource_path>')
        def mobile_audio_resource(job_id, resource_path):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                self.video_link.get_mobile_audio_job(job_id, tenant_id)
                file_path, _content_type = self.video_link.resource_file(job_id, resource_path)
                return send_file(file_path, as_attachment=False)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/mobile/audio-transcriptions/by-attempt/<external_attempt_id>')
        def mobile_audio_transcription_by_attempt(external_attempt_id):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                return jsonify(
                    self.video_link.get_mobile_audio_job_by_attempt(
                        external_attempt_id,
                        tenant_id,
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/mobile/audio-transcriptions/<job_id>')
        def mobile_audio_transcription(job_id):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                return jsonify(self.video_link.get_mobile_audio_job(job_id, tenant_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/mobile/audio-transcriptions/upload', methods=['POST'])
        def mobile_audio_transcription_upload():
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            media = request.files.get('media')
            if not media or not media.filename:
                return jsonify({'error': 'media file is required'}), int(HTTPStatus.BAD_REQUEST)
            temp_dir = self.tmp_root / 'mobile-audio-uploads'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{uuid.uuid4().hex}-{secure_filename(media.filename)}"
            try:
                media.save(temp_path)
                payload = dict(request.form.items())
                return (
                    jsonify(
                        self.video_link.create_mobile_audio_job(
                            payload,
                            temp_path,
                            media.filename,
                            pipeline_kind='transcription',
                            tenant_id=tenant_id,
                        )
                    ),
                    int(HTTPStatus.CREATED),
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)
            finally:
                temp_path.unlink(missing_ok=True)

        @self.app.route('/api/mobile/audio-transcriptions/<job_id>/ack', methods=['POST'])
        def mobile_audio_transcription_ack(job_id):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                return jsonify(
                    self.video_link.acknowledge_mobile_audio_job(job_id, tenant_id)
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/mobile/audio-transcriptions/<job_id>/resources/<path:resource_path>')
        def mobile_audio_transcription_resource(job_id, resource_path):
            tenant_id, denied = require_mobile_audio_tenant()
            if denied:
                return denied
            try:
                self.video_link.get_mobile_audio_job(job_id, tenant_id)
                file_path, _content_type = self.video_link.resource_file(job_id, resource_path)
                return send_file(file_path, as_attachment=False)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs', methods=['POST'])
        def video_link_create_job():
            try:
                return jsonify(self.video_link.create_job(request.get_json(silent=True) or {})), int(HTTPStatus.CREATED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/upload', methods=['POST'])
        def video_link_create_upload_job():
            media = request.files.get('media')
            if not media or not media.filename:
                return jsonify({'error': 'media file is required'}), int(HTTPStatus.BAD_REQUEST)
            temp_dir = self.tmp_root / 'video-link-uploads'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{uuid.uuid4().hex}-{secure_filename(media.filename)}"
            try:
                media.save(temp_path)
                payload = dict(request.form.items())
                return (
                    jsonify(self.video_link.create_uploaded_media_job(payload, temp_path, media.filename)),
                    int(HTTPStatus.CREATED),
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)
            finally:
                temp_path.unlink(missing_ok=True)

        @self.app.route('/api/video-link/jobs/batch', methods=['POST'])
        def video_link_create_jobs():
            try:
                return jsonify(self.video_link.create_jobs(request.get_json(silent=True) or {})), int(HTTPStatus.CREATED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>')
        def video_link_get_job(job_id):
            try:
                public_host = request.host.split(':', 1)[0]
                return jsonify(self.video_link.public_job(self.video_link.load_job(job_id), public_host=public_host))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>', methods=['DELETE'])
        def video_link_delete_job(job_id):
            try:
                return jsonify(self.video_link.delete_job(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/run', methods=['POST'])
        def video_link_run_job(job_id):
            try:
                payload = request.get_json(silent=True) or {}
                return jsonify(self.video_link.start_run(job_id, profile=payload.get('profile'))), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/stop', methods=['POST'])
        def video_link_stop_job(job_id):
            try:
                return jsonify(self.video_link.stop_job(job_id)), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/open-run-dir', methods=['POST'])
        def video_link_open_run_dir(job_id):
            try:
                return jsonify(self.video_link.open_run_dir(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/vscode-session', methods=['POST'])
        def video_link_vscode_session(job_id):
            try:
                payload = request.get_json(silent=True) or {}
                public_host = request.host.split(':', 1)[0]
                restart = str(payload.get('restart', 'false')).lower() in {'1', 'true', 'yes', 'on'}
                return jsonify(self.video_link.start_vscode_session(job_id, public_host=public_host, restart=restart))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/vscode-session', methods=['DELETE'])
        def video_link_stop_vscode_session(job_id):
            try:
                return jsonify(self.video_link.stop_vscode_session(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/stages/<stage>', methods=['POST'])
        def video_link_run_stage(job_id, stage):
            try:
                return jsonify(self.video_link.run_stage(job_id, stage))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/stages/<stage>/rerun', methods=['POST'])
        def video_link_rerun_from_stage(job_id, stage):
            try:
                payload = request.get_json(silent=True) or {}
                return jsonify(
                    self.video_link.rerun_from_stage(
                        job_id,
                        stage,
                        profile=payload.get('profile'),
                        refresh_runtime_profile=str(
                            payload.get('refresh_runtime_profile', 'false')
                        ).lower() in {'1', 'true', 'yes', 'on'},
                    )
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/logs/<stage>')
        def video_link_stage_log(job_id, stage):
            try:
                limit = request.args.get('tail', default=80, type=int)
                full = str(request.args.get('full', 'false')).lower() in {'1', 'true', 'yes', 'on'}
                return jsonify(self.video_link.stage_log(job_id, stage, limit, full))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/video')
        def video_link_preview_video(job_id):
            try:
                video_path, mimetype = self.video_link.preview_video_file(job_id)
                return send_file(video_path, mimetype=mimetype, conditional=True)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/resource')
        def video_link_resource_file(job_id):
            try:
                file_path, mimetype = self.video_link.resource_file(job_id, request.args.get('path', ''))
                return send_file(file_path, mimetype=mimetype, conditional=True)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/resources/<path:relative_path>')
        def video_link_resource_path(job_id, relative_path):
            try:
                file_path, mimetype = self.video_link.resource_file(job_id, relative_path)
                return send_file(file_path, mimetype=mimetype, conditional=True)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/study-guide')
        def video_link_study_guide(job_id):
            try:
                return jsonify(self.video_link.study_guide(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/frame-time-map')
        def video_link_frame_time_map(job_id):
            try:
                return jsonify(self.video_link.frame_time_map(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/qa-index')
        def video_link_qa_index(job_id):
            try:
                return jsonify(self.video_link.qa_index(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/qa/history')
        def video_link_qa_history(job_id):
            try:
                limit = int(request.args.get('limit', '50'))
                return jsonify(self.video_link.qa_history(job_id, limit=limit))
            except ValueError:
                return jsonify({'error': 'limit must be an integer'}), int(HTTPStatus.BAD_REQUEST)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/web-evidence')
        def video_link_web_evidence(job_id):
            try:
                return jsonify(self.video_link.web_evidence(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/qa/ask', methods=['POST'])
        def video_link_qa_ask(job_id):
            try:
                return jsonify(self.video_link.ask_qa(job_id, request.get_json(silent=True) or {}))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-candidate')
        def video_link_skill_candidate(job_id):
            try:
                return jsonify(self.video_link.skill_candidate(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation')
        def video_link_skill_distillation(job_id):
            try:
                return jsonify(self.video_link.skill_candidate(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation/workspace')
        def video_link_skill_distillation_workspace(job_id):
            try:
                return jsonify(self.video_link.skill_distillation_workspace(job_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation/items/<path:item_id>')
        def video_link_skill_distillation_item(job_id, item_id):
            try:
                return jsonify(self.video_link.skill_distillation_item(job_id, item_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation/start', methods=['POST'])
        def video_link_start_skill_distillation(job_id):
            try:
                return jsonify(
                    self.video_link.start_skill_distillation(
                        job_id,
                        request.get_json(silent=True) or {},
                    )
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation/review-overview', methods=['POST'])
        def video_link_review_skill_overview(job_id):
            try:
                return jsonify(
                    self.video_link.review_skill_distillation_overview(
                        job_id,
                        request.get_json(silent=True) or {},
                    )
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation/review-candidates', methods=['POST'])
        def video_link_review_skill_candidates(job_id):
            try:
                return jsonify(
                    self.video_link.review_skill_distillation_candidates(
                        job_id,
                        request.get_json(silent=True) or {},
                    )
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation/resume', methods=['POST'])
        def video_link_resume_skill_distillation(job_id):
            try:
                return jsonify(
                    self.video_link.resume_skill_distillation(job_id)
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation/cancel', methods=['POST'])
        def video_link_cancel_skill_distillation(job_id):
            try:
                return jsonify(
                    self.video_link.cancel_skill_distillation(job_id)
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-distillation/enable', methods=['POST'])
        def video_link_enable_skill_distillation(job_id):
            try:
                return jsonify(
                    self.video_link.enable_skill_distillation(
                        job_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-candidate/generate', methods=['POST'])
        def video_link_generate_skill_candidate(job_id):
            try:
                return jsonify(
                    self.video_link.generate_skill_candidate(
                        job_id,
                        request.get_json(silent=True) or {},
                    )
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/video-link/jobs/<job_id>/skill-candidate/enable', methods=['POST'])
        def video_link_enable_skill_candidate(job_id):
            try:
                return jsonify(
                    self.video_link.enable_skill_candidate(
                        job_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects')
        def skill_projects_list():
            try:
                return jsonify(self.video_link.list_skill_projects())
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/workbench')
        def skill_projects_workbench():
            try:
                return jsonify(
                    self.video_link.skill_project_workbench(
                        request.args.get('project_id', '')
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects', methods=['POST'])
        def skill_projects_create():
            try:
                return jsonify(
                    self.video_link.create_skill_project(request.get_json(silent=True) or {})
                ), int(HTTPStatus.CREATED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>')
        def skill_projects_get(project_id):
            try:
                return jsonify(self.video_link.get_skill_project(project_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>', methods=['PATCH'])
        def skill_projects_update(project_id):
            try:
                return jsonify(
                    self.video_link.update_skill_project(
                        project_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/sources', methods=['POST'])
        def skill_projects_add_source(project_id):
            try:
                return jsonify(
                    self.video_link.add_skill_project_source(
                        project_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/packages/preview')
        def skill_projects_preview_package(project_id):
            try:
                return jsonify(
                    self.video_link.preview_skill_project_package(
                        project_id,
                        request.args.get('package_id', ''),
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/packages', methods=['POST'])
        def skill_projects_import_package(project_id):
            try:
                return jsonify(
                    self.video_link.import_skill_project_package(
                        project_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/sources/<source_id>', methods=['DELETE'])
        def skill_projects_remove_source(project_id, source_id):
            try:
                return jsonify(self.video_link.remove_skill_project_source(project_id, source_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/assess', methods=['POST'])
        def skill_projects_assess(project_id):
            try:
                return jsonify(
                    self.video_link.assess_skill_project(
                        project_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route(
            '/api/skill-projects/<project_id>/capability-checks/<check_id>/run',
            methods=['POST'],
        )
        def skill_projects_run_capability_check(project_id, check_id):
            try:
                return jsonify(
                    self.video_link.run_skill_project_capability_check(
                        project_id,
                        check_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/workspace')
        def skill_projects_workspace(project_id):
            try:
                return jsonify(self.video_link.skill_project_workspace(project_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/resource')
        def skill_projects_resource(project_id):
            try:
                file_path, mimetype = self.video_link.skill_project_resource_file(
                    project_id,
                    request.args.get('path', ''),
                )
                return send_file(file_path, mimetype=mimetype, conditional=True)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/distillation/start', methods=['POST'])
        def skill_projects_start(project_id):
            try:
                return jsonify(
                    self.video_link.start_skill_project_distillation(
                        project_id,
                        request.get_json(silent=True) or {},
                    )
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/distillation/review-overview', methods=['POST'])
        def skill_projects_review_overview(project_id):
            try:
                return jsonify(
                    self.video_link.review_skill_project_overview(
                        project_id,
                        request.get_json(silent=True) or {},
                    )
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/distillation/review-candidates', methods=['POST'])
        def skill_projects_review_candidates(project_id):
            try:
                return jsonify(
                    self.video_link.review_skill_project_candidates(
                        project_id,
                        request.get_json(silent=True) or {},
                    )
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/distillation/resume', methods=['POST'])
        def skill_projects_resume(project_id):
            try:
                return jsonify(
                    self.video_link.resume_skill_project_distillation(project_id)
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/distillation/cancel', methods=['POST'])
        def skill_projects_cancel(project_id):
            try:
                return jsonify(
                    self.video_link.cancel_skill_project_distillation(project_id)
                ), int(HTTPStatus.ACCEPTED)
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skill-projects/<project_id>/distillation/enable', methods=['POST'])
        def skill_projects_enable(project_id):
            try:
                return jsonify(
                    self.video_link.enable_skill_project_distillation(
                        project_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skills')
        def skills_list():
            try:
                return jsonify(
                    self.video_link.list_skills(
                        request.args.get('state', 'enabled'),
                        request.args.get('query', ''),
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skills/<state>/<skill_id>')
        def skills_detail(state, skill_id):
            try:
                return jsonify(self.video_link.get_skill(state, skill_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skills/<state>/<skill_id>', methods=['PUT'])
        def skills_update(state, skill_id):
            try:
                return jsonify(
                    self.video_link.update_skill(
                        state,
                        skill_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skills/enabled/<name>/disable', methods=['POST'])
        def skills_disable(name):
            try:
                return jsonify(self.video_link.disable_skill(name))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skills/disabled/<name>/restore', methods=['POST'])
        def skills_restore_disabled(name):
            try:
                return jsonify(self.video_link.restore_disabled_skill(name))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skills/trash/<skill_id>/restore', methods=['POST'])
        def skills_restore_trash(skill_id):
            try:
                return jsonify(self.video_link.restore_trash_skill(skill_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skills/<state>/<skill_id>', methods=['DELETE'])
        def skills_delete(state, skill_id):
            try:
                return jsonify(
                    self.video_link.delete_skill(
                        state,
                        skill_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route('/api/skills/<state>/<skill_id>/versions')
        def skills_versions(state, skill_id):
            try:
                return jsonify(self.video_link.skill_versions(state, skill_id))
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)

        @self.app.route(
            '/api/skills/<state>/<skill_id>/versions/<version_id>/restore',
            methods=['POST'],
        )
        def skills_restore_version(state, skill_id, version_id):
            try:
                return jsonify(
                    self.video_link.restore_skill_version(
                        state,
                        skill_id,
                        version_id,
                        request.get_json(silent=True) or {},
                    )
                )
            except BridgeError as exc:
                return jsonify({'error': exc.message}), int(exc.status)
            
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
                
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return jsonify({'error': 'Invalid file type'}), 400
                
            try:
                # Create session
                session_id = str(uuid.uuid4())
                session_upload_dir = self.uploads_dir / session_id
                session_results_dir = self.results_dir / session_id
                session_upload_dir.mkdir(parents=True)
                session_results_dir.mkdir(parents=True)
                
                # Save file
                filename = secure_filename(file.filename)
                filepath = session_upload_dir / filename
                file.save(filepath)
                
                self.sessions[session_id] = {
                    'video_path': str(filepath),
                    'results_dir': str(session_results_dir),
                    'filename': filename
                }
                
                return jsonify({
                    'session_id': session_id,
                    'message': 'File uploaded successfully'
                })
                
            except Exception as e:
                logger.error(f"Upload error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/analyze/<session_id>', methods=['POST'])
        def analyze(session_id):
            if session_id not in self.sessions:
                return jsonify({'error': 'Invalid session'}), 404
                
            session = self.sessions[session_id]
            
            # Build command
            cmd = ['video-analyzer', session['video_path']]
            
            # Add optional parameters
            for param, value in request.form.items():
                if value:  # Only add parameters with values
                    if param in ['keep-frames', 'dev']:  # Flags without values
                        cmd.append(f'--{param}')
                    else:
                        cmd.extend([f'--{param}', value])
                        
            # Create output directory if it doesn't exist
            results_dir = Path(session['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Add output directory
            cmd.extend(['--output', str(results_dir)])
            
            # Store output directory in session for later use
            session['output_dir'] = str(results_dir)
            logger.debug(f"Set output directory to: {results_dir}")
            
            # Store command in session for streaming
            session['cmd'] = cmd
            
            return jsonify({'message': 'Analysis started'})
            
        @self.app.route('/analyze/<session_id>/stream')
        def stream_output(session_id):
            if session_id not in self.sessions:
                return jsonify({'error': 'Invalid session'}), 404
                
            session = self.sessions[session_id]
            if 'cmd' not in session:
                return jsonify({'error': 'Analysis not started'}), 400
                
            def generate_output():
                logger.debug(f"Starting analysis with command: {' '.join(session['cmd'])}")
                try:
                    process = subprocess.Popen(
                        session['cmd'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1
                    )
                    
                    for line in process.stdout:
                        line = line.strip()
                        if line:  # Only send non-empty lines
                            logger.debug(f"Output: {line}")
                            yield f"data: {line}\n\n"
                    
                    process.wait()
                    if process.returncode == 0:
                        logger.info("Analysis completed successfully")
                        yield f"data: Analysis completed successfully\n\n"
                    else:
                        logger.error(f"Analysis failed with code {process.returncode}")
                        yield f"data: Analysis failed with code {process.returncode}\n\n"
                except Exception as e:
                    logger.error(f"Error during analysis: {e}")
                    yield f"data: Error during analysis: {str(e)}\n\n"
                    yield f"data: Analysis failed\n\n"
                    
            return Response(
                generate_output(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
            
        @self.app.route('/results/<session_id>')
        def get_results(session_id):
            if session_id not in self.sessions:
                return jsonify({'error': 'Invalid session'}), 404
                
            session = self.sessions[session_id]
            results_dir = Path(session['results_dir'])
            logger.debug(f"Looking for results in: {results_dir}")
            
            if not results_dir.exists():
                logger.error(f"Results directory not found: {results_dir}")
                return jsonify({'error': 'Results directory not found'}), 404
            
            # List all files in results directory for debugging
            logger.debug("Files in results directory:")
            for file in results_dir.glob('**/*'):
                logger.debug(f"- {file}")
            
            # Check both the results directory and the default 'output' directory
            analysis_file = results_dir / 'analysis.json'
            default_output = Path('output/analysis.json')
            
            if default_output.exists():
                logger.debug(f"Found analysis file in default output directory: {default_output}")
                try:
                    # Move the file to our results directory
                    default_output.rename(analysis_file)
                    logger.debug(f"Moved analysis file to: {analysis_file}")
                except Exception as e:
                    logger.error(f"Error moving analysis file: {e}")
                    # If move fails, try to copy the content
                    try:
                        analysis_file.write_text(default_output.read_text())
                        logger.debug("Copied analysis file content")
                        default_output.unlink()
                    except Exception as copy_error:
                        logger.error(f"Error copying analysis file: {copy_error}")
                        return jsonify({'error': 'Error accessing analysis file'}), 500
            if not analysis_file.exists():
                logger.error(f"Analysis file not found: {analysis_file}")
                return jsonify({'error': 'Analysis file not found'}), 404
                
            try:
                return send_file(
                    analysis_file,
                    mimetype='application/json',
                    as_attachment=True,
                    download_name=f"analysis_{session['filename']}.json"
                )
            except Exception as e:
                logger.error(f"Error sending file: {e}")
                return jsonify({'error': f'Error sending file: {str(e)}'}), 500
            
        @self.app.route('/cleanup/<session_id>', methods=['POST'])
        def cleanup_session(session_id):
            if session_id not in self.sessions:
                return jsonify({'error': 'Invalid session'}), 404
                
            try:
                session = self.sessions[session_id]
                # Clean up upload directory
                upload_dir = Path(session['video_path']).parent
                if upload_dir.exists():
                    for file in upload_dir.glob('*'):
                        file.unlink()
                    upload_dir.rmdir()
                
                # Clean up results directory
                results_dir = Path(session['results_dir'])
                if results_dir.exists():
                    for file in results_dir.glob('**/*'):
                        if file.is_file():
                            file.unlink()
                    for dir_path in sorted(results_dir.glob('**/*'), reverse=True):
                        if dir_path.is_dir():
                            dir_path.rmdir()
                    results_dir.rmdir()
                
                # Clean up default output directory if it exists
                default_output_dir = Path('output')
                if default_output_dir.exists():
                    for file in default_output_dir.glob('**/*'):
                        if file.is_file():
                            file.unlink()
                    for dir_path in sorted(default_output_dir.glob('**/*'), reverse=True):
                        if dir_path.is_dir():
                            dir_path.rmdir()
                    default_output_dir.rmdir()
                
                del self.sessions[session_id]
                return jsonify({'message': 'Session cleaned up successfully'})
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def debug_console_context(self, job_id):
        context = {
            'cwd': str(VIDEO_LINK_REPO_ROOT),
            'job_id': job_id,
            'status': None,
            'failed_stage': None,
            'error': None,
            'log_path': None,
            'log_tail': None,
        }
        if not job_id:
            return context
        try:
            job = self.video_link.load_job(job_id)
        except BridgeError as exc:
            context['error'] = exc.message
            return context

        run_dir = Path(job.get('run_dir') or VIDEO_LINK_REPO_ROOT).resolve()
        try:
            run_dir.relative_to(VIDEO_LINK_REPO_ROOT.resolve())
        except ValueError:
            pass
        else:
            if run_dir.is_dir():
                context['cwd'] = str(run_dir)
        runner = job.get('runner') or {}
        stages = job.get('stages') or {}
        failed_stage = next(
            (
                stage
                for stage in STAGE_ORDER
                if (stages.get(stage) or {}).get('status') == 'failed'
            ),
            None,
        )
        stage = failed_stage or runner.get('current_stage') or job.get('current_stage')
        if not stage:
            stage = next(
                (
                    candidate
                    for candidate in reversed(STAGE_ORDER)
                    if (stages.get(candidate) or {}).get('status') in {'succeeded', 'skipped'}
                ),
                None,
            )
        stage_data = stages.get(stage) or {}
        context.update(
            {
                'status': job.get('status') or runner.get('status'),
                'failed_stage': failed_stage,
                'error': runner.get('error') or stage_data.get('error'),
                'title': job.get('title'),
                'video_url': job.get('video_url'),
            }
        )
        if stage:
            try:
                log = self.video_link.stage_log(job_id, stage, 160, False)
                context['log_path'] = log.get('log_path')
                context['log_tail'] = '\n'.join(log.get('lines') or [])[-16000:]
            except BridgeError:
                pass
        return context

    def run(self):
        self.app.run(
            host=self.host,
            port=self.port,
            debug=self.dev_mode
        )

def main():
    parser = argparse.ArgumentParser(description="Video Analyzer UI Server")
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--dev', action='store_true', help='Enable development mode')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--jobs-dir', default=str(DEFAULT_JOBS_DIR), help='Durable video-link job directory')
    
    args = parser.parse_args()
    
    # Configure logging
    log_config = {
        'level': logging.DEBUG if args.dev else logging.INFO,
        'format': '%(asctime)s - %(levelname)s - %(message)s',
    }
    if args.log_file:
        log_config['filename'] = args.log_file
    logging.basicConfig(**log_config)
    
    # Start server
    server = VideoAnalyzerUI(
        host=args.host,
        port=args.port,
        dev_mode=args.dev,
        jobs_dir=args.jobs_dir,
    )
    server.run()

if __name__ == '__main__':
    main()
