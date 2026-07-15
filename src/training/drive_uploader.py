"""
Google Drive チェックポイント＆アーティファクトアーカイブデーモン
---------------------------------------------------
オリジナルアップローダを拡張：
- DVCキャッシュバックアップ（重複排除オブジェクトストレージ）
- 評価レポートバックアップ
- ADRドキュメントバックアップ
- ローカルストレージ自動クリーンアップポリシー

ADR-019: クラウドファースト戦略によるストレージ最適化。
"""

import contextlib
import datetime
import glob
import os
import re
import shutil
import sys
import time
from pathlib import Path
from src.common.logger import logger

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    HAS_GOOGLE_DRIVE = True
except ImportError:
    HAS_GOOGLE_DRIVE = False

# ============================================================
# 設定
# ============================================================
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

OUTPUT_DIR = Path("models/output")
DRIVE_FOLDER_NAME = "Novel_LLM_Checkpoints"
POLL_INTERVAL = 30  # 秒
MIN_FOLDER_AGE = 60  # チェックポイント処理までの最小経過秒数

# zipとしてバックアップするディレクトリ
LOG_DIRS = [
    ("mlruns", "mlruns_backup"),
    ("models/output/runs", "tensorboard_backup"),
]

# 追加バックアップ対象（ソース、ラベル）
ARTIFACT_DIRS = [
    ("docs/ADR", "adr_backup"),
    ("logs", "logs_backup"),
]

# DVCキャッシュ：.dvc/cacheが存在し内容がある場合のみバックアップ
DVC_CACHE_DIR = Path(".dvc/cache")

# 最終モデル出力ディレクトリ
FINAL_MODEL_DIR = Path("models/output")
FINAL_MODEL_DRIVE_FOLDER = "Novel_LLM_Models"

# ローカルクリーンアップ：最新のN個のチェックポイントのみを保持
LOCAL_CHECKPOINT_KEEP = 2


def get_drive_service():
    """client_secret_*.jsonを使用したOAuth 2.0認証。"""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            secret_files = glob.glob("client_secret_*.json") + glob.glob("credentials.json")
            if not secret_files:
                raise FileNotFoundError("No client_secret_*.json or credentials.json found.")
            secret_file = secret_files[0]
            print(f"Using client secret: {secret_file}")
            flow = InstalledAppFlow.from_client_secrets_file(secret_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_or_create_drive_folder(service, folder_name, parent_id=None):
    """Google Drive上にフォルダを取得または作成。"""
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id)").execute()
    items = results.get("files", [])
    if items:
        return items[0]["id"]

    folder_metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        folder_metadata["parents"] = [parent_id]
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    print(f"Created remote folder '{folder_name}' (ID: {folder['id']})")
    return folder["id"]


def upload_file_to_drive(service, file_path, folder_id):
    """Upload a file to Google Drive with progress reporting."""
    file_name = file_path.name
    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(str(file_path), chunksize=1024 * 1024 * 5, resumable=True)
    request = service.files().create(body=file_metadata, media_body=media, fields="id")

    response = None
    print(f"Uploading {file_name}...")
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  {int(status.progress() * 100)}%")
    print(f"  Uploaded (ID: {response['id']})")
    return response["id"]


def delete_remote_file(service, file_id):
    """Delete a file on Google Drive."""
    try:
        service.files().delete(fileId=file_id).execute()
    except Exception as e:
        print(f"  Warning: Failed to delete remote file {file_id}: {e}", file=sys.stderr)


def file_exists_on_drive(service, file_name, folder_id):
    """Check if a file already exists on Google Drive."""
    query = f"name = '{file_name}' and '{folder_id}' in parents and trashed = false"
    return bool(service.files().list(q=query, fields="files(id)").execute().get("files", []))


def compress_and_upload(service, source_dir, folder_id, label):
    """Compress a directory to zip, upload, then clean up."""
    source_path = Path(source_dir)
    if not source_path.exists():
        return
    if source_path.is_dir() and not any(source_path.iterdir()):
        return

    zip_base = OUTPUT_DIR / label
    zip_path = Path(f"{zip_base}.zip")

    try:
        print(f"Compressing {source_dir}...")
        shutil.make_archive(str(zip_base), "zip", str(source_path))

        remote_name = zip_path.name
        if file_exists_on_drive(service, remote_name, folder_id):
            # Delete old version first
            query = f"name = '{remote_name}' and '{folder_id}' in parents and trashed = false"
            old = service.files().list(q=query, fields="files(id)").execute().get("files", [])
            for f in old:
                delete_remote_file(service, f["id"])

        upload_file_to_drive(service, zip_path, folder_id)
    except Exception as e:
        print(f"Error backing up {source_dir}: {e}", file=sys.stderr)
    finally:
        if zip_path.exists():
            with contextlib.suppress(Exception):
                os.remove(zip_path)


def backup_dvc_cache(service, folder_id):
    """
    Backup DVC cache to Google Drive.
    DVC cache files are content-addressed (SHA256), so only changed objects are uploaded.
    Only backs up if cache exists and has files.
    """
    if not DVC_CACHE_DIR.exists():
        return

    # Count files in cache
    cache_files = list(DVC_CACHE_DIR.rglob("*"))
    cache_files = [f for f in cache_files if f.is_file()]
    if not cache_files:
        return

    # Calculate total size
    total_size = sum(f.stat().st_size for f in cache_files)
    total_mb = total_size / (1024 * 1024)

    # Skip if cache is too large (> 500MB) - user should configure dvc remote instead
    if total_mb > 500:
        print(
            f"DVC cache too large ({total_mb:.1f} MB). Skipping backup. Consider configuring 'dvc remote'."
        )
        return

    print(f"Backing up DVC cache ({len(cache_files)} files, {total_mb:.1f} MB)...")
    compress_and_upload(service, DVC_CACHE_DIR, folder_id, "dvc_cache_backup")


def get_checkpoints(output_dir=None):
    """List valid checkpoint directories sorted by step number."""
    target_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    if not target_dir.exists():
        return []

    checkpoints = []
    for item in target_dir.iterdir():
        if item.is_dir() and re.match(r"^checkpoint-\d+$", item.name):
            step = int(item.name.split("-")[1])
            checkpoints.append((step, item))
    return sorted(checkpoints, key=lambda x: x[0])


def cleanup_old_checkpoints(keep=LOCAL_CHECKPOINT_KEEP, output_dir=None):
    """
    Remove old checkpoint directories locally, keeping only the latest `keep` checkpoints.
    This is the primary mechanism for reducing local storage pressure.
    """
    checkpoints = get_checkpoints(output_dir=output_dir)
    if len(checkpoints) <= keep:
        return

    # Remove oldest checkpoints (keep the latest `keep`)
    to_remove = checkpoints[:-keep]
    for _step, path in to_remove:
        uploaded_flag = path / ".uploaded"
        if uploaded_flag.exists():
            logger.info(f"Cleaning up old checkpoint: {path.name}")
            shutil.rmtree(path)
        else:
            logger.debug(f"Skipping cleanup of {path.name} (not uploaded yet)")


def cleanup_old_logs(max_log_files=10):
    """Keep only the most recent log files, remove older ones."""
    log_dir = Path("logs")
    if not log_dir.exists():
        return

    log_files = sorted(log_dir.glob("train_*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    for old_log in log_files[max_log_files:]:
        print(f"Cleaning old log: {old_log.name}")
        old_log.unlink(missing_ok=True)


def backup_final_model(service, root_folder_id):
    """
    Backup the final trained model to a dedicated Google Drive folder.
    The model is saved as a zip archive under 'Novel_LLM_Models/'.
    Only uploads if the model has changed (based on .uploaded flag).
    """
    if not FINAL_MODEL_DIR.exists():
        return

    # Check if model files exist (model.safetensors or config.json)
    has_model = (
        any(FINAL_MODEL_DIR.glob("*.safetensors")) or (FINAL_MODEL_DIR / "config.json").exists()
    )
    if not has_model:
        return

    # Create dedicated model folder on Drive
    model_folder_id = get_or_create_drive_folder(service, FINAL_MODEL_DRIVE_FOLDER, root_folder_id)

    # Check for .uploaded flag in model dir
    uploaded_flag = FINAL_MODEL_DIR / ".model_uploaded"
    if uploaded_flag.exists():
        return

    # Check if model is still being written (look for recent modifications)
    latest_mtime = 0
    for f in FINAL_MODEL_DIR.iterdir():
        if f.is_file() and f.name != ".model_uploaded":
            mtime = os.path.getmtime(f)
            if mtime > latest_mtime:
                latest_mtime = mtime

    if (time.time() - latest_mtime) < MIN_FOLDER_AGE:
        return  # Still writing

    # Compress the model directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"novel_llm_model_{timestamp}"
    zip_base = FINAL_MODEL_DIR.parent / zip_name
    zip_path = Path(f"{zip_base}.zip")

    try:
        print("Backing up final model to Google Drive...")
        shutil.make_archive(str(zip_base), "zip", str(FINAL_MODEL_DIR))

        upload_file_to_drive(service, zip_path, model_folder_id)
        uploaded_flag.touch()
        print(f"Final model backup complete: {zip_path.name}")

    except Exception as e:
        print(f"Error backing up final model: {e}", file=sys.stderr)
    finally:
        if zip_path.exists():
            with contextlib.suppress(Exception):
                os.remove(zip_path)


def backup_inference_reports(service, root_folder_id):
    """Backup inference evaluation reports (eval_report_*.md, eval_results_*.json)."""
    log_dir = Path("logs")
    if not log_dir.exists():
        return

    report_files = list(log_dir.glob("eval_report_*.md")) + list(
        log_dir.glob("eval_results_*.json")
    )
    if not report_files:
        return

    # Create inference reports folder
    reports_folder_id = get_or_create_drive_folder(
        service, "Novel_LLM_Inference_Reports", root_folder_id
    )

    # Only upload files newer than the last backup
    backup_flag = log_dir / ".inference_backup_timestamp"
    last_backup = 0
    if backup_flag.exists():
        last_backup = backup_flag.stat().st_mtime

    new_reports = [f for f in report_files if f.stat().st_mtime > last_backup]
    if not new_reports:
        return

    print(f"Backing up {len(new_reports)} new inference report(s)...")
    for report in new_reports:
        try:
            if not file_exists_on_drive(service, report.name, reports_folder_id):
                upload_file_to_drive(service, report, reports_folder_id)
        except Exception as e:
            print(f"Error backing up {report.name}: {e}", file=sys.stderr)

    backup_flag.touch()


def monitor_and_upload():
    """Main daemon loop: monitor directories, backup, and cleanup."""
    if not HAS_GOOGLE_DRIVE:
        print("Error: Google Drive dependencies not installed. Exiting daemon.", file=sys.stderr)
        return
    print("=" * 60)
    print("Google Drive Archiver Daemon Started")
    print(f"  Monitoring: {OUTPUT_DIR.resolve()}")
    print(f"  Checkpoints kept locally: {LOCAL_CHECKPOINT_KEEP}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print("=" * 60)

    try:
        service = get_drive_service()
        root_folder_id = get_or_create_drive_folder(service, DRIVE_FOLDER_NAME)
    except Exception as e:
        print(f"Failed to initialize Google Drive API: {e}", file=sys.stderr)
        return

    loop_count = 0
    while True:
        try:
            # === Periodic log/artifact backup (every 2 loops = ~60s) ===
            if loop_count % 2 == 0:
                # MLflow + TensorBoard logs
                for log_dir, label in LOG_DIRS:
                    compress_and_upload(service, log_dir, root_folder_id, label)

                # ADR documents
                for artifact_dir, label in ARTIFACT_DIRS:
                    compress_and_upload(service, artifact_dir, root_folder_id, label)

                # DVC cache (if small enough)
                backup_dvc_cache(service, root_folder_id)

                # Final model backup
                backup_final_model(service, root_folder_id)

                # Inference evaluation reports
                backup_inference_reports(service, root_folder_id)

            loop_count += 1

            # === Checkpoint processing ===
            checkpoints = get_checkpoints()
            if not checkpoints:
                time.sleep(POLL_INTERVAL)
                continue

            latest_step, latest_path = checkpoints[-1]

            for step, path in checkpoints:
                uploaded_flag = path / ".uploaded"
                if uploaded_flag.exists():
                    continue

                state_file = path / "trainer_state.json"
                if not state_file.exists():
                    continue

                mtime = os.path.getmtime(path)
                if (time.time() - mtime) < MIN_FOLDER_AGE:
                    continue

                zip_file = OUTPUT_DIR / f"{path.name}.zip"

                # Compress
                if not zip_file.exists():
                    print(f"Compressing {path.name}...")
                    shutil.make_archive(str(OUTPUT_DIR / path.name), "zip", str(path))

                # Upload
                try:
                    if not file_exists_on_drive(service, zip_file.name, root_folder_id):
                        upload_file_to_drive(service, zip_file, root_folder_id)
                    else:
                        print(f"  '{zip_file.name}' already on Drive. Skipping.")

                    # Mark as uploaded
                    uploaded_flag.touch()

                    # Cleanup zip
                    if zip_file.exists():
                        os.remove(zip_file)

                    # Remove old local checkpoints (except latest)
                    if step < latest_step:
                        print(f"  Removing local: {path.name}")
                        shutil.rmtree(path)

                except Exception as e:
                    print(f"Error processing {path.name}: {e}", file=sys.stderr)
                    if zip_file.exists():
                        with contextlib.suppress(Exception):
                            os.remove(zip_file)

            # === Local storage cleanup (every 10 loops = ~5min) ===
            if loop_count % 10 == 0:
                cleanup_old_checkpoints(keep=LOCAL_CHECKPOINT_KEEP)
                cleanup_old_logs(max_log_files=10)

        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL)


try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:
        pass


class DriveUploadCallback(TrainerCallback):
    """Trainer Callback: 定期的にチェックポイントをGoogle Driveへアップロード"""

    def __init__(self, upload_interval_steps: int = 1000):
        self.upload_interval_steps = upload_interval_steps
        self._service = None
        self._folder_id = None

    def _get_service(self):
        if not HAS_GOOGLE_DRIVE:
            return None, None
        if self._service is None:
            import glob
            import os

            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            scopes = ["https://www.googleapis.com/auth/drive.file"]
            creds = None
            if os.path.exists("token.json"):
                creds = Credentials.from_authorized_user_file("token.json", scopes)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    secret_files = glob.glob("client_secret_*.json") + glob.glob("credentials.json")
                    if not secret_files:
                        return None, None
                    flow = InstalledAppFlow.from_client_secrets_file(secret_files[0], scopes)
                    creds = flow.run_local_server(port=0)
                with open("token.json", "w") as token:
                    token.write(creds.to_json())
            self._service = build("drive", "v3", credentials=creds)
            # Create/get folder
            folder_name = "Novel_LLM_Checkpoints"
            query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = self._service.files().list(q=query, fields="files(id)").execute()
            items = results.get("files", [])
            if items:
                self._folder_id = items[0]["id"]
            else:
                folder_metadata = {
                    "name": folder_name,
                    "mimeType": "application/vnd.google-apps.folder",
                }
                folder = self._service.files().create(body=folder_metadata, fields="id").execute()
                self._folder_id = folder["id"]
        return self._service, self._folder_id

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.upload_interval_steps == 0 and state.global_step > 0:
            self._upload_checkpoint(state.global_step, output_dir=args.output_dir)

    def on_train_end(self, args, state, control, **kwargs):
        self._upload_checkpoint(state.global_step, output_dir=args.output_dir, force=True)

    def force_final_upload(self, global_step: int, output_dir: str = None):
        self._upload_checkpoint(global_step, output_dir=output_dir, force=True)

    def _upload_checkpoint(self, step: int, output_dir: str = None, force: bool = False):
        try:
            service, folder_id = self._get_service()
            if service is None:
                logger.warning("Google Drive service not available (credentials or dependencies missing). Skipping backup.")
                return

            import os
            import shutil
            from pathlib import Path

            target_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
            checkpoint_dir = target_output_dir / f"checkpoint-{step}"
            if not checkpoint_dir.exists():
                return

            zip_path = target_output_dir / f"checkpoint-{step}.zip"
            if zip_path.exists() and not force:
                return

            logger.info(f"Compressing checkpoint-{step}...")
            shutil.make_archive(str(target_output_dir / f"checkpoint-{step}"), "zip", str(checkpoint_dir))

            # Upload
            from googleapiclient.http import MediaFileUpload

            file_metadata = {"name": f"checkpoint-{step}.zip", "parents": [folder_id]}
            media = MediaFileUpload(str(zip_path), chunksize=1024 * 1024 * 5, resumable=True)
            request = service.files().create(body=file_metadata, media_body=media, fields="id")

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Uploading checkpoint-{step}.zip: {int(status.progress() * 100)}%")

            logger.info(f"Uploaded checkpoint-{step}.zip (ID: {response['id']})")

            # Mark as uploaded
            (checkpoint_dir / ".uploaded").touch()

            # Cleanup zip
            if zip_path.exists():
                os.remove(zip_path)

            # Cleanup old checkpoints locally
            cleanup_old_checkpoints(keep=LOCAL_CHECKPOINT_KEEP, output_dir=str(target_output_dir))

        except Exception as e:
            logger.warning(f"Error in DriveUploadCallback: {e}")


if __name__ == "__main__":
    monitor_and_upload()
