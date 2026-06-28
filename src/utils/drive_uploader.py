import os
import sys
import time
import shutil
import re
import glob
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive API スコープ（読み書き権限）
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# 設定
OUTPUT_DIR = Path("models/output")
DRIVE_FOLDER_NAME = "Novel_LLM_Checkpoints"
POLL_INTERVAL = 30  # ポーリング間隔（秒）
MIN_FOLDER_AGE = 60  # ディレクトリ更新から処理開始までの最小待機時間（秒、書き込み途中の検知防止）

def get_drive_service():
    """OAuth 2.0 ユーザー認証情報を使用して認証（client_secret_*.json を自動探索）"""
    creds = None
    # 既存のトークンがあればロード
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # 有効な認証情報がない場合はユーザー認証を実行
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        
        if not creds:
            # カレントディレクトリから 'client_secret_*.json' または 'credentials.json' を自動探索
            secret_files = glob.glob("client_secret_*.json") + glob.glob("credentials.json")
            if not secret_files:
                print("Error: 'client_secret_*.json' または 'credentials.json' がカレントディレクトリに見つかりません。", file=sys.stderr)
                sys.exit(1)
            
            secret_file = secret_files[0]
            print(f"Using client secret file: {secret_file}")
            
            flow = InstalledAppFlow.from_client_secrets_file(secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # 認証情報を保存
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def get_or_create_drive_folder(service, folder_name):
    """Google Drive 上に指定フォルダが存在しなければ作成し、そのIDを返す"""
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, fields="files(id)").execute()
    items = results.get('files', [])
    
    if items:
        return items[0]['id']
    
    # 存在しない場合は新規作成
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    print(f"Created remote folder '{folder_name}' on Google Drive with ID: {folder['id']}")
    return folder['id']

def upload_file_to_drive(service, file_path, folder_id):
    """ファイルを Google Drive の指定フォルダにアップロードする"""
    file_name = file_path.name
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    
    # 進行状況付きでアップロード
    media = MediaFileUpload(str(file_path), chunksize=1024*1024*5, resumable=True)
    request = service.files().create(body=file_metadata, media_body=media, fields='id')
    
    response = None
    print(f"Uploading {file_name} to Google Drive...")
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")
            
    print(f"Upload complete. File ID on Google Drive: {response['id']}")
    return response['id']

def get_checkpoints():
    """models/output 内の有効な checkpoint-XXXX フォルダを列挙し、ステップ数の昇順でソートして返す"""
    if not OUTPUT_DIR.exists():
        return []
    
    checkpoints = []
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir() and re.match(r"^checkpoint-\d+$", item.name):
            step = int(item.name.split("-")[1])
            checkpoints.append((step, item))
            
    return sorted(checkpoints, key=lambda x: x[0])

def monitor_and_upload():
    """ディレクトリを監視し、非同期でチェックポイントのzip圧縮・アップロード・ローカル削除を実行"""
    print("=====================================================================")
    print("Google Drive Checkpoint Archiver Deamon Started")
    print(f"Monitoring Directory: {OUTPUT_DIR.resolve()}")
    print("=====================================================================")
    
    try:
        service = get_drive_service()
        folder_id = get_or_create_drive_folder(service, DRIVE_FOLDER_NAME)
    except Exception as e:
        print(f"Failed to initialize Google Drive API: {e}", file=sys.stderr)
        return

    while True:
        try:
            checkpoints = get_checkpoints()
            if not checkpoints:
                time.sleep(POLL_INTERVAL)
                continue
                
            # 最新のチェックポイント（最後の要素）は学習継続に必要であるため、ローカルから削除しない
            latest_step, latest_path = checkpoints[-1]
            
            for step, path in checkpoints:
                # trainer_state.json が存在し、フォルダの書き込み更新から一定時間経っているか確認
                state_file = path / "trainer_state.json"
                if not state_file.exists():
                    # まだ書き込み完了していない可能性があるためスキップ
                    continue
                
                # フォルダの最終更新時刻チェック（書き込み途中の安全対策）
                mtime = os.path.getmtime(path)
                age = time.time() - mtime
                if age < MIN_FOLDER_AGE:
                    # まだ更新中である可能性があるため待つ
                    continue
                
                zip_file = OUTPUT_DIR / f"{path.name}.zip"
                
                # 1. すでにzipが存在するか、または新規作成
                if not zip_file.exists():
                    print(f"Compressing {path.name} to zip archive...")
                    shutil.make_archive(str(OUTPUT_DIR / path.name), 'zip', str(path))
                    print(f"Zip created: {zip_file.name}")
                
                # 2. Google Drive へのアップロード
                try:
                    # すでにクラウド上に同名ファイルがあるか確認（二重アップロード防止）
                    query = f"name = '{zip_file.name}' and '{folder_id}' in parents and trashed = false"
                    exists = service.files().list(q=query, fields="files(id)").execute().get('files', [])
                    
                    if not exists:
                        upload_file_to_drive(service, zip_file, folder_id)
                    else:
                        print(f"File '{zip_file.name}' already exists on Google Drive. Skipping upload.")
                    
                    # 3. アップロード完了後のクリーンアップ
                    # zip ファイルの削除
                    if zip_file.exists():
                        os.remove(zip_file)
                        
                    # 最新ではないチェックポイントのみローカルフォルダを削除する
                    if step < latest_step:
                        print(f"Safely removing local directory: {path.name} (Latest checkpoint is {latest_path.name})")
                        shutil.rmtree(path)
                    else:
                        print(f"Keeping local directory: {path.name} (Required for current run context / resume)")
                        
                except Exception as e:
                    print(f"Error processing {path.name}: {e}", file=sys.stderr)
                    # エラー時は zip ファイルをクリーンアップして次回リトライ
                    if zip_file.exists():
                        try:
                            os.remove(zip_file)
                        except Exception:
                            pass
            
        except Exception as e:
            print(f"Unexpected error in monitor loop: {e}", file=sys.stderr)
            
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    monitor_and_upload()
