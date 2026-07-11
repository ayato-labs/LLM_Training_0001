import os
import sys
from pathlib import Path

from src.utils.drive_uploader import (
    get_drive_service,
    get_or_create_drive_folder,
    upload_file_to_drive,
)


def run_api_test():
    print("=====================================================================")
    print("Starting Google Drive API Unit Test")
    print("=====================================================================")

    # 1. サービス認証テスト
    try:
        print("Step 1: Authenticating with Google Drive API...")
        service = get_drive_service()
        print("Step 1 Success: Authentication completed successfully!")
    except Exception as e:
        print(f"Step 1 Failed: Authentication failed. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. テスト用フォルダの作成/取得テスト
    test_folder_name = "Novel_LLM_Checkpoints_TEST"
    try:
        print(f"\nStep 2: Checking/Creating test folder '{test_folder_name}'...")
        folder_id = get_or_create_drive_folder(service, test_folder_name)
        print(f"Step 2 Success: Folder ID is {folder_id}")
    except Exception as e:
        print(f"Step 2 Failed: Folder creation failed. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. テストファイルのアップロードテスト
    test_file_path = Path("drive_test_file.txt")
    test_file_path.write_text(
        "This is a temporary test file for Google Drive API verification.", encoding="utf-8"
    )

    try:
        print(f"\nStep 3: Uploading test file '{test_file_path.name}'...")
        file_id = upload_file_to_drive(service, test_file_path, folder_id)
        print(f"Step 3 Success: File uploaded successfully! Google Drive File ID: {file_id}")
    except Exception as e:
        print(f"Step 3 Failed: Upload failed. Error: {e}", file=sys.stderr)
    finally:
        # テスト用ローカルファイルのクリーンアップ
        if test_file_path.exists():
            os.remove(test_file_path)
            print("\nCleaned up local temporary test file.")

    print("\n=====================================================================")
    print("Google Drive API Unit Test Finished.")
    print("=====================================================================")


if __name__ == "__main__":
    run_api_test()
