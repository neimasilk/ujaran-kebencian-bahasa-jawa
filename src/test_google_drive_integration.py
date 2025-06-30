#!/usr/bin/env python3
"""
Test Case untuk Google Drive Integration
Testing basic authentication dan file operations dengan Google Drive API

Author: AI Assistant
Date: 2025-01-27
"""

import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

# Note: Dependencies yang perlu diinstall:
# pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    print("Warning: Google Drive dependencies not installed.")
    print("Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

class GoogleDriveTestClient:
    """
    Test client untuk Google Drive operations
    """
    
    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.service = None
        self.test_folder_id = None
        
    def authenticate(self):
        """
        Authenticate dengan Google Drive menggunakan OAuth 2.0
        """
        if not GOOGLE_DRIVE_AVAILABLE:
            raise ImportError("Google Drive dependencies not available")
            
        creds = None
        
        # Load existing token jika ada
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
        
        # Jika tidak ada valid credentials, lakukan OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Credentials file '{self.credentials_file}' not found. "
                        "Please download it from Google Cloud Console."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.scopes
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials untuk next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive authentication successful")
        return True
    
    def create_test_folder(self, folder_name="ujaran-kebencian-test"):
        """
        Buat folder test di Google Drive
        """
        if not self.service:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        # Check jika folder sudah ada
        existing_folders = self.service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()
        
        if existing_folders['files']:
            self.test_folder_id = existing_folders['files'][0]['id']
            print(f"✅ Using existing test folder: {folder_name}")
        else:
            # Buat folder baru
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            self.test_folder_id = folder.get('id')
            print(f"✅ Created test folder: {folder_name}")
        
        return self.test_folder_id
    
    def upload_test_file(self, content, filename="test-checkpoint.json"):
        """
        Upload test file ke Google Drive
        """
        if not self.service or not self.test_folder_id:
            raise RuntimeError("Not authenticated or test folder not created.")
        
        # Buat temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(content, temp_file, indent=2)
            temp_file_path = temp_file.name
        
        try:
            file_metadata = {
                'name': filename,
                'parents': [self.test_folder_id]
            }
            
            media = MediaFileUpload(temp_file_path, mimetype='application/json')
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            print(f"✅ Uploaded test file: {filename} (ID: {file_id})")
            return file_id
            
        finally:
            # Cleanup temporary file
            os.unlink(temp_file_path)
    
    def download_test_file(self, file_id, local_filename="downloaded-test.json"):
        """
        Download file dari Google Drive
        """
        if not self.service:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        request = self.service.files().get_media(fileId=file_id)
        
        with open(local_filename, 'wb') as local_file:
            downloader = MediaIoBaseDownload(local_file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
        
        print(f"✅ Downloaded file: {local_filename}")
        return local_filename
    
    def list_files_in_folder(self):
        """
        List semua files dalam test folder
        """
        if not self.service or not self.test_folder_id:
            raise RuntimeError("Not authenticated or test folder not created.")
        
        results = self.service.files().list(
            q=f"'{self.test_folder_id}' in parents",
            fields="files(id, name, size, modifiedTime)"
        ).execute()
        
        files = results.get('files', [])
        print(f"\n📁 Files in test folder ({len(files)} files):")
        for file in files:
            size = int(file.get('size', 0)) if file.get('size') else 0
            modified = file.get('modifiedTime', 'Unknown')
            print(f"  - {file['name']} (ID: {file['id']}, Size: {size} bytes, Modified: {modified})")
        
        return files
    
    def delete_test_file(self, file_id):
        """
        Delete file dari Google Drive
        """
        if not self.service:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        self.service.files().delete(fileId=file_id).execute()
        print(f"✅ Deleted file with ID: {file_id}")
    
    def cleanup_test_folder(self):
        """
        Cleanup test folder dan semua contents
        """
        if not self.service or not self.test_folder_id:
            return
        
        # List dan delete semua files dalam folder
        files = self.list_files_in_folder()
        for file in files:
            self.delete_test_file(file['id'])
        
        # Delete folder itu sendiri
        self.service.files().delete(fileId=self.test_folder_id).execute()
        print(f"✅ Deleted test folder")

def run_basic_tests():
    """
    Run basic test cases untuk Google Drive integration
    """
    print("🚀 Starting Google Drive Integration Tests\n")
    
    if not GOOGLE_DRIVE_AVAILABLE:
        print("❌ Google Drive dependencies not available. Please install them first.")
        return False
    
    try:
        # Initialize client
        client = GoogleDriveTestClient()
        
        # Test 1: Authentication
        print("Test 1: Authentication")
        client.authenticate()
        
        # Test 2: Create test folder
        print("\nTest 2: Create test folder")
        client.create_test_folder()
        
        # Test 3: Upload test file
        print("\nTest 3: Upload test file")
        test_data = {
            "checkpoint_id": "test_001",
            "timestamp": datetime.now().isoformat(),
            "processed_indices": [1, 2, 3, 4, 5],
            "total_samples": 100,
            "current_batch": 1,
            "test_mode": True
        }
        
        file_id = client.upload_test_file(test_data, "test-checkpoint-001.json")
        
        # Test 4: List files
        print("\nTest 4: List files in folder")
        files = client.list_files_in_folder()
        
        # Test 5: Download file
        print("\nTest 5: Download file")
        downloaded_file = client.download_test_file(file_id, "downloaded-checkpoint.json")
        
        # Verify downloaded content
        with open(downloaded_file, 'r') as f:
            downloaded_data = json.load(f)
        
        if downloaded_data == test_data:
            print("✅ Downloaded data matches uploaded data")
        else:
            print("❌ Downloaded data does not match uploaded data")
        
        # Test 6: Upload multiple files
        print("\nTest 6: Upload multiple files")
        for i in range(2, 5):
            test_data_multi = {
                "checkpoint_id": f"test_{i:03d}",
                "timestamp": datetime.now().isoformat(),
                "processed_indices": list(range(1, i*10)),
                "total_samples": 100,
                "current_batch": i
            }
            client.upload_test_file(test_data_multi, f"test-checkpoint-{i:03d}.json")
        
        # List all files again
        print("\nFinal file list:")
        client.list_files_in_folder()
        
        # Cleanup
        print("\nTest 7: Cleanup")
        cleanup = input("Do you want to cleanup test files? (y/n): ").lower().strip()
        if cleanup == 'y':
            client.cleanup_test_folder()
        else:
            print("Test files preserved for manual inspection")
        
        # Cleanup local downloaded file
        if os.path.exists(downloaded_file):
            os.unlink(downloaded_file)
            print(f"✅ Cleaned up local file: {downloaded_file}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def setup_instructions():
    """
    Print setup instructions untuk Google Drive API
    """
    print("""
📋 Setup Instructions untuk Google Drive Integration:

1. Buat Google Cloud Project:
   - Go to https://console.cloud.google.com/
   - Create new project atau pilih existing project
   - Enable Google Drive API

2. Create OAuth 2.0 Credentials:
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "OAuth 2.0 Client ID"
   - Choose "Desktop Application"
   - Download credentials sebagai 'credentials.json'
   - Place file di root directory project ini

3. Install Dependencies:
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

4. Run Test:
   python test_google_drive_integration.py

5. First Run:
   - Browser akan terbuka untuk OAuth consent
   - Login dengan Google account
   - Grant permissions untuk Drive access
   - Token akan disimpan di 'token.json' untuk future use

Note: File 'credentials.json' dan 'token.json' jangan di-commit ke repository!
Tambahkan ke .gitignore untuk security.
    """)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_instructions()
    else:
        # Check jika credentials file ada
        if not os.path.exists('credentials.json'):
            print("❌ credentials.json not found!")
            print("\nPlease run: python test_google_drive_integration.py --setup")
            print("for setup instructions.\n")
            sys.exit(1)
        
        success = run_basic_tests()
        sys.exit(0 if success else 1)