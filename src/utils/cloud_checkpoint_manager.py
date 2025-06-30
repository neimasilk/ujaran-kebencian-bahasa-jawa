#!/usr/bin/env python3
"""
Cloud Checkpoint Manager untuk Google Drive Integration
Mengelola checkpoint dan dataset persistence menggunakan Google Drive API

Author: AI Assistant
Date: 2025-01-27
"""

import os
import json
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    from googleapiclient.errors import HttpError
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

class CloudCheckpointManager:
    """
    Manager untuk checkpoint persistence menggunakan Google Drive
    """
    
    def __init__(self, 
                 credentials_file: str = 'credentials.json',
                 token_file: str = 'token.json',
                 project_folder: str = 'ujaran-kebencian-datasets',
                 local_cache_dir: str = 'src/checkpoints'):
        """
        Initialize CloudCheckpointManager
        
        Args:
            credentials_file: Path ke Google OAuth credentials file
            token_file: Path ke token file untuk stored credentials
            project_folder: Nama folder di Google Drive untuk project ini
            local_cache_dir: Directory untuk local checkpoint cache
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.project_folder = project_folder
        self.local_cache_dir = Path(local_cache_dir)
        
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.service = None
        self.project_folder_id = None
        self.checkpoint_folder_id = None
        self.datasets_folder_id = None
        self.results_folder_id = None
        
        # Ensure local cache directory exists
        self.local_cache_dir.mkdir(exist_ok=True)
        
        self._authenticated = False
        self._offline_mode = False
    
    def authenticate(self) -> bool:
        """
        Authenticate dengan Google Drive API
        
        Returns:
            bool: True jika authentication berhasil
        """
        if not GOOGLE_DRIVE_AVAILABLE:
            print("Warning: Google Drive dependencies not available. Running in offline mode.")
            self._offline_mode = True
            return False
        
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
            
            # Refresh atau create new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_file):
                        print(f"Warning: Credentials file '{self.credentials_file}' not found.")
                        print("Running in offline mode. Use --setup for instructions.")
                        self._offline_mode = True
                        return False
                        
            # Create flow for new credentials if needed
            if not creds or not creds.valid:
                if not os.path.exists(self.credentials_file):
                    print(f"Warning: Credentials file '{self.credentials_file}' not found.")
                    print("Running in offline mode. Use --setup for instructions.")
                    self._offline_mode = True
                    return False
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.scopes
                )
                creds = flow.run_local_server(port=0)
                
                # Save credentials
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
            
            # Build service
            self.service = build('drive', 'v3', credentials=creds)
            self._authenticated = True
            
            # Setup project folders
            self._setup_project_folders()
            
            return True
            
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            self._offline_mode = True
            return False
    
    def display_resume_info(self, checkpoint_data: Dict[str, Any]):
        """
        Display clear resume information
        
        Args:
            checkpoint_data: Checkpoint data to display info for
        """
        try:
            processed_count = len(checkpoint_data.get('processed_indices', []))
            total_samples = checkpoint_data.get('metadata', {}).get('total_samples', 'Unknown')
            last_batch = checkpoint_data.get('metadata', {}).get('last_batch', 'Unknown')
            timestamp = checkpoint_data.get('timestamp', 'Unknown')
            checkpoint_id = checkpoint_data.get('checkpoint_id', 'Unknown')
            
            print("\n" + "="*60)
            print("🔄 RESUMING FROM CHECKPOINT")
            print("="*60)
            print(f"📋 Checkpoint ID: {checkpoint_id}")
            print(f"📊 Progress: {processed_count}/{total_samples} samples processed")
            print(f"📦 Last batch: {last_batch}")
            print(f"⏰ Last saved: {timestamp}")
            print(f"🎯 Continuing from sample #{processed_count + 1}")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"⚠️ Could not display resume info: {e}")
            print("🔄 Resuming from available checkpoint data...\n")
    
    def validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Validate checkpoint data integrity
        
        Args:
            checkpoint_data: Checkpoint data to validate
            
        Returns:
            bool: True if checkpoint is valid
        """
        try:
            required_fields = ['checkpoint_id', 'processed_indices', 'timestamp']
            
            for field in required_fields:
                if field not in checkpoint_data:
                    print(f"❌ Invalid checkpoint: missing {field}")
                    return False
            
            # Validate data consistency
            processed_indices = checkpoint_data['processed_indices']
            if not isinstance(processed_indices, list):
                print("❌ Invalid checkpoint: processed_indices must be list")
                return False
            
            # Check if indices are valid
            if processed_indices and not all(isinstance(i, int) and i >= 0 for i in processed_indices):
                print("❌ Invalid checkpoint: processed_indices contains invalid values")
                return False
            
            print(f"✅ Checkpoint validation passed: {len(processed_indices)} samples")
            return True
            
        except Exception as e:
            print(f"❌ Checkpoint validation failed: {e}")
            return False
            
            # Setup project folders
            self._setup_project_folders()
            return True
            
        except Exception as e:
            print(f"Warning: Google Drive authentication failed: {e}")
            print("Running in offline mode.")
            self._offline_mode = True
            return False
    
    def _setup_project_folders(self):
        """
        Setup project folder structure di Google Drive dengan recovery mechanism
        """
        if not self._authenticated:
            return
        
        try:
            print(f"🔧 Setting up Google Drive folder structure...")
            
            # Create atau find main project folder
            self.project_folder_id = self._get_or_create_folder(self.project_folder)
            print(f"📁 Main folder: {self.project_folder}")
            
            # Create atau find checkpoints subfolder
            self.checkpoint_folder_id = self._get_or_create_folder(
                'checkpoints', 
                parent_id=self.project_folder_id
            )
            print(f"💾 Checkpoints folder created/found")
            
            # Create atau find datasets subfolder
            self.datasets_folder_id = self._get_or_create_folder(
                'datasets',
                parent_id=self.project_folder_id
            )
            print(f"📊 Datasets folder created/found")
            
            # Create atau find results subfolder
            self.results_folder_id = self._get_or_create_folder(
                'results',
                parent_id=self.project_folder_id
            )
            print(f"📈 Results folder created/found")
            
            print(f"✅ Project folder structure setup complete: {self.project_folder}")
            
        except Exception as e:
            print(f"❌ Failed to setup project folders: {e}")
            print(f"🔄 This will be retried on next operation")
            self._offline_mode = True
    
    def _get_or_create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> str:
        """
        Get existing folder atau create new one
        
        Args:
            folder_name: Nama folder
            parent_id: Parent folder ID (None untuk root)
            
        Returns:
            str: Folder ID
        """
        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        results = self.service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()
        
        files = results.get('files', [])
        
        if files:
            return files[0]['id']
        else:
            # Create new folder
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                folder_metadata['parents'] = [parent_id]
            
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            return folder.get('id')
    
    def verify_and_recover_folders(self) -> bool:
        """
        Verify folder structure exists and recover if missing
        
        Returns:
            bool: True if folders are ready
        """
        if not self._authenticated:
            print("⚠️ Cannot verify folders: not authenticated")
            return False
        
        try:
            print("🔍 Verifying Google Drive folder structure...")
            
            # Check if main project folder exists
            if not self.project_folder_id:
                print("🔄 Main project folder missing, recreating...")
                self._setup_project_folders()
                return True
            
            # Verify each subfolder exists
            folders_to_check = [
                ('checkpoints', 'checkpoint_folder_id'),
                ('datasets', 'datasets_folder_id'),
                ('results', 'results_folder_id')
            ]
            
            recovery_needed = False
            for folder_name, attr_name in folders_to_check:
                folder_id = getattr(self, attr_name, None)
                if not folder_id:
                    print(f"🔄 {folder_name} folder missing, will recreate...")
                    recovery_needed = True
                    break
                
                # Verify folder still exists in Google Drive
                try:
                    self.service.files().get(fileId=folder_id).execute()
                except:
                    print(f"🔄 {folder_name} folder deleted from Google Drive, will recreate...")
                    recovery_needed = True
                    break
            
            if recovery_needed:
                print("🛠️ Recovering folder structure...")
                self._setup_project_folders()
            else:
                print("✅ All folders verified and ready")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to verify/recover folders: {e}")
            return False
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_id: str) -> bool:
        """
        Save checkpoint ke Google Drive dan local cache
        
        Args:
            checkpoint_data: Data checkpoint untuk disimpan
            checkpoint_id: Unique identifier untuk checkpoint
            
        Returns:
            bool: True jika berhasil disimpan
        """
        # Always save to local cache first
        local_success = self._save_checkpoint_local(checkpoint_data, checkpoint_id)
        
        if self._offline_mode or not self._authenticated:
            print(f"💾 Checkpoint saved locally: {checkpoint_id}")
            return local_success
        
        # Try to save to Google Drive
        try:
            cloud_success = self._save_checkpoint_cloud(checkpoint_data, checkpoint_id)
            if cloud_success:
                print(f"☁️ Checkpoint saved to cloud: {checkpoint_id}")
                return True
            else:
                print(f"⚠️ Cloud save failed, checkpoint saved locally: {checkpoint_id}")
                return local_success
                
        except Exception as e:
            print(f"⚠️ Cloud save error: {e}. Checkpoint saved locally: {checkpoint_id}")
            return local_success
    
    def _save_checkpoint_local(self, checkpoint_data: Dict[str, Any], checkpoint_id: str) -> bool:
        """
        Save checkpoint ke local cache
        """
        try:
            checkpoint_file = self.local_cache_dir / f"{checkpoint_id}.json"
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint locally: {e}")
            return False
    
    def _save_checkpoint_cloud(self, checkpoint_data: Dict[str, Any], checkpoint_id: str) -> bool:
        """
        Save checkpoint ke Google Drive
        """
        if not self._authenticated or not self.checkpoint_folder_id:
            return False
        
        temp_file_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
                json.dump(checkpoint_data, temp_file, indent=2, ensure_ascii=False)
                temp_file_path = temp_file.name
            
            filename = f"{checkpoint_id}.json"
            
            # Check if file already exists
            existing_file_id = self._find_file_in_folder(filename, self.checkpoint_folder_id)
            
            file_metadata = {
                'name': filename,
                'parents': [self.checkpoint_folder_id]
            }
            
            media = MediaFileUpload(temp_file_path, mimetype='application/json')
            
            if existing_file_id:
                # Update existing file
                self.service.files().update(
                    fileId=existing_file_id,
                    body={'name': filename},
                    media_body=media
                ).execute()
            else:
                # Create new file
                self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
            
            return True
                
        except Exception as e:
            print(f"❌ Failed to save checkpoint to cloud: {e}")
            return False
        finally:
            # Cleanup temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass  # Ignore cleanup errors
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint dari Google Drive atau local cache
        
        Args:
            checkpoint_id: Checkpoint ID untuk di-load
            
        Returns:
            Dict dengan checkpoint data atau None jika tidak ditemukan
        """
        # Try cloud first jika available
        if not self._offline_mode and self._authenticated:
            cloud_data = self._load_checkpoint_cloud(checkpoint_id)
            if cloud_data:
                # Save to local cache untuk backup
                self._save_checkpoint_local(cloud_data, checkpoint_id)
                print(f"☁️ Checkpoint loaded from cloud: {checkpoint_id}")
                return cloud_data
        
        # Fallback ke local cache
        local_data = self._load_checkpoint_local(checkpoint_id)
        if local_data:
            print(f"💾 Checkpoint loaded from local cache: {checkpoint_id}")
            return local_data
        
        print(f"❌ Checkpoint not found: {checkpoint_id}")
        return None
    
    def _load_checkpoint_local(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint dari local cache
        """
        try:
            checkpoint_file = self.local_cache_dir / f"{checkpoint_id}.json"
            
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"❌ Failed to load checkpoint from local cache: {e}")
            return None
    
    def _load_checkpoint_cloud(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint dari Google Drive
        """
        if not self._authenticated or not self.checkpoint_folder_id:
            return None
        
        try:
            filename = f"{checkpoint_id}.json"
            file_id = self._find_file_in_folder(filename, self.checkpoint_folder_id)
            
            if not file_id:
                return None
            
            # Download file
            request = self.service.files().get_media(fileId=file_id)
            
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
                downloader = MediaIoBaseDownload(temp_file, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                
                temp_file_path = temp_file.name
            
            try:
                # Read downloaded file
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
            finally:
                # Cleanup temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            print(f"❌ Failed to load checkpoint from cloud: {e}")
            return None
    
    def _find_file_in_folder(self, filename: str, folder_id: str) -> Optional[str]:
        """
        Find file dalam specific folder
        
        Returns:
            File ID jika ditemukan, None jika tidak
        """
        try:
            results = self.service.files().list(
                q=f"name='{filename}' and '{folder_id}' in parents",
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            return files[0]['id'] if files else None
            
        except Exception:
            return None
    
    def list_checkpoints(self) -> List[Dict[str, str]]:
        """
        List semua available checkpoints
        
        Returns:
            List of checkpoint info (id, source, timestamp)
        """
        checkpoints = []
        
        # List local checkpoints
        if self.local_cache_dir.exists():
            for checkpoint_file in self.local_cache_dir.glob("*.json"):
                checkpoint_id = checkpoint_file.stem
                stat = checkpoint_file.stat()
                checkpoints.append({
                    'id': checkpoint_id,
                    'source': 'local',
                    'timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'size': stat.st_size
                })
        
        # List cloud checkpoints jika available
        if not self._offline_mode and self._authenticated and self.checkpoint_folder_id:
            try:
                results = self.service.files().list(
                    q=f"'{self.checkpoint_folder_id}' in parents",
                    fields="files(id, name, size, modifiedTime)"
                ).execute()
                
                for file in results.get('files', []):
                    if file['name'].endswith('.json'):
                        checkpoint_id = file['name'][:-5]  # Remove .json extension
                        
                        # Check if already in local list
                        existing = next((c for c in checkpoints if c['id'] == checkpoint_id), None)
                        if existing:
                            existing['source'] = 'both'
                        else:
                            checkpoints.append({
                                'id': checkpoint_id,
                                'source': 'cloud',
                                'timestamp': file.get('modifiedTime', ''),
                                'size': int(file.get('size', 0))
                            })
                            
            except Exception as e:
                print(f"Warning: Failed to list cloud checkpoints: {e}")
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def sync_checkpoints(self) -> bool:
        """
        Sync checkpoints antara local dan cloud
        
        Returns:
            bool: True jika sync berhasil
        """
        if self._offline_mode or not self._authenticated:
            print("⚠️ Cannot sync: offline mode or not authenticated")
            return False
        
        try:
            print("🔄 Syncing checkpoints...")
            
            # Get all checkpoints
            checkpoints = self.list_checkpoints()
            
            sync_count = 0
            for checkpoint in checkpoints:
                checkpoint_id = checkpoint['id']
                
                if checkpoint['source'] == 'local':
                    # Upload local checkpoint to cloud
                    local_data = self._load_checkpoint_local(checkpoint_id)
                    if local_data and self._save_checkpoint_cloud(local_data, checkpoint_id):
                        sync_count += 1
                        print(f"  ⬆️ Uploaded: {checkpoint_id}")
                        
                elif checkpoint['source'] == 'cloud':
                    # Download cloud checkpoint to local
                    cloud_data = self._load_checkpoint_cloud(checkpoint_id)
                    if cloud_data and self._save_checkpoint_local(cloud_data, checkpoint_id):
                        sync_count += 1
                        print(f"  ⬇️ Downloaded: {checkpoint_id}")
            
            print(f"✅ Sync complete: {sync_count} checkpoints synced")
            return True
            
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            return False
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint terbaru berdasarkan timestamp
        
        Returns:
            Latest checkpoint data atau None
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        latest = checkpoints[0]  # Already sorted by timestamp desc
        return self.load_checkpoint(latest['id'])
    
    def cleanup_old_checkpoints(self, keep_count: int = 10) -> int:
        """
        Cleanup old checkpoints, keep only the latest N
        
        Args:
            keep_count: Number of checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_count:
            return 0
        
        to_delete = checkpoints[keep_count:]
        deleted_count = 0
        
        for checkpoint in to_delete:
            checkpoint_id = checkpoint['id']
            
            # Delete local
            local_file = self.local_cache_dir / f"{checkpoint_id}.json"
            if local_file.exists():
                local_file.unlink()
                deleted_count += 1
            
            # Delete cloud jika available
            if not self._offline_mode and self._authenticated and self.checkpoint_folder_id:
                try:
                    filename = f"{checkpoint_id}.json"
                    file_id = self._find_file_in_folder(filename, self.checkpoint_folder_id)
                    if file_id:
                        self.service.files().delete(fileId=file_id).execute()
                except Exception as e:
                    print(f"Warning: Failed to delete cloud checkpoint {checkpoint_id}: {e}")
        
        print(f"🗑️ Cleaned up {deleted_count} old checkpoints")
        return deleted_count
    
    def upload_dataset(self, local_file_path: str, cloud_filename: str) -> bool:
        """
        Upload dataset file (CSV) ke Google Drive
        
        Args:
            local_file_path: Path ke file lokal
            cloud_filename: Nama file di Google Drive
            
        Returns:
            bool: True jika upload berhasil
        """
        if self._offline_mode or not self._authenticated:
            print(f"⚠️ Cannot upload dataset: offline mode or not authenticated")
            return False
        
        if not os.path.exists(local_file_path):
            print(f"❌ Local file not found: {local_file_path}")
            return False
        
        try:
            # Create atau find datasets subfolder
            datasets_folder_id = self._get_or_create_folder(
                'datasets', 
                parent_id=self.project_folder_id
            )
            
            # Check if file already exists
            existing_file_id = self._find_file_in_folder(cloud_filename, datasets_folder_id)
            
            file_metadata = {
                'name': cloud_filename,
                'parents': [datasets_folder_id]
            }
            
            # Determine MIME type based on file extension
            if cloud_filename.endswith('.csv'):
                mimetype = 'text/csv'
            elif cloud_filename.endswith('.json'):
                mimetype = 'application/json'
            else:
                mimetype = 'application/octet-stream'
            
            media = MediaFileUpload(local_file_path, mimetype=mimetype)
            
            if existing_file_id:
                # Update existing file
                self.service.files().update(
                    fileId=existing_file_id,
                    body={'name': cloud_filename},
                    media_body=media
                ).execute()
                print(f"📤 Dataset updated in Google Drive: {cloud_filename}")
            else:
                # Create new file
                self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"📤 Dataset uploaded to Google Drive: {cloud_filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload dataset: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status informasi dari checkpoint manager
        
        Returns:
            Dict dengan status information
        """
        checkpoints = self.list_checkpoints()
        
        local_count = sum(1 for c in checkpoints if c['source'] in ['local', 'both'])
        cloud_count = sum(1 for c in checkpoints if c['source'] in ['cloud', 'both'])
        
        return {
            'authenticated': self._authenticated,
            'offline_mode': self._offline_mode,
            'total_checkpoints': len(checkpoints),
            'local_checkpoints': local_count,
            'cloud_checkpoints': cloud_count,
            'project_folder': self.project_folder,
            'local_cache_dir': str(self.local_cache_dir),
            'latest_checkpoint': checkpoints[0]['id'] if checkpoints else None
        }