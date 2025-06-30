# Strategi Penyimpanan Dataset Persistent dengan Google Drive

## Latar Belakang Masalah
- Pengguna sering berpindah komputer (kampus, rumah, komputer berbeda)
- Dataset dan checkpoint perlu tetap persistent dan dapat diakses dari mana saja
- Solusi lokal tidak memadai untuk skenario multi-device

## Solusi yang Diusulkan: Google Drive Integration

### 1. Arsitektur Sistem
```
Local Machine
├── src/utils/
│   ├── google_drive_client.py     # Google Drive API client
│   ├── cloud_checkpoint_manager.py # Cloud-based checkpoint manager
│   └── hybrid_storage_manager.py   # Hybrid local/cloud storage
├── checkpoints/                    # Local cache (optional)
└── datasets/                       # Local cache (optional)

Google Drive
├── ujaran-kebencian-datasets/
│   ├── raw-dataset.csv
│   ├── processed-results/
│   │   ├── batch-1-results.csv
│   │   ├── batch-2-results.csv
│   │   └── final-results.csv
│   └── checkpoints/
│       ├── labeling_checkpoint_001.json
│       ├── labeling_checkpoint_002.json
│       └── current_checkpoint.json
```

### 2. Strategi Autentikasi Google Drive

#### Option A: Service Account (Recommended untuk Production)
**Kelebihan:**
- Tidak perlu interaksi user manual
- Cocok untuk automation
- Credential dapat disimpan secara aman

**Kekurangan:**
- Setup lebih kompleks
- Perlu Google Cloud Project

**Implementasi:**
```python
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Menggunakan service account key file
credentials = service_account.Credentials.from_service_account_file(
    'path/to/service-account-key.json',
    scopes=['https://www.googleapis.com/auth/drive']
)
service = build('drive', 'v3', credentials=credentials)
```

#### Option B: OAuth 2.0 User Consent (Recommended untuk Development)
**Kelebihan:**
- Setup lebih mudah
- Menggunakan akun Google personal
- Tidak perlu Google Cloud Project berbayar

**Kekurangan:**
- Perlu consent manual pertama kali
- Token perlu refresh berkala

**Implementasi:**
```python
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# OAuth flow untuk mendapatkan credentials
flow = Flow.from_client_secrets_file(
    'credentials.json',
    scopes=['https://www.googleapis.com/auth/drive']
)
```

### 3. Fitur yang Akan Diimplementasi

#### 3.1 CloudCheckpointManager
- Upload/download checkpoint ke/dari Google Drive
- Sinkronisasi otomatis setiap N processed samples
- Conflict resolution jika ada multiple devices
- Backup checkpoint lokal sebagai fallback

#### 3.2 CloudDatasetManager
- Upload dataset ke Google Drive
- Download dataset jika tidak ada lokal
- Incremental sync untuk hasil labeling
- Compression untuk menghemat storage

#### 3.3 HybridStorageManager
- Kombinasi local cache + cloud storage
- Smart caching strategy
- Offline mode dengan sync ketika online
- Bandwidth optimization

### 4. Workflow yang Diusulkan

```
1. Startup:
   ├── Check Google Drive connection
   ├── Download latest checkpoint (if exists)
   ├── Download dataset (if not cached locally)
   └── Initialize labeling pipeline

2. During Processing:
   ├── Process samples locally
   ├── Save checkpoint locally every N samples
   ├── Upload checkpoint to Google Drive every M samples
   └── Upload intermediate results periodically

3. Shutdown/Interruption:
   ├── Save current state locally
   ├── Upload final checkpoint to Google Drive
   └── Upload any pending results

4. Resume (on different machine):
   ├── Authenticate with Google Drive
   ├── Download latest checkpoint
   ├── Download dataset if needed
   └── Resume from last saved state
```

### 5. Keamanan dan Best Practices

#### 5.1 Credential Management
- Gunakan environment variables untuk sensitive data
- Jangan commit credentials ke repository
- Implementasi credential rotation
- Gunakan encrypted storage untuk tokens

#### 5.2 Data Privacy
- Encrypt sensitive data sebelum upload
- Implementasi access control
- Regular audit access logs
- Compliance dengan data protection regulations

#### 5.3 Error Handling
- Retry mechanism untuk network failures
- Graceful degradation ke local storage
- Comprehensive logging
- Data integrity checks

### 6. Test Case yang Akan Dibuat

#### 6.1 Test Case 1: Basic Authentication
```python
def test_google_drive_authentication():
    """Test basic Google Drive authentication"""
    # Test OAuth flow
    # Test service account authentication
    # Test token refresh
    pass
```

#### 6.2 Test Case 2: File Operations
```python
def test_file_upload_download():
    """Test basic file operations"""
    # Upload small test file
    # Download and verify integrity
    # Test file overwrite
    # Test file deletion
    pass
```

#### 6.3 Test Case 3: Checkpoint Sync
```python
def test_checkpoint_synchronization():
    """Test checkpoint sync between devices"""
    # Simulate processing on device A
    # Upload checkpoint
    # Simulate resume on device B
    # Verify state consistency
    pass
```

#### 6.4 Test Case 4: Conflict Resolution
```python
def test_conflict_resolution():
    """Test handling of concurrent modifications"""
    # Simulate processing on multiple devices
    # Create conflicting checkpoints
    # Test merge strategies
    pass
```

### 7. Implementation Plan

#### Phase 1: Basic Google Drive Integration (Week 1)
- Setup Google Drive API credentials
- Implement basic file upload/download
- Create simple test cases

#### Phase 2: Checkpoint Management (Week 2)
- Implement CloudCheckpointManager
- Integrate with existing PersistentLabelingPipeline
- Test checkpoint sync functionality

#### Phase 3: Dataset Management (Week 3)
- Implement CloudDatasetManager
- Add compression and optimization
- Test with actual dataset

#### Phase 4: Production Features (Week 4)
- Add error handling and retry logic
- Implement security best practices
- Performance optimization
- Comprehensive testing

### 8. Estimasi Biaya dan Resource

#### Google Drive Storage
- Free tier: 15GB (cukup untuk dataset 41,759 samples)
- Paid tier: $1.99/month untuk 100GB (jika perlu lebih)

#### Google Cloud API (jika menggunakan service account)
- Drive API: Free untuk usage normal
- Possible charges untuk high-volume operations

#### Bandwidth Considerations
- Dataset size: ~50MB (estimated)
- Checkpoint size: ~1KB per checkpoint
- Daily sync: <100MB (estimated)

### 9. Alternatif Solusi

#### Option 1: GitHub LFS (Large File Storage)
- Pro: Terintegrasi dengan Git workflow
- Con: Limited free storage (1GB)

#### Option 2: Dropbox API
- Pro: Simple API, good sync
- Con: Limited free storage (2GB)

#### Option 3: OneDrive API
- Pro: 5GB free storage
- Con: Microsoft ecosystem dependency

#### Option 4: Self-hosted Cloud Storage
- Pro: Full control, unlimited storage
- Con: Maintenance overhead, infrastructure cost

### 10. Kesimpulan dan Rekomendasi

**Rekomendasi:** Implementasi Google Drive dengan OAuth 2.0 untuk development phase, kemudian migrate ke Service Account untuk production.

**Alasan:**
1. Google Drive API mature dan well-documented
2. 15GB free storage cukup untuk project ini
3. Good integration dengan Python ecosystem
4. Reliable sync dan conflict resolution
5. Dapat di-scale untuk production use

**Next Steps:**
1. Setup Google Cloud Project dan enable Drive API
2. Implement basic authentication test
3. Create minimal viable prototype
4. Test dengan small dataset sample
5. Iterate berdasarkan hasil testing