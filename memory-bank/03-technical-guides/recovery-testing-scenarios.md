# Recovery Testing Scenarios - Google Drive Labeling System

## Overview
Dokumentasi skenario testing untuk memastikan sistem labeling dapat recovery dengan baik dari berbagai jenis interupsi dan dapat melanjutkan proses dari checkpoint terakhir.

## Test Scenarios

### Scenario 1: Graceful Interruption (Ctrl+C)
**Objective**: Memastikan sistem dapat di-interrupt dengan aman dan menyimpan progress ke Google Drive

**Steps**:
1. Jalankan `python labeling.py` atau `python src/google_drive_labeling.py`
2. Biarkan proses beberapa batch (minimal 2-3 batch)
3. Tekan `Ctrl+C` untuk interrupt
4. Verifikasi:
   - Progress tersimpan ke checkpoint lokal
   - Checkpoint ter-sync ke Google Drive
   - Pesan konfirmasi sync ditampilkan

**Expected Results**:
- ✅ Checkpoint tersimpan dengan informasi batch terakhir yang diproses
- ✅ File hasil (.csv) ter-upload ke Google Drive
- ✅ Pesan jelas tentang progress yang tersimpan
- ✅ Instruksi untuk resume ditampilkan

### Scenario 2: Resume After Graceful Interruption
**Objective**: Memastikan sistem dapat melanjutkan dari checkpoint terakhir dengan informasi yang jelas

**Steps**:
1. Setelah Scenario 1, jalankan kembali `python labeling.py`
2. Verifikasi sistem mendeteksi checkpoint
3. Verifikasi informasi resume ditampilkan dengan jelas
4. Biarkan proses beberapa batch lagi

**Expected Results**:
- ✅ Sistem mendeteksi checkpoint dari Google Drive
- ✅ Informasi jelas tentang batch terakhir yang diproses
- ✅ Proses dilanjutkan dari index yang tepat
- ✅ Tidak ada duplikasi data
- ✅ Progress counter akurat

### Scenario 3: Hard Interruption (Power Loss/Force Shutdown)
**Objective**: Memastikan sistem dapat recovery dari interupsi keras tanpa kehilangan data

**Steps**:
1. Jalankan `python labeling.py`
2. Biarkan proses beberapa batch
3. Simulasi hard interruption:
   - Force close terminal (Alt+F4)
   - Atau shutdown komputer secara paksa
   - Atau kill process dari task manager
4. Restart dan jalankan kembali `python labeling.py`

**Expected Results**:
- ✅ Sistem dapat recovery dari checkpoint terakhir di Google Drive
- ✅ Minimal kehilangan data (maksimal 1 batch terakhir)
- ✅ Sistem dapat melanjutkan proses
- ✅ Tidak ada corruption pada file hasil

### Scenario 4: Multi-Device Recovery
**Objective**: Memastikan checkpoint di Google Drive dapat digunakan di komputer berbeda

**Steps**:
1. Jalankan labeling di komputer A
2. Interrupt dengan Ctrl+C
3. Pindah ke komputer B dengan:
   - Setup project yang sama
   - Credentials Google Drive yang sama
4. Jalankan `python labeling.py` di komputer B

**Expected Results**:
- ✅ Checkpoint ter-download dari Google Drive
- ✅ Proses dilanjutkan dari posisi terakhir
- ✅ Tidak ada konflik data
- ✅ Hasil akhir konsisten

### Scenario 5: Directory Recovery
**Objective**: Memastikan sistem dapat membuat ulang direktori Google Drive jika dihapus

**Steps**:
1. Hapus direktori berikut di Google Drive:
   - `ujaran-kebencian-labeling/`
   - `ujaran-kebencian-datasets/`
   - Atau folder project lainnya
2. Jalankan `python labeling.py`
3. Verifikasi sistem membuat ulang struktur folder

**Expected Results**:
- ✅ Sistem otomatis membuat ulang folder structure
- ✅ Proses labeling dapat berjalan normal
- ✅ Checkpoint dan hasil tersimpan ke folder baru

## Implementation Requirements

### 1. Enhanced Directory Creation
```python
def _setup_project_folders(self):
    """
    Setup project folder structure di Google Drive dengan recovery mechanism
    """
    if not self._authenticated:
        return
    
    try:
        # Create main project folder
        self.project_folder_id = self._get_or_create_folder(self.project_folder)
        
        # Create subfolders
        self.checkpoint_folder_id = self._get_or_create_folder(
            'checkpoints', 
            parent_id=self.project_folder_id
        )
        
        self.datasets_folder_id = self._get_or_create_folder(
            'datasets',
            parent_id=self.project_folder_id
        )
        
        self.results_folder_id = self._get_or_create_folder(
            'results',
            parent_id=self.project_folder_id
        )
        
        print(f"✅ Project folders setup complete: {self.project_folder}")
        
    except Exception as e:
        print(f"❌ Failed to setup project folders: {e}")
        self._offline_mode = True
```

### 2. Enhanced Resume Information
```python
def display_resume_info(self, checkpoint_data):
    """
    Display clear resume information
    """
    processed_count = len(checkpoint_data.get('processed_indices', []))
    total_samples = checkpoint_data.get('metadata', {}).get('total_samples', 'Unknown')
    last_batch = checkpoint_data.get('metadata', {}).get('last_batch', 'Unknown')
    timestamp = checkpoint_data.get('timestamp', 'Unknown')
    
    print("\n" + "="*60)
    print("🔄 RESUMING FROM CHECKPOINT")
    print("="*60)
    print(f"📊 Progress: {processed_count}/{total_samples} samples processed")
    print(f"📦 Last batch: {last_batch}")
    print(f"⏰ Last saved: {timestamp}")
    print(f"🎯 Continuing from sample #{processed_count + 1}")
    print("="*60 + "\n")
```

### 3. Robust Checkpoint Validation
```python
def validate_checkpoint(self, checkpoint_data):
    """
    Validate checkpoint data integrity
    """
    required_fields = ['checkpoint_id', 'processed_indices', 'timestamp']
    
    for field in required_fields:
        if field not in checkpoint_data:
            raise ValueError(f"Invalid checkpoint: missing {field}")
    
    # Validate data consistency
    processed_indices = checkpoint_data['processed_indices']
    if not isinstance(processed_indices, list):
        raise ValueError("Invalid checkpoint: processed_indices must be list")
    
    return True
```

## Testing Checklist

### Pre-Test Setup
- [ ] Google Drive credentials configured
- [ ] Dataset file available
- [ ] Clean environment (no existing checkpoints)
- [ ] Network connectivity verified

### Scenario 1: Graceful Interruption
- [ ] Process starts successfully
- [ ] Multiple batches processed
- [ ] Ctrl+C handled gracefully
- [ ] Checkpoint saved locally
- [ ] Checkpoint synced to Google Drive
- [ ] Clear status messages displayed

### Scenario 2: Resume After Graceful
- [ ] Checkpoint detected on restart
- [ ] Resume information displayed clearly
- [ ] Process continues from correct position
- [ ] No data duplication
- [ ] Progress counter accurate

### Scenario 3: Hard Interruption Recovery
- [ ] Process survives hard interruption
- [ ] Recovery from Google Drive checkpoint
- [ ] Minimal data loss (≤1 batch)
- [ ] Process continues successfully
- [ ] No file corruption

### Scenario 4: Multi-Device Recovery
- [ ] Checkpoint downloads on different device
- [ ] Process resumes correctly
- [ ] No data conflicts
- [ ] Consistent final results

### Scenario 5: Directory Recovery
- [ ] Detects missing directories
- [ ] Recreates folder structure
- [ ] Process continues normally
- [ ] Files saved to new folders

## Error Handling Requirements

### Network Issues
- Graceful fallback to offline mode
- Retry mechanism for temporary failures
- Clear error messages for permanent failures

### Authentication Issues
- Clear instructions for re-authentication
- Automatic token refresh when possible
- Fallback to offline mode when needed

### File System Issues
- Handle permission errors
- Manage disk space issues
- Temporary file cleanup

## Performance Considerations

### Checkpoint Frequency
- Balance between data safety and performance
- Configurable checkpoint intervals
- Emergency checkpoint on interruption

### Cloud Sync Optimization
- Batch uploads when possible
- Compress large checkpoint files
- Incremental sync for large datasets

## Documentation Updates Needed

1. **User Guide**: Add recovery scenarios to main documentation
2. **Troubleshooting Guide**: Expand with recovery procedures
3. **Developer Guide**: Document checkpoint format and validation
4. **Setup Guide**: Include directory structure explanation

## Success Criteria

The system passes all recovery tests when:
- ✅ Zero data loss in graceful interruptions
- ✅ ≤1 batch data loss in hard interruptions
- ✅ 100% success rate in resume operations
- ✅ Clear and accurate status information
- ✅ Automatic directory recovery
- ✅ Multi-device compatibility
- ✅ Robust error handling

---

**Last Updated**: 2025-01-01
**Version**: 1.0
**Status**: Ready for Implementation and Testing