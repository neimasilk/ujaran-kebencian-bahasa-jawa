# Testing Scenarios Documentation

## Overview
Dokumentasi lengkap untuk testing scenarios sistem Google Drive labeling, mencakup skenario interupsi dan recovery yang diminta.

## Test Scenarios

### 1. Graceful Interruption (Ctrl+C)

**Objective**: Menguji kemampuan sistem untuk menangani interupsi graceful dan menyimpan checkpoint dengan benar.

**Steps**:
1. Jalankan `python labeling.py`
2. Biarkan proses beberapa batch (5-10 batch)
3. Tekan `Ctrl+C` untuk interrupt
4. Sistem harus:
   - Menampilkan pesan "🛑 Received signal 2 (Ctrl+C). Initiating graceful shutdown..."
   - Menyimpan checkpoint lokal
   - Melakukan emergency sync ke Google Drive
   - Menampilkan pesan "💾 Saving checkpoint and syncing to Google Drive..."
   - Exit dengan pesan "🔄 Process stopped. Run 'python labeling.py' again to resume."

**Expected Results**:
- Checkpoint tersimpan di lokal dan cloud
- Tidak ada data loss
- Proses berhenti dengan graceful

**Test Command**:
```bash
python test_recovery_scenarios.py --scenario graceful
```

### 2. Resume Functionality

**Objective**: Menguji kemampuan sistem untuk resume dari checkpoint dengan informasi yang jelas.

**Steps**:
1. Setelah graceful interruption, jalankan kembali `python labeling.py`
2. Sistem harus:
   - Mendeteksi checkpoint yang ada
   - Menampilkan informasi resume yang jelas:
     ```
     📋 RESUME INFORMATION
     =====================
     📁 Dataset: data/sample_test_data.csv
     📊 Progress: 25/50 records processed (50.0%)
     📝 Output File: test_recovery_output.csv
     🕐 Last Checkpoint: 2024-01-15 10:30:45
     🔄 Resuming from batch: 6
     ```
   - Melanjutkan dari batch terakhir yang diproses
   - Tidak memproses ulang data yang sudah selesai

**Expected Results**:
- Informasi resume jelas dan akurat
- Tidak ada duplikasi processing
- Melanjutkan dari posisi yang tepat

**Test Command**:
```bash
python labeling.py  # Setelah interruption
```

### 3. Hard Interruption (Power Loss/Terminal Kill)

**Objective**: Menguji recovery setelah interupsi keras tanpa graceful shutdown.

**Steps**:
1. Jalankan `python labeling.py`
2. Biarkan proses beberapa batch
3. Simulasi power loss dengan:
   - Kill terminal secara paksa
   - Kill process dengan `taskkill /F /PID <pid>` (Windows)
   - Atau matikan komputer secara paksa
4. Restart dan jalankan kembali `python labeling.py`

**Expected Results**:
- Sistem dapat recovery dari checkpoint terakhir
- Mungkin kehilangan 1-2 batch terakhir (acceptable)
- Resume dari checkpoint cloud jika tersedia
- Fallback ke checkpoint lokal jika cloud tidak tersedia

**Test Command**:
```bash
python test_recovery_scenarios.py --scenario hard
```

### 4. Multi-Device Recovery

**Objective**: Menguji kemampuan resume di komputer/device yang berbeda menggunakan cloud checkpoint.

**Steps**:
1. **Device 1**: Jalankan labeling, proses beberapa batch, interrupt
2. **Device 2**: Clone repository di komputer lain
3. **Device 2**: Setup Google Drive authentication
4. **Device 2**: Jalankan `python labeling.py`
5. Sistem harus:
   - Download checkpoint dari Google Drive
   - Menampilkan informasi resume
   - Melanjutkan dari posisi terakhir Device 1

**Expected Results**:
- Seamless transition antar device
- Cloud checkpoint sebagai "golden standard"
- Tidak ada data loss atau duplikasi

**Test Command**:
```bash
python test_recovery_scenarios.py --scenario multi_device
```

### 5. Google Drive Folder Recovery

**Objective**: Menguji kemampuan sistem untuk recreate folder Google Drive yang terhapus.

**Steps**:
1. Hapus 3 folder di Google Drive:
   - `ujaran-kebencian-datasets`
   - `ujaran-kebencian-datasets-test` 
   - `ujaran-kebencian-labeling`
2. Jalankan `python labeling.py`
3. Sistem harus:
   - Mendeteksi folder yang hilang
   - Recreate folder structure:
     ```
     ujaran-kebencian-labeling/
     ├── checkpoints/
     ├── datasets/
     └── results/
     ```
   - Menampilkan pesan recovery

**Expected Results**:
- Folder structure ter-recreate otomatis
- Sistem dapat melanjutkan operasi normal
- Tidak ada error karena missing folders

**Test Command**:
```bash
python test_recovery_scenarios.py --scenario folder
```

## Implementation Details

### New Methods Added

#### CloudCheckpointManager

1. **`verify_and_recover_folders()`**
   - Verifikasi eksistensi folder structure
   - Recreate jika hilang
   - Return True jika berhasil

2. **`display_resume_info(checkpoint_data)`**
   - Menampilkan informasi resume yang jelas
   - Format user-friendly
   - Include progress percentage

3. **`validate_checkpoint(checkpoint_data)`**
   - Validasi integritas checkpoint
   - Check required fields
   - Verify data consistency

#### GoogleDriveLabelingPipeline

1. **Enhanced `setup()`**
   - Added folder verification step
   - Better error handling

2. **Enhanced `run_labeling()`**
   - Improved checkpoint validation
   - Clear resume information display
   - Fallback mechanism (cloud → local → fresh start)

### Error Handling Strategy

1. **Graceful Degradation**:
   - Cloud checkpoint → Local checkpoint → Fresh start
   - Continue operation even if some components fail

2. **Validation**:
   - Validate checkpoint integrity before use
   - Clear error messages for debugging

3. **Recovery**:
   - Auto-recovery for common issues
   - Manual intervention guidance when needed

## Testing Checklist

### Pre-Test Setup
- [ ] Google Drive authentication working
- [ ] Sample dataset available
- [ ] Clean test environment
- [ ] Backup important data

### Graceful Interruption Test
- [ ] Process starts successfully
- [ ] Several batches processed
- [ ] Ctrl+C handled gracefully
- [ ] Emergency sync completed
- [ ] Checkpoint saved locally and cloud
- [ ] Clear shutdown message displayed

### Resume Test
- [ ] Checkpoint detected on restart
- [ ] Resume information displayed clearly
- [ ] Processing continues from correct position
- [ ] No duplicate processing
- [ ] Progress tracking accurate

### Hard Interruption Test
- [ ] Process killed abruptly
- [ ] Recovery possible on restart
- [ ] Minimal data loss (acceptable)
- [ ] Cloud checkpoint prioritized
- [ ] Fallback to local checkpoint works

### Multi-Device Test
- [ ] Checkpoint synced to cloud
- [ ] Different device can authenticate
- [ ] Cloud checkpoint downloaded
- [ ] Resume works across devices
- [ ] Data consistency maintained

### Folder Recovery Test
- [ ] Folders can be deleted manually
- [ ] System detects missing folders
- [ ] Folders recreated automatically
- [ ] Correct folder structure
- [ ] Operations continue normally

## Performance Considerations

### Checkpoint Frequency
- Save checkpoint every 5 batches (configurable)
- Emergency sync on interruption
- Periodic cloud sync every 5 minutes

### Error Recovery Time
- Folder verification: < 10 seconds
- Checkpoint validation: < 5 seconds
- Cloud sync: depends on network (typically < 30 seconds)

### Resource Usage
- Minimal overhead for checkpoint operations
- Efficient cloud API usage
- Local caching to reduce API calls

## Troubleshooting Guide

### Common Issues

1. **"Cloud checkpoint validation failed"**
   - Checkpoint file corrupted
   - Network interruption during save
   - Solution: Use local checkpoint or start fresh

2. **"Could not verify folder structure"**
   - Google Drive API quota exceeded
   - Authentication expired
   - Solution: Re-authenticate, wait for quota reset

3. **"Resume information not available"**
   - No valid checkpoint found
   - Checkpoint format changed
   - Solution: Start fresh labeling process

### Debug Commands

```bash
# Test all scenarios
python test_recovery_scenarios.py --scenario all

# Test specific scenario
python test_recovery_scenarios.py --scenario graceful
python test_recovery_scenarios.py --scenario hard
python test_recovery_scenarios.py --scenario multi_device
python test_recovery_scenarios.py --scenario folder

# Manual folder verification
python -c "from src.utils.cloud_checkpoint_manager import CloudCheckpointManager; cm = CloudCheckpointManager(); cm.authenticate(); cm.verify_and_recover_folders()"
```

## Success Criteria

### Must Have
- [x] Graceful interruption handling
- [x] Clear resume information
- [x] Multi-device continuity
- [x] Folder auto-recovery
- [x] Checkpoint validation

### Should Have
- [x] Comprehensive error messages
- [x] Performance optimization
- [x] Automated testing
- [x] Documentation

### Nice to Have
- [ ] Web dashboard for monitoring
- [ ] Email notifications on errors
- [ ] Advanced analytics

## Conclusion

Sistem recovery dan testing scenarios telah diimplementasi dengan fokus pada:

1. **Robustness**: Sistem dapat handle berbagai jenis interupsi
2. **Clarity**: Informasi resume yang jelas untuk user
3. **Continuity**: Multi-device support dengan cloud checkpoints
4. **Recovery**: Auto-recovery untuk folder structure
5. **Testing**: Comprehensive test scenarios untuk validasi

Semua skenario testing telah didokumentasikan dan dapat dijalankan secara otomatis atau manual sesuai kebutuhan development dan debugging.