# Error Handling Guide - Labeling System

## 1. DeepSeek API Errors

### 1.1 Authentication Errors

#### Error: `401 Unauthorized`
**Penyebab**: API key tidak valid atau expired

**Solusi**:
```bash
# 1. Cek API key di .env
cat .env | grep DEEPSEEK_API_KEY

# 2. Test API key
python src/check_env.py

# 3. Jika invalid, update di .env
# DEEPSEEK_API_KEY=your_new_api_key
```

**Pencegahan**:
- Set reminder untuk renewal API key
- Monitor API key expiration
- Backup API key di secure location

### 1.2 Rate Limiting Errors

#### Error: `429 Too Many Requests`
**Penyebab**: Melebihi rate limit API

**Solusi**:
```python
# Automatic retry dengan exponential backoff
# Sudah diimplementasi di src/utils/deepseek_client.py

# Manual: Tunggu dan retry
import time
time.sleep(60)  # Wait 1 minute
```

**Pencegahan**:
- Gunakan batch size lebih kecil (5-10 items)
- Tambah delay antar request
- Monitor usage di DeepSeek dashboard

### 1.3 Quota/Billing Errors

#### Error: `402 Payment Required` atau `403 Quota Exceeded`
**Penyebab**: Habis quota atau billing issue

**Solusi**:
1. **Cek billing status** di DeepSeek dashboard
2. **Top up balance** jika perlu
3. **Upgrade plan** jika quota habis
4. **Pause labeling** sampai resolved

**Pencegahan**:
- Set billing alerts
- Monitor daily spending
- Use cost optimization strategies

### 1.4 Network/Timeout Errors

#### Error: `ConnectionError`, `TimeoutError`
**Penyebab**: Network issues atau server timeout

**Solusi**:
```python
# Automatic retry sudah diimplementasi
# Manual troubleshooting:

# 1. Test internet connection
ping google.com

# 2. Test DeepSeek API endpoint
curl -I https://api.deepseek.com

# 3. Restart labeling dengan checkpoint
python src/demo_persistent_labeling.py
```

**Pencegahan**:
- Stable internet connection
- Increase timeout values
- Use checkpoint system

## 2. Google Drive Integration Errors

### 2.1 Authentication Errors

#### Error: `google.auth.exceptions.RefreshError`
**Penyebab**: Google credentials expired atau invalid

**Solusi**:
```bash
# 1. Test Google Drive connection
python src/test_google_drive_integration.py

# 2. Re-authenticate
# Delete existing credentials
rm ~/.credentials/drive-python-quickstart.json

# 3. Run authentication flow
python src/utils/cloud_checkpoint_manager.py
```

**Pencegahan**:
- Use Service Account untuk production
- Set credential refresh automation
- Backup credentials securely

### 2.2 Permission Errors

#### Error: `403 Forbidden` atau `Permission denied`
**Penyebab**: Insufficient permissions untuk Google Drive

**Solusi**:
1. **Check folder permissions** di Google Drive
2. **Grant access** ke service account email
3. **Use correct folder ID** di configuration

**Pencegahan**:
- Document folder structure
- Use dedicated folder untuk project
- Regular permission audit

### 2.3 Sync Errors

#### Error: `Upload failed` atau `Download failed`
**Penyebab**: Network issues atau file conflicts

**Solusi**:
```python
# Manual sync
from src.utils.cloud_checkpoint_manager import CloudCheckpointManager
manager = CloudCheckpointManager()
manager.force_sync()  # Force upload/download
```

**Pencegahan**:
- Regular sync intervals
- Conflict resolution strategy
- Local backup before sync

## 3. Data Processing Errors

### 3.1 CSV Loading Errors

#### Error: `UnicodeDecodeError`, `pandas.errors.EmptyDataError`
**Penyebab**: Corrupt CSV atau encoding issues

**Solusi**:
```python
# 1. Check file encoding
file -bi src/data_collection/raw-dataset.csv

# 2. Try different encodings
import pandas as pd
df = pd.read_csv('file.csv', encoding='utf-8-sig')
# or
df = pd.read_csv('file.csv', encoding='latin1')

# 3. Validate CSV structure
print(df.head())
print(df.columns)
```

**Pencegahan**:
- Validate CSV before processing
- Use consistent encoding (UTF-8)
- Backup original data

### 3.2 JSON Parsing Errors

#### Error: `json.JSONDecodeError`
**Penyebab**: Invalid JSON response dari API

**Solusi**:
```python
# Debug response
print(f"Raw response: {response.text}")
print(f"Status code: {response.status_code}")

# Fallback parsing
try:
    data = response.json()
except json.JSONDecodeError:
    # Log error dan skip item
    logger.error(f"Invalid JSON: {response.text}")
    continue
```

**Pencegahan**:
- Validate API responses
- Implement fallback parsing
- Log all parsing errors

### 3.3 Checkpoint Corruption

#### Error: `Checkpoint file corrupted`
**Penyebab**: Incomplete write atau disk issues

**Solusi**:
```bash
# 1. List available checkpoints
ls -la src/checkpoints/

# 2. Try backup checkpoint
cp src/checkpoints/backup_*.json src/checkpoints/current.json

# 3. Download from Google Drive
python -c "from src.utils.cloud_checkpoint_manager import CloudCheckpointManager; CloudCheckpointManager().download_checkpoint()"

# 4. Start fresh if all fail
rm src/checkpoints/*.json
python src/demo_persistent_labeling.py
```

**Pencegahan**:
- Multiple checkpoint backups
- Atomic write operations
- Regular checkpoint validation

## 4. Environment & Setup Errors

### 4.1 Missing Dependencies

#### Error: `ModuleNotFoundError`
**Penyebab**: Missing Python packages

**Solusi**:
```bash
# 1. Install missing package
pip install package_name

# 2. Reinstall all requirements
pip install -r requirements.txt --force-reinstall

# 3. Check Python version
python --version  # Should be 3.8+
```

**Pencegahan**:
- Use virtual environment
- Pin package versions
- Regular dependency updates

### 4.2 Environment Variables

#### Error: `KeyError: 'DEEPSEEK_API_KEY'`
**Penyebab**: Missing environment variables

**Solusi**:
```bash
# 1. Check .env file exists
ls -la .env

# 2. Copy from template
cp .env.template .env

# 3. Edit with your values
nano .env

# 4. Test environment
python src/check_env.py
```

**Pencegahan**:
- Document all required env vars
- Use .env.template
- Validate environment on startup

### 4.3 File Permissions

#### Error: `PermissionError`
**Penyebab**: Insufficient file permissions

**Solusi**:
```bash
# Windows
icacls src /grant Everyone:F /T

# Linux/Mac
chmod -R 755 src/
chown -R $USER:$USER src/
```

**Pencegahan**:
- Proper file permissions setup
- Run with appropriate user
- Document permission requirements

## 5. Performance Issues

### 5.1 Slow API Responses

**Symptoms**: Labeling takes too long

**Diagnosis**:
```python
# Add timing logs
import time
start_time = time.time()
# ... API call ...
end_time = time.time()
print(f"API call took {end_time - start_time:.2f} seconds")
```

**Solutions**:
- Use `deepseek-chat` untuk simple tasks
- Reduce batch size
- Optimize prompts
- Use discount hours

### 5.2 Memory Issues

**Symptoms**: `MemoryError`, system slowdown

**Solutions**:
```python
# Process in smaller chunks
chunk_size = 10  # Reduce from default
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
    
# Clear memory periodically
import gc
gc.collect()
```

### 5.3 Disk Space Issues

**Symptoms**: `No space left on device`

**Solutions**:
```bash
# Check disk usage
df -h

# Clean old logs
find src/logs -name "*.log" -mtime +7 -delete

# Clean old checkpoints
find src/checkpoints -name "*.json" -mtime +30 -delete
```

## 6. Monitoring & Alerting

### 6.1 Error Logging

**Setup comprehensive logging**:
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('src/logs/error.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log errors with context
try:
    # ... operation ...
except Exception as e:
    logger.error(f"Operation failed: {str(e)}", exc_info=True)
```

### 6.2 Health Checks

**Regular system health monitoring**:
```python
# Create health check script
def health_check():
    checks = {
        'api_key': check_api_key(),
        'google_drive': check_google_drive(),
        'disk_space': check_disk_space(),
        'memory': check_memory_usage()
    }
    return all(checks.values()), checks

# Run daily
if __name__ == "__main__":
    healthy, details = health_check()
    if not healthy:
        send_alert(details)
```

### 6.3 Cost Monitoring

**Track API costs**:
```python
# Monitor daily spending
def check_daily_cost():
    today_cost = calculate_today_usage()
    budget_limit = get_daily_budget()
    
    if today_cost > budget_limit * 0.8:  # 80% threshold
        send_cost_alert(today_cost, budget_limit)
    
    return today_cost < budget_limit
```

## 7. Recovery Procedures

### 7.1 Complete System Recovery

**When everything fails**:
```bash
# 1. Backup current state
cp -r src/checkpoints src/checkpoints_backup
cp -r src/logs src/logs_backup

# 2. Fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# or
fresh_env\Scripts\activate  # Windows

# 3. Reinstall
pip install -r requirements.txt

# 4. Restore configuration
cp .env.backup .env

# 5. Test basic functionality
python src/check_env.py

# 6. Resume from last checkpoint
python src/demo_persistent_labeling.py
```

### 7.2 Data Recovery

**When data is lost**:
```bash
# 1. Check Google Drive backup
python -c "from src.utils.cloud_checkpoint_manager import CloudCheckpointManager; CloudCheckpointManager().list_backups()"

# 2. Download latest backup
python -c "from src.utils.cloud_checkpoint_manager import CloudCheckpointManager; CloudCheckpointManager().restore_latest()"

# 3. Verify data integrity
python -c "import pandas as pd; df = pd.read_csv('src/quick-demo-results.csv'); print(f'Recovered {len(df)} records')"
```

## 8. Emergency Contacts & Escalation

### Level 1: Self-Service
- Check this error guide
- Review logs in `src/logs/`
- Try basic troubleshooting

### Level 2: Team Support
- Consult team arsitek
- Update issue in `memory-bank/papan-proyek.md`
- Share error logs dengan tim

### Level 3: External Support
- DeepSeek API support
- Google Cloud support
- Infrastructure team

---

**💡 Pro Tips**:
- Always backup before major changes
- Test in small batches first
- Monitor costs in real-time
- Keep error logs for analysis
- Document new error scenarios