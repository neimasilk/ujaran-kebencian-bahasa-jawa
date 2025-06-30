# Workflow Diagram - Labeling System

## 1. Overall System Flow

```mermaid
flowchart TD
    A[Raw Dataset CSV] --> B[Load Data]
    B --> C{Cost Strategy}
    C -->|Discount Only| D[Wait for Discount Hours]
    C -->|Always| E[Start Labeling]
    D --> E
    E --> F[DeepSeek API Call]
    F --> G{API Response}
    G -->|Success| H[Save to Checkpoint]
    G -->|Error| I[Retry Logic]
    I --> F
    H --> J[Sync to Google Drive]
    J --> K{More Data?}
    K -->|Yes| E
    K -->|No| L[Final Results CSV]
```

## 2. Cost Optimization Flow

```mermaid
flowchart TD
    A[Start Labeling Session] --> B{Check Current Time}
    B -->|Discount Hours<br/>19:00-08:00 UTC+8| C[Use Selected Model]
    B -->|Standard Hours<br/>08:00-19:00 UTC+8| D{Strategy Mode}
    D -->|Discount Only| E[Queue for Later]
    D -->|Always| F{Data Type}
    F -->|Simple/Negative| G[Use deepseek-chat]
    F -->|Complex/Positive| H[Use deepseek-reasoner]
    C --> I[Process Batch]
    G --> I
    H --> I
    E --> J[Wait for Discount]
    J --> C
    I --> K[Update Cost Tracking]
```

## 3. Persistence & Checkpoint Flow

```mermaid
flowchart TD
    A[Start Labeling] --> B{Checkpoint Exists?}
    B -->|Yes| C[Load from Checkpoint]
    B -->|No| D[Start Fresh]
    C --> E[Resume from Last Position]
    D --> E
    E --> F[Process Batch]
    F --> G[Save Local Checkpoint]
    G --> H{Sync Interval?}
    H -->|Yes| I[Upload to Google Drive]
    H -->|No| J{More Batches?}
    I --> J
    J -->|Yes| E
    J -->|No| K[Final Sync]
    K --> L[Complete]
```

## 4. Error Handling Flow

```mermaid
flowchart TD
    A[API Call] --> B{Response Status}
    B -->|200 OK| C[Parse Response]
    B -->|Rate Limit| D[Wait & Retry]
    B -->|Auth Error| E[Check API Key]
    B -->|Network Error| F[Retry with Backoff]
    B -->|Other Error| G[Log Error]
    C --> H{Valid JSON?}
    H -->|Yes| I[Extract Labels]
    H -->|No| J[Log Parse Error]
    D --> A
    E --> K[Update Credentials]
    F --> A
    G --> L[Skip Item]
    I --> M[Save Result]
    J --> L
    K --> A
    L --> N[Continue Next]
    M --> N
```

## 5. Data Processing Pipeline

```mermaid
flowchart LR
    A[Raw Text] --> B[Text Preprocessing]
    B --> C[Batch Formation]
    C --> D[API Request]
    D --> E[Response Processing]
    E --> F[Label Extraction]
    F --> G[Validation]
    G --> H{Valid Label?}
    H -->|Yes| I[Save to Results]
    H -->|No| J[Mark for Review]
    I --> K[Update Progress]
    J --> K
    K --> L{Batch Complete?}
    L -->|No| C
    L -->|Yes| M[Checkpoint Save]
```

## 6. Team Workflow

```mermaid
flowchart TD
    A[Team Member] --> B[Read Quick Start Guide]
    B --> C[Setup Environment]
    C --> D[Run Environment Check]
    D --> E{Setup OK?}
    E -->|No| F[Fix Issues]
    F --> D
    E -->|Yes| G[Choose Labeling Mode]
    G --> H{Mode Type}
    H -->|Demo| I[Run Quick Demo]
    H -->|Production| J[Run Persistent Labeling]
    I --> K[Review Results]
    J --> L[Monitor Progress]
    K --> M[Report Status]
    L --> N{Session Complete?}
    N -->|No| L
    N -->|Yes| O[Backup Checkpoint]
    M --> P[Update Project Board]
    O --> P
```

## 7. Daily Operations Flow

```mermaid
flowchart TD
    A[Start Day] --> B[Check Previous Progress]
    B --> C[Review Cost Budget]
    C --> D[Plan Today's Batch]
    D --> E{Discount Hours?}
    E -->|Yes| F[Start Labeling]
    E -->|No| G[Queue for Later]
    F --> H[Monitor Progress]
    G --> I[Work on Other Tasks]
    H --> J{Batch Complete?}
    J -->|No| H
    J -->|Yes| K[Backup Results]
    I --> L{Discount Time?}
    L -->|Yes| F
    L -->|No| I
    K --> M[Update Documentation]
    M --> N[End Day]
```

## Key Components

### Input Files
- `src/data_collection/raw-dataset.csv` - Raw unlabeled data
- `.env` - API keys and configuration
- `src/checkpoints/*.json` - Progress checkpoints

### Processing Scripts
- `src/demo_cost_efficient_labeling.py` - Quick demo
- `src/demo_persistent_labeling.py` - Production labeling
- `src/demo_cost_optimization.py` - Cost analysis

### Output Files
- `src/*-results.csv` - Labeled results
- `src/checkpoints/*.json` - Progress saves
- `src/logs/*.log` - Operation logs

### Utilities
- `src/utils/deepseek_client.py` - API client
- `src/utils/cost_optimizer.py` - Cost management
- `src/utils/cloud_checkpoint_manager.py` - Google Drive sync

## Time Estimates

| Task | Duration | Notes |
|------|----------|-------|
| Environment Setup | 5-10 min | First time only |
| Quick Demo (10 items) | 2-3 min | Testing purposes |
| Batch Labeling (100 items) | 15-30 min | Depends on model |
| Checkpoint Sync | 1-2 min | Per batch |
| Daily Backup | 5 min | End of day |

## Cost Optimization Schedule

| Time (UTC+8) | Status | Recommended Action |
|--------------|--------|--------------------|
| 08:00-19:00 | Standard Rate | Use discount-only mode or simple model |
| 19:00-08:00 | Discount Rate | Full labeling with any model |

---

**Note**: Diagram ini menggunakan Mermaid syntax. Untuk visualisasi yang lebih baik, gunakan tools yang mendukung Mermaid seperti GitHub, GitLab, atau Mermaid Live Editor.