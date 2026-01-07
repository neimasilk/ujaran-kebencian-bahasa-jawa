# Javanese Hate Speech Detection Project Structure

This document outlines the organized structure of the project repository. The codebase has been cleaned to separate the active research paper submission work from legacy experiments and artifacts.

## 📂 Active Directories

| Directory | Description |
|-----------|-------------|
| **`docs/`** | **[PRIMARY]** Contains the active documentation and paper submission files. |
| &nbsp;&nbsp;&nbsp;&nbsp;`paper-submission/` | The main workspace for the final paper, LaTeX source, and compiled PDF. |
| **`data/`** | Contains dataset files (raw, processed, augmented). |
| **`models/`** | Saved model checkpoints and weights. |
| **`src/`** | Source code for the application/models. |
| **`tests/`** | Unit and integration tests. |
| **`memory-bank/`** | Project context and memory files for AI assistants. |

## 🗄️ Archive Directory (`archive/`)

All previous experiments, logs, scripts, and temporary files have been moved here to declutter the root workspace.

| Sub-directory | Contents |
|---------------|----------|
| `assets/` | Image files (`.png`) from experiments (plots, confusion matrices). |
| `scripts/` | All Python scripts (`.py`) and Notebooks (`.ipynb`) from previous phases. |
| `docs/` | Archived documentation files (`.md`, `.pdf`, `.docx`, `.bib`). |
| `results/` | Experimental results data (`.json`, `.csv`, `.db`, `.xml`). |
| `legacy_structure/` | Old folder structures (`experiments/`, `logs/`, `paper/`, etc.) preserved as-is. |
| `tmp_artifacts/` | Temporary folders (`tmp_*`) generated during training runs. |

## 🚀 Getting Started

1. **Paper Submission:** Navigate to `docs/paper-submission/` to find the latest draft and LaTeX files.
2. **Reproducing Results:** Refer to the scripts in `archive/scripts/` if you need to re-run specific historical experiments, but be aware they are now in an archive path.
3. **New Experiments:** Create new scripts in a dedicated `scripts/` folder (if needed) or within `src/` to keep the root clean.

---
*Last Updated: 7 January 2026*
