# Setting Up Remote Git Repository

Your local Git repository has been initialized and the initial commit has been made. Follow these steps to set up a remote repository:

## Option 1: GitHub (Recommended)

1. **Create a new repository on GitHub:**
   - Go to [GitHub.com](https://github.com)
   - Click "New repository"
   - Name it `etf-analysis` or similar
   - Choose "Public" or "Private" as needed
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Connect your local repository to GitHub:**
   ```bash
   cd /Users/vnktajay/Desktop/Etfs
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

## Option 2: GitLab

1. **Create a new project on GitLab:**
   - Go to [GitLab.com](https://gitlab.com)
   - Click "New project" → "Create blank project"
   - Name it `etf-analysis`
   - Choose visibility level
   - **DO NOT** initialize with README
   - Click "Create project"

2. **Connect your local repository to GitLab:**
   ```bash
   cd /Users/vnktajay/Desktop/Etfs
   git remote add origin https://gitlab.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

## Option 3: Bitbucket

1. **Create a new repository on Bitbucket:**
   - Go to [Bitbucket.org](https://bitbucket.org)
   - Click "Create repository"
   - Name it `etf-analysis`
   - Choose "Private" or "Public"
   - **DO NOT** initialize with README
   - Click "Create repository"

2. **Connect your local repository to Bitbucket:**
   ```bash
   cd /Users/vnktajay/Desktop/Etfs
   git remote add origin https://bitbucket.org/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

## Verify Setup

After pushing, verify everything worked:

```bash
git remote -v
git log --oneline
```

## Future Updates

To push future changes:

```bash
git add .
git commit -m "Your commit message"
git push
```

## Important Notes

- **Data Files**: The large CSV files (etf_final_dataset.csv, etf_pairwise_corr.csv, etc.) are excluded by .gitignore to keep the repository size manageable
- **Virtual Environment**: The `etf_env/` folder is excluded to avoid platform-specific issues
- **Cache Files**: Temporary and cache files are excluded for clean repository

## Repository Structure

Your repository now contains:
- ✅ Core analysis script (`etf.py`)
- ✅ Jupyter notebook (`etf_analysis.ipynb`)
- ✅ Performance plots (`plots/` directory)
- ✅ Documentation (`README.md`)
- ✅ Dependencies (`requirements.txt`)
- ✅ Git configuration (`.gitignore`)
- ❌ Large data files (excluded by .gitignore)
- ❌ Virtual environment (excluded by .gitignore)
- ❌ Cache files (excluded by .gitignore)

This setup ensures your repository is clean, professional, and ready for collaboration while keeping it lightweight.
