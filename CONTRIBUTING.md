# Contributing to the Hackathon

Thank you for participating in the **3LC x AWS Cloud @ UT Dallas Hackathon**!

## For Participants

This repository contains the **starter code** for the hackathon challenge. Here's how to use it:

### 1. Fork or Clone This Repository

**Option A: Fork (Recommended for submission)**
```bash
# Click "Fork" on GitHub, then:
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```

**Option B: Clone directly**
```bash
git clone https://github.com/ORIGINAL_REPO_URL.git
cd REPO_NAME
```

### 2. Set Up Your Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (link provided at hackathon)
# Extract to data/ folder

# Verify setup
python src/register_tables.py
```

### 3. Work on Your Solution

- Modify the code in `src/` or `notebooks/`
- Experiment with different approaches
- Use 3LC for data-centric improvements
- Document your changes

### 4. Commit Your Progress

```bash
git add .
git commit -m "Describe your changes"
git push origin main
```

### 5. Submission Requirements

For final submission, your repository should include:

1. **All your code** (models, training scripts, notebooks)
2. **Technical writeup** (1-2 pages as `WRITEUP.md` or PDF)
3. **Proof of scores** (screenshots or 3LC table exports in `results/`)
4. **Updated README** describing your approach
5. **Requirements** for reproducing your work

### 6. Invite Collaborators

Add these GitHub users as collaborators to your repository:
- `@paulendresen` (Paul Endresen - 3LC)
- `@rishikeshjadh` (Rishikesh Jadhav - AWS Cloud Club)

**How to add collaborators:**
1. Go to your repo → Settings → Collaborators
2. Click "Add people"
3. Enter their GitHub usernames
4. Send invitations

### 7. Submit via Google Form

Complete the submission form (link provided) with:
- Your GitHub repository URL
- Final accuracy score
- 3LC username
- Brief summary

## Code of Conduct

- ✅ Be respectful and collaborative
- ✅ Ask questions when stuck
- ✅ Share knowledge (after deadline)
- ✅ Submit original work
- ❌ No plagiarism or code copying
- ❌ No sharing solutions before deadline

## Getting Help

**During the hackathon:**
- **Slack/Discord:** #hackathon-support
- **In-person:** Visit the help desk
- **Documentation:** Check `README.md` and `HACKATHON_OVERVIEW_DOCUMENT.md`

**Technical issues:**
- 3LC problems: Contact Paul or check [docs.3lc.ai](https://docs.3lc.ai)
- Setup issues: Ask in #hackathon-support
- Dataset issues: Contact organizers

## Best Practices

### Git Workflow
```bash
# Create a feature branch for experiments
git checkout -b experiment-resnet

# Make changes and commit regularly
git add src/train.py
git commit -m "Add ResNet18 transfer learning"

# Push to your fork
git push origin experiment-resnet

# Merge when satisfied
git checkout main
git merge experiment-resnet
```

### Project Organization
```
your-repo/
├── src/                  # Your training scripts
├── notebooks/            # Jupyter notebooks
├── models/               # Saved model checkpoints
├── results/              # Metrics, plots, screenshots
│   ├── metrics.csv
│   ├── plots/
│   └── 3lc_exports/
├── WRITEUP.md           # Your technical writeup
└── README.md            # Updated with your approach
```

### Documentation
- Comment your code clearly
- Explain your data-centric approach
- Document what worked and what didn't
- Include visualizations from 3LC

## Questions?

Check the comprehensive guide in `HACKATHON_OVERVIEW_DOCUMENT.md` or ask in the support channels!

---

**Good luck and happy hacking!** 🚀

