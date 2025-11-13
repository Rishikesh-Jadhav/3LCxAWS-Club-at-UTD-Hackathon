# 3LC x AWS Cloud @ UT Dallas Hackathon
## Data Centric AI Challenge - Chihuahua vs Muffin Classification

**Document Version:** 1.0  
**Date:** November 18th, 2025  
**Authors:** Paul Endresen (3LC) & Rishikesh Jadhav (AWS Cloud Club @ UTD)

---

## 1. Overview

### Event Details
- **Event Name:** 3LC x AWS Cloud Club @ UT Dallas Hackathon
- **Challenge:** Data Centric AI Image Classification
- **Dataset:** Chihuahua vs Muffin (Binary Classification)
- **When:** November 18th, 2025 – Kickoff at 3PM
- **Where:** JSOM 1.118, UT Dallas Campus
- **Duration:** 48 hours
- **Format:** In-person hackathon with hands-on support

### What is Data-Centric AI?

Data-centric AI is an approach that focuses on **systematically improving the quality of data** rather than just improving model architectures. Instead of building bigger and more complex models, participants will learn to:

- **Analyze datasets** using advanced visualization tools
- **Identify problematic samples** that confuse models
- **Iterate intelligently** on training data quality
- **Track improvements** through systematic experimentation

This hackathon introduces participants to **3LC (Three Lines of Code)**, a cutting-edge platform for data-centric AI workflows.

---

## 2. Description

### Challenge Statement

Participants will build an image classification system to distinguish between two visually similar classes:
- **Chihuahuas** (small dogs)
- **Muffins** (baked goods)

This challenge is deceptively difficult because muffins and chihuahuas share surprising visual similarities (color, texture, shape), making it a perfect case study for data-centric AI techniques.

### Learning Objectives

By the end of this hackathon, participants will:

1. **Understand data-centric AI principles** and how they differ from model-centric approaches
2. **Use 3LC tools** for dataset analysis, visualization, and improvement
3. **Implement the Train-Fix-Retrain loop** to iteratively improve model performance
4. **Work with embeddings** to understand data distribution and identify edge cases
5. **Track experiments** using table revisions and metrics
6. **Apply best practices** in machine learning workflows

### Technical Stack

- **Programming Language:** Python 3.8+
- **Deep Learning Framework:** PyTorch
- **Data-Centric Platform:** 3LC (Three Lines of Code)
- **Cloud Infrastructure:** AWS S3 for dataset hosting
- **Version Control:** Git/GitHub

---

## 3. Dataset Description

### Dataset: Chihuahua vs Muffin

**Source:** Custom curated dataset for this hackathon  
**Format:** RGB images in JPEG/PNG format  
**Classes:** 2 (Binary Classification)

### Dataset Structure

```
data/
├── train/
│   ├── chihuahua/    (100 images)
│   └── muffin/       (100 images)
└── test/             (Held-out test set)
```

### Specifications

| Property | Value |
|----------|-------|
| **Image Size** | 128 x 128 pixels |
| **Training Images** | 200 total (100 per class) |
| **Test Images** | Separate evaluation set (labels withheld) |
| **Color Space** | RGB (3 channels) |
| **File Format** | JPEG, PNG |
| **Class Balance** | Balanced (50/50 split) |

### Dataset Characteristics

- **Visual Similarity:** High similarity between classes (golden-brown colors, round shapes, textured surfaces)
- **Variety:** Different angles, lighting conditions, and backgrounds
- **Difficulty:** Ambiguous samples that challenge even human classification
- **Quality:** Some samples may have labeling errors or be truly ambiguous (intentional for data-centric learning)

### Dataset Access

**📦 Public S3 Bucket:** [Will be provided at kickoff]

Participants will download the dataset from an **AWS S3 public bucket**. The link will be shared during the kickoff session and will be accessible throughout the hackathon.

### Dataset Download Instructions

1. Download from the provided S3 link
2. Extract the zip file
3. Place contents in the `data/` folder of the starter code
4. Verify folder structure matches the layout above

---

## 4. Evaluation Methodology

### Primary Metric

**Classification Accuracy** on the held-out test set

```
Accuracy = (Correct Predictions / Total Predictions) × 100%
```

### Evaluation Process

1. **Model Submission:** Participants submit their trained model and code via GitHub
2. **Test Set Evaluation:** Organizers run models on the withheld test set
3. **Score Recording:** Final accuracy scores are recorded
4. **Verification:** Top 3 submissions are verified for reproducibility

### Submission Requirements

Participants must submit:

1. **GitHub Repository** containing:
   - All source code (training scripts, notebooks, models)
   - 1-2 page technical writeup (PDF or Markdown)
   - Proof of scores (screenshots or zipped 3LC table exports)
   - Updated README with approach documentation

2. **Google Form Submission** (link provided during event):
   - Final test accuracy score
   - GitHub repository link
   - 3LC username
   - Team/individual name

3. **GitHub Collaborators:**
   - Invite **@paulendresen** (Paul Endresen)
   - Invite **@rishikeshjadh** (Rishikesh Jadhav)

### Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Test Accuracy** | 60% | Performance on held-out test set |
| **Approach Quality** | 20% | Use of data-centric techniques |
| **Code Quality** | 10% | Clean, documented, reproducible code |
| **Writeup** | 10% | Clear explanation of approach and insights |

### Baseline Performance

**Baseline Model:** Simple CNN (provided in starter code)  
**Baseline Accuracy:** ~83%

**Goal:** Beat the baseline using data-centric AI techniques!

---

## 5. Setup Instructions

### Prerequisites

- **Computer:** Laptop with Python 3.8+ installed
- **Operating System:** Windows, macOS, or Linux
- **RAM:** Minimum 8GB (16GB recommended)
- **Storage:** 2GB free space
- **GPU:** Optional but recommended for faster training
- **Internet:** Required for downloading dataset and packages

### Installation Steps

#### Step 1: Download Starter Code

```bash
# Download from GitHub (link provided at event)
git clone <repository-url>
cd Hackathon_Image_Classification_Challenge
```

#### Step 2: Set Up Python Environment

**Option A: Using pip (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda create -n hackathon python=3.8
conda activate hackathon
pip install -r requirements.txt
```

#### Step 3: Install 3LC

```bash
pip install tlc
```

#### Step 4: Sign Up for 3LC

1. Go to [https://3lc.ai](https://3lc.ai)
2. Create a free account
3. Get your API key from the dashboard
4. **Important:** Must register to be eligible for prizes!

#### Step 5: Download Dataset

1. Download from the S3 link provided
2. Extract to `data/` folder
3. Verify structure:
   ```
   data/
   ├── train/
   │   ├── chihuahua/
   │   └── muffin/
   └── test/
   ```

#### Step 6: Test Setup

```bash
# Run dataset registration script
python src/register_tables.py

# Open starter notebook
jupyter notebook notebooks/starter_notebook.ipynb
```

### Package Requirements

Key dependencies (see `requirements.txt` for full list):
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `numpy`, `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `Pillow`, `opencv-python` - Image processing
- `scikit-learn` - ML utilities
- `jupyter` - Notebook environment
- `tlc` - 3LC platform (installed separately)

---

## 6. New Concepts: Tables and Datasets

### What is 3LC?

**3LC (Three Lines of Code)** is a data-centric AI platform that helps ML practitioners:
- Visualize datasets in rich, interactive dashboards
- Explore embeddings to understand data distribution
- Identify problematic samples that hurt performance
- Track table revisions across iterations
- Make data-driven decisions about dataset improvements

### Core Concepts

#### 1. Tables

**Definition:** A 3LC Table is an enhanced dataset with metadata, embeddings, and quality metrics.

**Key Features:**
- **Versioning:** Tables support revisions to track changes over time
- **Metadata:** Each sample has rich metadata (predictions, confidence, embeddings)
- **Metrics:** Track quality metrics across the entire dataset
- **Lineage:** Understand how datasets evolve through iterations

**Analogy:** Think of Tables like **Git for datasets** – just as Git tracks code changes, 3LC tracks dataset revisions.

#### 2. Datasets

**Definition:** Collections of data registered with 3LC for analysis.

**Support:** Images, text, tabular data, and more

**Registration:**
```python
import tlc
table = tlc.Table.from_image_folder(
    "data/train",
    table_name="chihuahua_muffin_v1"
)
```

#### 3. Embeddings

**Definition:** High-dimensional vector representations of data learned by neural networks.

**Purpose:**
- Capture semantic similarity between samples
- Enable visualization in 2D/3D space
- Identify clusters, outliers, and edge cases

**Visualization:** 3LC automatically creates 2D/3D scatter plots using dimensionality reduction (UMAP, t-SNE).

**Use Cases:**
- Find similar images
- Identify mislabeled data
- Discover dataset biases
- Locate ambiguous samples

#### 4. Table Revisions

**Concept:** Each time you modify your dataset, create a new revision.

**Benefits:**
- Compare performance across versions
- A/B test dataset changes
- Roll back to previous versions
- Document improvement journey

**Example Workflow:**
```
v1: Original dataset → 83% accuracy
v2: Removed 10 ambiguous samples → 85% accuracy
v3: Added augmentation → 87% accuracy
v4: Fixed 5 label errors → 89% accuracy
```

### Analogies for Understanding

| Concept | Analogy | Explanation |
|---------|---------|-------------|
| **3LC Tables** | Git for Data | Version control for datasets |
| **Embeddings** | GPS Coordinates | Position of data in "meaning space" |
| **Dashboard** | Google Analytics | Visual insights into data quality |
| **Revisions** | Document Versions | Track iterations and improvements |

---

## 7. Train-Fix-Retrain Loop Instructions

### Overview

The **Train-Fix-Retrain Loop** is the core workflow for data-centric AI. Instead of endlessly tweaking model architectures, you systematically improve your data.

### The 5-Step Process

#### Step 1: Train (Baseline)

**Goal:** Establish baseline performance

**Actions:**
1. Run the provided starter code
2. Train baseline CNN on full dataset
3. Achieve ~83% accuracy
4. Save model and predictions

**Code:**
```python
python src/train.py
# or use notebooks/starter_notebook.ipynb
```

**Output:** 
- Trained model
- Validation accuracy
- Predictions on validation set

---

#### Step 2: Analyze (3LC Dashboard)

**Goal:** Understand where and why the model fails

**Actions:**
1. Register dataset as 3LC Table
2. Upload model predictions and embeddings
3. Open 3LC Dashboard
4. Explore visualizations

**Dashboard Features:**

**a) Embeddings View**
- 2D/3D scatter plot of all images
- Color-coded by class or prediction
- Interactive: click to see image
- Identify clusters and outliers

**b) Table View**
- Sortable list of all samples
- Filter by: class, prediction, confidence
- Custom metrics columns
- Bulk selection tools

**c) Metrics Panel**
- Overall accuracy
- Per-class precision/recall
- Confidence distribution
- Loss curves

**d) Sample Inspector**
- Drill down into individual images
- See predictions, confidence, embeddings
- Compare similar samples
- Flag for removal/relabeling

**Questions to Ask:**
- Which samples are misclassified?
- Are there patterns in errors? (e.g., dark backgrounds, side angles)
- Are there outliers in embedding space?
- Is there a clear separation between classes?
- Are some samples ambiguous even to humans?

---

#### Step 3: Fix (Data-Centric Improvements)

**Goal:** Improve dataset quality based on insights

**Common Fixes:**

**a) Remove Ambiguous Samples**
- Identify samples that are genuinely ambiguous
- Remove images that confuse the model
- Document removals in writeup

**b) Fix Label Errors**
- Find mislabeled images using predictions
- Correct labels manually
- Create new table revision

**c) Rebalance Classes**
- Check for imbalance in subgroups
- Add augmented samples if needed
- Ensure diverse representation

**d) Add Data Augmentation**
- Increase augmentation strength
- Add new transforms (rotation, color jitter, crops)
- Test different augmentation strategies

**e) Remove Outliers**
- Find samples far from cluster centers
- Investigate if they're low-quality or irrelevant
- Remove if justified

**f) Focus on Hard Examples**
- Identify consistently misclassified samples
- Add more similar samples (if available)
- Apply targeted augmentation

**Documentation:** Create a new 3LC table revision documenting your changes.

---

#### Step 4: Retrain

**Goal:** Train on improved dataset

**Actions:**
1. Update dataset based on Step 3 fixes
2. Retrain model with same architecture
3. Evaluate on validation set
4. Upload results to new 3LC table revision

**Fair Comparison:** Use same model architecture to isolate the impact of data improvements.

**Code:**
```python
# Train on improved dataset
python src/train.py --dataset_version v2
```

---

#### Step 5: Iterate

**Goal:** Repeat the loop to maximize accuracy

**Actions:**
1. Compare results in 3LC Dashboard
2. Analyze new embeddings and predictions
3. Identify remaining issues
4. Apply new fixes
5. Retrain again

**Iteration Strategy:**
- Make small, focused changes each iteration
- Track what works and what doesn't
- Document insights in your writeup
- Aim for 2-3 iterations during the hackathon

**Stopping Criteria:**
- Time runs out
- Accuracy plateaus
- No more clear improvements identified

---

### Dashboard Mechanics

#### Navigating the 3LC Dashboard

**1. Project View**
- See all your tables and revisions
- Compare metrics across versions
- Access documentation and tutorials

**2. Table View**
- Default view showing all samples
- Columns: image, label, prediction, confidence, custom metrics
- Sorting: Click column headers
- Filtering: Use filter bar (e.g., "prediction != label")
- Selection: Click checkboxes for bulk actions

**3. Embeddings View**
- Toggle between 2D and 3D
- Color by: class, prediction, confidence, custom metric
- Hover: See thumbnail and metadata
- Click: Open sample inspector
- Lasso select: Select multiple samples

**4. Sample Inspector**
- Large image preview
- Metadata table
- Prediction probabilities
- Similar samples (based on embedding distance)
- Actions: Flag, delete, change label

**5. Revisions Panel**
- See all revisions of current table
- Compare metrics side-by-side
- Diff revisions to see changes
- Switch between revisions

#### Key Dashboard Actions

**Find Misclassified Samples:**
```
Filter: prediction != label
Sort by: confidence (ascending)
```

**Find Low-Confidence Predictions:**
```
Filter: max_confidence < 0.7
Sort by: confidence (ascending)
```

**Find Outliers:**
```
View: Embeddings (2D)
Action: Look for isolated points
```

**Compare Revisions:**
```
1. Open revision A
2. Click "Compare" button
3. Select revision B
4. View side-by-side metrics
```

---

## 8. Incentives

### Cash Prizes

| Place | Prize | Description |
|-------|-------|-------------|
| 🥇 **1st Place** | **$200** | Highest test accuracy with quality writeup |
| 🥈 **2nd Place** | **$100** | Second-highest test accuracy |
| 🥉 **3rd Place** | **$50** | Third-highest test accuracy |

**Total Prize Pool:** $350

### Additional Rewards

- 🎓 **Certificates:** All participants receive a certificate of completion
- 🍕 **Free Food:** Complimentary meals throughout the hackathon
- 🤝 **Networking:** Meet engineers from 3LC and AWS
- 📚 **Learning:** Gain hands-on experience with cutting-edge tools
- 💼 **Portfolio:** Build a project for your resume/portfolio
- 🎁 **Swag:** 3LC and AWS branded merchandise

### Eligibility Requirements

**⚠️ IMPORTANT:** To be eligible for cash prizes, you MUST:

1. ✅ Register a 3LC account at [https://3lc.ai](https://3lc.ai)
2. ✅ Submit via Google Form by the deadline
3. ✅ Include proof of scores (3LC table exports or screenshots)
4. ✅ Invite collaborators to your GitHub repo (@paulendresen, @rishikeshjadh)
5. ✅ Attend the closing ceremony (virtual option available)

**Account Registration:** Sign up for 3LC **before the submission deadline**. This is mandatory for prize eligibility.

### Judging Timeline

1. **Sunday, November 20th, 8 PM:** Submission deadline
2. **Sunday, 11 PM:** Initial scoring complete
3. **Monday morning:** Verification of top 3
4. **Monday afternoon:** Winners announced
5. **Within 1 week:** Prizes distributed

---

## 9. Evaluation & Submission

### GitHub Repository Creation

#### Step 1: Create Repository

```bash
# Initialize Git repo
git init
git add .
git commit -m "Initial commit - Hackathon starter code"

# Create GitHub repo (via GitHub website)
# Then push:
git remote add origin <your-repo-url>
git push -u origin main
```

#### Step 2: Organize Repository

**Required Structure:**
```
your-repo/
├── README.md              # Updated with your approach
├── WRITEUP.md            # Technical writeup (1-2 pages)
├── data/                 # (Optional - usually too large)
├── notebooks/
│   └── experiment.ipynb  # Your experiments
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
│   └── best_model.pth
├── results/
│   ├── metrics.csv
│   ├── plots/
│   └── 3lc_exports/      # Proof of scores
├── requirements.txt
└── .gitignore
```

#### Step 3: Invite Collaborators

1. Go to your GitHub repo
2. Settings → Collaborators
3. Add:
   - `@paulendresen`
   - `@rishikeshjadh`
4. They will review your submission

### Technical Writeup (1-2 Pages)

**Format:** PDF or Markdown in your repo

**Required Sections:**

#### 1. Problem Statement (1 paragraph)
- Brief description of the challenge
- Why it's difficult

#### 2. Approach (2-3 paragraphs)
- Model architecture used
- Training configuration
- Key decisions and rationale

#### 3. Data-Centric Insights (2-3 paragraphs)
- What did you learn from 3LC?
- Which samples were problematic?
- How did embeddings help?
- Patterns in errors

#### 4. Iterations (1-2 paragraphs per iteration)
- Describe each Train-Fix-Retrain cycle
- What changes you made
- Impact on accuracy
- 3LC table revisions

#### 5. Results (1 paragraph + table)
- Final accuracy
- Comparison to baseline
- Key metrics

**Example Table:**
| Revision | Changes | Validation Accuracy | Test Accuracy |
|----------|---------|---------------------|---------------|
| v1 | Baseline | 83.0% | TBD |
| v2 | Removed 10 ambiguous samples | 85.2% | TBD |
| v3 | Fixed 3 label errors | 86.8% | TBD |
| v4 | Added augmentation | 88.1% | TBD |

#### 6. Challenges (1 paragraph)
- What was hard?
- What didn't work?
- Lessons learned

#### 7. Future Work (1 paragraph)
- If you had more time, what would you try?
- Ideas for improvement

**Writing Tips:**
- Be concise and clear
- Include visuals (plots, screenshots from 3LC)
- Show your data-centric thinking
- Explain *why* not just *what*

### Proof of Scores

**Required Evidence:**

1. **3LC Table Exports**
   - Export your final table revision
   - Include as zip file in `results/3lc_exports/`
   
2. **Screenshots**
   - 3LC Dashboard showing metrics
   - Embeddings visualization
   - Table revision comparison
   - Include in `results/screenshots/`

3. **Metrics File**
   - CSV with per-epoch results
   - Final validation accuracy
   - `results/metrics.csv`

### Google Form Submission

**Form Fields (link provided at event):**

1. **Name / Team Name**
2. **Email**
3. **3LC Username** (verify eligibility)
4. **GitHub Repository URL**
5. **Final Test Accuracy** (on validation set)
6. **Brief Summary** (100 words)
7. **Confirmation:** "I have invited @paulendresen and @rishikeshjadh as collaborators"

**Submission Deadline:** Sunday, November 20th, 8:00 PM CST

**Late Submissions:** Not accepted (hard deadline for fair evaluation)

### Evaluation Rubric

**Total: 100 points**

| Category | Points | Criteria |
|----------|--------|----------|
| **Test Accuracy** | 60 | Performance on held-out test set (relative to baseline) |
| **Data-Centric Approach** | 20 | Evidence of 3LC usage, iterations, data improvements |
| **Code Quality** | 10 | Clean, documented, reproducible, follows best practices |
| **Writeup Quality** | 10 | Clear explanation, insights, visuals, completeness |

**Test Accuracy Scoring:**
- 60 pts: >90% accuracy
- 50 pts: 88-90% accuracy
- 40 pts: 86-88% accuracy
- 30 pts: 84-86% accuracy
- 20 pts: 82-84% accuracy
- 10 pts: <82% accuracy

**Data-Centric Approach Scoring:**
- 20 pts: Multiple iterations, clear 3LC usage, documented insights
- 15 pts: 2 iterations, 3LC table revisions, some insights
- 10 pts: 1 iteration, basic 3LC usage
- 5 pts: Minimal 3LC usage
- 0 pts: No evidence of data-centric approach

---

## 10. Support & Resources

### During the Hackathon

**On-Site Support:**
- Tech support desk at JSOM 1.118
- Roaming mentors from 3LC and AWS

**Office Hours:**
- Saturday: 2-4 PM, 8-10 PM
- Sunday: 10 AM-12 PM, 4-6 PM

**Communication Channels:**
- Slack: #hackathon-support
- Discord: Link provided at event
- Email: [support email to be provided]

### Documentation Resources

**3LC Resources:**
- Official Docs: [https://docs.3lc.ai](https://docs.3lc.ai)
- Tutorials: [https://3lc.ai/tutorials](https://3lc.ai/tutorials)
- API Reference: [https://docs.3lc.ai/api](https://docs.3lc.ai/api)

**PyTorch Resources:**
- Official Tutorials: [https://pytorch.org/tutorials](https://pytorch.org/tutorials)
- Image Classification Guide: [https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

**AWS Resources:**
- Free Tier: [https://aws.amazon.com/free](https://aws.amazon.com/free)
- S3 Documentation: [https://docs.aws.amazon.com/s3/](https://docs.aws.amazon.com/s3/)

### Frequently Asked Questions

**Q: Can I use pretrained models (transfer learning)?**  
A: Yes! Transfer learning is encouraged.

**Q: Can I work in teams?**  
A: Solo or teams of 2-3 are allowed. Specify in submission.

**Q: Can I use external datasets?**  
A: No, you must use only the provided dataset.

**Q: What if I can't install 3LC?**  
A: Contact support immediately. 3LC usage is required for prizes.

**Q: Can I use other tools (W&B, TensorBoard)?**  
A: Yes, but 3LC must be your primary analysis tool.

**Q: How is the test set evaluated?**  
A: Organizers will run your code on the withheld test set.

**Q: What if my code doesn't run?**  
A: Ensure clear instructions in README. Organizers will contact for clarification.

**Q: Can I submit multiple times?**  
A: Only final submission counts. Update your repo, but submit form once.

---

## 11. Schedule

### Saturday, November 18th

| Time | Activity |
|------|----------|
| 3:00 PM | Opening Ceremony & Kickoff |
| 3:30 PM | Dataset Release & Setup Support |
| 4:00 PM | Coding Begins |
| 6:00 PM | Dinner Provided |
| 8:00 PM | Office Hours |
| 10:00 PM | Venue Closes |

### Sunday, November 19th

| Time | Activity |
|------|----------|
| 9:00 AM | Venue Opens |
| 10:00 AM | Office Hours |
| 12:00 PM | Lunch Provided |
| 4:00 PM | Final Office Hours |
| 6:00 PM | Dinner Provided |
| 8:00 PM | **Submission Deadline** |
| 9:00 PM | Closing Ceremony (Optional) |

---

## 12. Contact Information

**Event Organizers:**

**Paul Endresen**  
- Role: 3LC Engineer  
- GitHub: @paulendresen  
- Email: [to be provided]

**Rishikesh Jadhav**  
- Role: AWS Cloud Club at UT Dallas  
- GitHub: @rishikeshjadh  
- Email: [to be provided]

**AWS Cloud Club at UT Dallas**  
- Website: [to be provided]  
- Social Media: [to be provided]

**3LC (Three Lines of Code)**  
- Website: [https://3lc.ai](https://3lc.ai)  
- Documentation: [https://docs.3lc.ai](https://docs.3lc.ai)

---

## 13. Code of Conduct

All participants must adhere to the following:

- ✅ Be respectful and inclusive
- ✅ Collaborate and help others
- ✅ Ask questions when stuck
- ✅ Submit original work (your own code)
- ✅ Give credit for external resources
- ❌ No plagiarism or copying code
- ❌ No sharing solutions before deadline
- ❌ No harassment or discrimination

Violations may result in disqualification.

---

## 14. Acknowledgments

**Sponsors:**
- **3LC (Three Lines of Code)** – Data-Centric AI Platform
- **AWS Cloud Club at UT Dallas** – Event hosting and organization

**Special Thanks:**
- UT Dallas for venue support
- All volunteers and mentors
- All participants for joining!

---

## Good luck, and happy hacking! 🚀🐶🧁

**Remember:** Focus on data quality, not just model complexity. The best improvements often come from better data, not bigger models!

