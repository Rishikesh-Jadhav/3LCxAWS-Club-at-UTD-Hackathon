# 🧠 3LC x AWS Cloud @ UT Dallas Hackathon

## Image Classification Challenge – *"Chihuahua or Muffin?"*

---

### 📘 Overview

Welcome to the **Data Centric AI Challenge** presented by **3LC** (Three Lines of Code) and the **AWS Cloud Club at UT Dallas**! 

In this hands-on 48-hour hackathon, you'll tackle one of the most entertaining challenges in computer vision: **distinguishing chihuahuas from muffins**. This seemingly simple task has famously stumped even advanced image classification models due to the striking visual similarities between the two classes.

Your mission is to build, train, and optimize an image classification model that can accurately differentiate between these two classes. This challenge will test your skills in:
- Data exploration and preprocessing
- Model architecture design
- Training optimization
- Data-centric AI techniques using 3LC

**When:** November 18th, 2025 – Kickoff at 3PM  
**Where:** JSOM 1.118  
**Duration:** 48 hours  
**Prizes:** 🥇 $200 | 🥈 $100 | 🥉 $50 + Certificates for all participants

---

### 🧩 Description

This challenge focuses on **data-centric AI** principles. Rather than just building bigger models, you'll learn to:
1. **Analyze your dataset** using 3LC's visualization tools
2. **Identify problematic samples** that confuse your model
3. **Iterate intelligently** on your training data
4. **Track improvements** through the Train-Fix-Retrain loop

#### Challenge Steps:
1. **Download the dataset** from the S3 link provided below
2. **Set up your environment** with the provided starter code
3. **Register your dataset** with 3LC Tables to enable advanced analysis
4. **Train your baseline model** (starter code achieves ~83% accuracy)
5. **Use 3LC Dashboard** to explore embeddings, identify edge cases, and refine your approach
6. **Iterate and improve** – the Train-Fix-Retrain loop is key!
7. **Submit your results** via GitHub repository with proof of scores

---

### 📦 Dataset Description

**Dataset Name:** Chihuahua vs Muffin Dataset

**Structure:**
```
data/
├── train/
│   ├── chihuahua/    (100 images)
│   └── muffin/       (100 images)
└── test/             (held-out test set)
```

**Specifications:**
- **Image Size:** 128x128 pixels (RGB)
- **Training Set:** 200 images total (100 per class)
- **Test Set:** Separate evaluation folder (labels withheld)
- **Classes:** 
  - `chihuahua` – Images of chihuahua dogs
  - `muffin` – Images of muffins (often resembling chihuahuas!)
- **Format:** JPEG/PNG images

#### 👉 Dataset Download (S3 Link)

**Public S3 Bucket:** [Dataset Download Link - To Be Announced]

*Note: The dataset link will be provided at kickoff. All participants will download from the same public AWS S3 bucket.*

---

### 🧠 Evaluation

**Evaluation Metric:** Classification Accuracy on held-out test set

**Submission Requirements:**
- GitHub repository with your code
- 1-2 page technical writeup (PDF/Markdown)
- Proof of scores (zipped 3LC tables or screenshots)
- Report submission via Google Form (link to be shared)

**What Gets Evaluated:**
- Final test accuracy on hidden labels
- Code quality and documentation
- Explanation of your data-centric approach
- Creative use of 3LC tools and insights

*Detailed evaluation rubric and test set access will be announced during the hackathon.*

---

### 🛠️ Setup Instructions

#### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Git for version control
- 3LC account (free signup at [3lc.ai](https://3lc.ai))

#### Installation

1. **Clone or Download this repository:**
```bash
git clone <your-hackathon-repo>
cd Hackathon_Image_Classification_Challenge
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset:**
   - Download from the S3 link provided
   - Extract to the `data/` folder
   - Ensure folder structure matches the layout above

4. **Register with 3LC:**
   - Sign up at [3lc.ai](https://3lc.ai)
   - Follow the setup guide in `notebooks/starter_notebook.ipynb`

---

### 📊 New Concepts: 3LC Tables and Datasets

#### What is 3LC?

**3LC (Three Lines of Code)** is a data-centric AI platform that helps you:
- **Visualize your dataset** in rich, interactive dashboards
- **Explore embeddings** to understand data distribution
- **Identify problematic samples** that hurt model performance
- **Track table revisions** across training iterations

#### Key Concepts:

**1. Tables**
- A 3LC Table is an enhanced dataset with metadata, embeddings, and quality metrics
- Tables can be versioned (revisions) to track changes over time

**2. Datasets**
- Collections of data registered with 3LC for analysis
- Supports images, text, and other modalities

**3. Embeddings**
- High-dimensional representations of your data (from CNN layers)
- 3LC visualizes these in 2D/3D for exploration
- Use embeddings to find clusters, outliers, and confusing samples

**Analogy:**
> Think of 3LC Tables like **Git for your datasets**. Just as Git tracks code changes, 3LC tracks dataset revisions, quality metrics, and model insights across experiments.

---

### 🔄 Train-Fix-Retrain Loop Instructions

This is the core workflow for data-centric AI:

#### Step 1: Train (Baseline)
- Run `src/train.py` or use the starter notebook
- Train your model on the full training set
- Achieve ~83% baseline accuracy

#### Step 2: Analyze (3LC Dashboard)
- Open 3LC Dashboard and load your table
- **Explore embeddings:** Look for misclassified samples
- **Check table revisions:** Compare metrics across runs
- **Identify patterns:** Are muffins with dark backgrounds confused with chihuahuas?

#### Step 3: Fix (Data Centric Improvements)
Options include:
- **Remove outliers:** Delete truly ambiguous samples
- **Augment data:** Add rotations, flips, color jitter
- **Rebalance classes:** Adjust for any imbalance
- **Relabel errors:** Fix incorrect labels

#### Step 4: Retrain
- Update your training script with fixes
- Retrain the model on the improved dataset
- Compare results in 3LC – did accuracy improve?

#### Step 5: Iterate
- Repeat the loop 2-3 times
- Track your revisions in 3LC
- Document what worked in your writeup

**Dashboard Mechanics:**
- **Table Revisions:** Each iteration creates a new revision
- **Embeddings View:** Scatter plots show data distribution
- **Metrics Panel:** Track accuracy, loss, and custom metrics
- **Sample Inspector:** Drill down into individual images

---

### 🏆 Incentives

**Cash Prizes:**
- 🥇 **1st Place:** $200
- 🥈 **2nd Place:** $100
- 🥉 **3rd Place:** $50

**Additional Perks:**
- 🎓 **Certificates** for all participants
- 🍕 **Free food** throughout the hackathon
- 🤝 **Networking** with AWS and 3LC engineers
- 📚 **Learning** cutting-edge data-centric AI techniques

**⚠️ Important:**
> **You MUST register a 3LC account to be eligible for prizes.** All participants must sign up at [3lc.ai](https://3lc.ai) before the submission deadline.

---

### 🚀 Getting Started

#### Quick Start (5 minutes)

1. **Install requirements:**
```bash
pip install -r requirements.txt
```

2. **Open the starter notebook:**
```bash
jupyter notebook notebooks/starter_notebook.ipynb
```

3. **Run all cells** to:
   - Load the dataset
   - Train a baseline CNN
   - Achieve ~83% accuracy

4. **Explore 3LC Dashboard** and start optimizing!

#### Alternative: Python Scripts

If you prefer scripts over notebooks:

```bash
# Register your dataset with 3LC
python src/register_tables.py

# Train the baseline model
python src/train.py
```

---

### 📤 Submission Instructions

#### Deliverables:

1. **GitHub Repository** containing:
   - All your code (models, training scripts, notebooks)
   - 1-2 page technical writeup (PDF or Markdown)
   - Proof of scores (screenshots or zipped 3LC table exports)
   - Updated README with your approach

2. **Google Form Submission** (link to be provided):
   - Your final test accuracy score
   - GitHub repository link
   - 3LC username

3. **GitHub Collaborators:**
   - Invite **@paulendresen** and **@rishikeshjadh** as collaborators to your repo

#### Writeup Guidelines:

Your 1-2 page writeup should include:
- **Problem statement** (brief)
- **Approach:** What model architecture did you use?
- **Data-centric insights:** What did you learn from 3LC?
- **Iterations:** Describe your Train-Fix-Retrain cycles
- **Results:** Final accuracy and key metrics
- **Challenges:** What was hard? What didn't work?

**Format:** PDF or Markdown (in your repo as `WRITEUP.md`)

---

### 📚 Resources

- **3LC Documentation:** [docs.3lc.ai](https://docs.3lc.ai)
- **PyTorch Tutorials:** [pytorch.org/tutorials](https://pytorch.org/tutorials)
- **Image Classification Guide:** [PyTorch Image Classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- **AWS Free Tier:** [aws.amazon.com/free](https://aws.amazon.com/free)

---

### 👨‍💻 Credits

**Challenge Design and Baseline Model:**
- **Paul Endresen** – 3LC Engineer
- **Rishikesh Jadhav** – AWS Cloud Club at UT Dallas

**Organizers:**
- **3LC (Three Lines of Code)**
- **AWS Cloud Club at UT Dallas**

---

### 📞 Contact & Support

**During the Hackathon:**
- Ask questions in the hackathon Slack/Discord channel
- Office hours: TBD
- Tech support available on-site

**Questions?**
- Email: [Insert contact email]
- Slack: #hackathon-support

---

### 📜 License

This starter code is provided as-is for educational purposes. Dataset usage is restricted to this hackathon challenge.

---

## Good luck, and may the best data scientist win! 🚀🐶🧁

