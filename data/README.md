# Dataset Directory

This folder contains the Chihuahua vs Muffin dataset for the hackathon.

## Structure

```
data/
├── train/
│   ├── chihuahua/    (Place training chihuahua images here)
│   └── muffin/       (Place training muffin images here)
└── test/             (Place test images here)
```

## Getting the Dataset

**Option 1: Download from S3** (Recommended)
- Download link will be provided at the hackathon kickoff
- Extract the zip file into this `data/` folder
- Verify the folder structure matches above

**Option 2: Manual Setup**
If you have the dataset elsewhere:
1. Place 100 chihuahua images in `train/chihuahua/`
2. Place 100 muffin images in `train/muffin/`
3. Place test images in `test/` folder

## Verification

After downloading, verify your dataset:
```bash
python src/register_tables.py
```

You should see:
- Training images: 200 (100 per class)
- Classes are well balanced
- Test images: (count will vary)

## Note

The dataset is **not included in this repository** due to size constraints. 
You must download it separately using the link provided by the organizers.

