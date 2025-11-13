# Quick Setup Instructions

## For Hackathon Participants

### Step 1: Download this Package
Download and extract the hackathon package to your computer.

### Step 2: Install Python Dependencies
```bash
cd Hackathon_Image_Classification_Challenge
pip install -r requirements.txt
```

### Step 3: Download Dataset
1. Get the S3 link from the hackathon organizers
2. Download the dataset zip file
3. Extract to the `data/` folder
4. Verify structure:
   ```
   data/
   ├── train/
   │   ├── chihuahua/
   │   └── muffin/
   └── test/
   ```

### Step 4: Install and Setup 3LC
```bash
pip install tlc
```

Then sign up at [https://3lc.ai](https://3lc.ai) to get your account.

### Step 5: Test Your Setup
```bash
# Test dataset registration
python src/register_tables.py

# Or open the starter notebook
jupyter notebook notebooks/starter_notebook.ipynb
```

### Step 6: Start Coding!
- Read the `README.md` for full instructions
- Check `HACKATHON_OVERVIEW_DOCUMENT.md` for detailed guidelines
- Run the baseline model
- Use 3LC to analyze and improve
- Submit your results!

---

## Common Issues

**Issue: "No module named torch"**  
Solution: `pip install torch torchvision`

**Issue: "data/train not found"**  
Solution: Download dataset and extract to correct location

**Issue: "3LC authentication failed"**  
Solution: Sign up at 3lc.ai and follow setup instructions

**Issue: "CUDA out of memory"**  
Solution: Reduce batch size in hyperparameters

---

## Need Help?
- Check the `README.md`
- Visit #hackathon-support on Slack
- Ask mentors at the venue
- Review 3LC documentation: https://docs.3lc.ai

Good luck! 🚀

