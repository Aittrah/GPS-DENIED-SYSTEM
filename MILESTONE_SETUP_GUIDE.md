# Step-by-Step Guide: Creating "Phase 12-14: Tools and Docs" Milestone

This guide will walk you through manually creating the milestone and assigning issues using GitHub's web interface.

## Step 1: Create the Milestone

1. Go to your repository: https://github.com/Aittrah/GPS-DENIED-SYSTEM
2. Click on the **"Milestones"** tab in the navigation
3. Click the **"New milestone"** button (or go directly to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/milestones/new)

## Step 2: Fill in Milestone Details

On the "New milestone" page, fill in:

**Title:** 
```
Phase 12-14: Tools and Docs
```

**Due date (optional):**
- Leave empty or set a target date if desired

**Description:**
```
Database tools, integration tests, and documentation

Includes:
- CLI tools (image collection, batch indexing, database inspection)
- Integration testing (simulation tests, end-to-end mission test)
- Documentation (hardware deployment guide, database creation guide)
```

## Step 3: Save the Milestone

Click the **"Create milestone"** button

You should see a success message. Note the milestone number (e.g., "Phase 12-14: Tools and Docs #3")

## Step 4: Assign Issues to the Milestone

Now you need to add these 7 issues to the milestone:

### Issue #21: Implement image collection tool
1. Go to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/issues/21
2. On the right sidebar, find **"Milestone"** section
3. Click on it and select **"Phase 12-14: Tools and Docs"**

### Issue #22: Implement batch indexing tool
1. Go to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/issues/22
2. Assign to **"Phase 12-14: Tools and Docs"** milestone

### Issue #23: Implement database inspection tool
1. Go to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/issues/23
2. Assign to **"Phase 12-14: Tools and Docs"** milestone

### Issue #24: Implement simulation integration tests
1. Go to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/issues/24
2. Assign to **"Phase 12-14: Tools and Docs"** milestone

### Issue #25: Implement end-to-end mission test
1. Go to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/issues/25
2. Assign to **"Phase 12-14: Tools and Docs"** milestone

### Issue #26: Write hardware deployment guide
1. Go to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/issues/26
2. Assign to **"Phase 12-14: Tools and Docs"** milestone

### Issue #27: Write reference database creation guide
1. Go to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/issues/27
2. Assign to **"Phase 12-14: Tools and Docs"** milestone

## Step 5: Verify

1. Go back to: https://github.com/Aittrah/GPS-DENIED-SYSTEM/milestones
2. Click on **"Phase 12-14: Tools and Docs"**
3. Confirm all 7 issues are listed under the milestone

## Alternative: Use the Script (Faster)

If you prefer automation, use the Python script instead:

```bash
# Install requests if needed
pip install requests

# Get your GitHub token from: https://github.com/settings/tokens?type=beta

# Run the script
python scripts/setup_milestone.py <your-github-token>
```

The script will do all of this automatically in seconds!

---

**Done!** Your milestone is now set up with all issues assigned.
