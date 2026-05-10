#!/usr/bin/env python3
"""
Script to create and assign issues to the "Phase 12-14: Tools and Docs" milestone.

This script:
1. Creates a new milestone named "Phase 12-14: Tools and Docs"
2. Assigns the following issues to this milestone:
   - #21: Implement image collection tool
   - #22: Implement batch indexing tool
   - #23: Implement database inspection tool
   - #24: Implement simulation integration tests
   - #25: Implement end-to-end mission test
   - #26: Write hardware deployment guide
   - #27: Write reference database creation guide

Usage:
    python scripts/setup_milestone.py <github_token>

Environment:
    Requires GITHUB_TOKEN environment variable or pass as argument
"""

import os
import sys
import requests
from typing import Optional

# Configuration
REPO_OWNER = "Aittrah"
REPO_NAME = "GPS-DENIED-SYSTEM"
MILESTONE_TITLE = "Phase 12-14: Tools and Docs"
MILESTONE_DESCRIPTION = "Database tools, integration tests, and documentation"
ISSUES_TO_ADD = [21, 22, 23, 24, 25, 26, 27]

# GitHub API endpoint
API_BASE = "https://api.github.com"


def get_github_token() -> str:
    """Get GitHub token from environment or command line."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not provided")
        print("Usage: python scripts/setup_milestone.py <github_token>")
        print("Or set GITHUB_TOKEN environment variable")
        sys.exit(1)
    return token


def make_request(method: str, endpoint: str, token: str, data: Optional[dict] = None) -> dict:
    """Make a request to GitHub API."""
    url = f"{API_BASE}{endpoint}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, headers=headers, json=data)
    elif method == "PATCH":
        response = requests.patch(url, headers=headers, json=data)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    response.raise_for_status()
    return response.json()


def create_milestone(token: str) -> dict:
    """Create the milestone."""
    print(f"Creating milestone '{MILESTONE_TITLE}'...")
    
    endpoint = f"/repos/{REPO_OWNER}/{REPO_NAME}/milestones"
    data = {
        "title": MILESTONE_TITLE,
        "description": MILESTONE_DESCRIPTION,
        "state": "open"
    }
    
    milestone = make_request("POST", endpoint, token, data)
    print(f"✓ Milestone created (ID: {milestone['number']})")
    return milestone


def assign_issue_to_milestone(issue_number: int, milestone_number: int, token: str) -> None:
    """Assign an issue to a milestone."""
    endpoint = f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_number}"
    data = {"milestone": milestone_number}
    
    make_request("PATCH", endpoint, token, data)
    print(f"✓ Issue #{issue_number} assigned to milestone")


def get_milestone_by_title(title: str, token: str) -> Optional[dict]:
    """Get milestone by title."""
    endpoint = f"/repos/{REPO_OWNER}/{REPO_NAME}/milestones"
    milestones = make_request("GET", endpoint, token)
    
    for milestone in milestones:
        if milestone["title"] == title:
            return milestone
    
    return None


def main():
    """Main function."""
    token = get_github_token()
    
    try:
        # Check if milestone already exists
        existing_milestone = get_milestone_by_title(MILESTONE_TITLE, token)
        
        if existing_milestone:
            print(f"Milestone '{MILESTONE_TITLE}' already exists (ID: {existing_milestone['number']})")
            milestone = existing_milestone
        else:
            milestone = create_milestone(token)
        
        # Assign issues to milestone
        print(f"\nAssigning {len(ISSUES_TO_ADD)} issues to milestone...")
        for issue_number in ISSUES_TO_ADD:
            assign_issue_to_milestone(issue_number, milestone["number"], token)
        
        print(f"\n✓ All done! Milestone '{MILESTONE_TITLE}' now contains {len(ISSUES_TO_ADD)} issues")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
