import os
import re
import subprocess
import requests
import yaml
import logging
from time import sleep
from functools import wraps
from typing import Tuple, Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# ============================================================
# CONFIGURATION
# ============================================================
REPO_PATH = os.getenv("REPO_PATH")
YAML_DIR = os.getenv("YAML_DIR")
GITLAB_URL = os.getenv("GITLAB_URL")
PROJECT_ID = os.getenv("PROJECT_ID")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
GIT_COMMITTER_EMAIL = os.getenv("GIT_COMMITTER_EMAIL")
GIT_COMMITTER_NAME  = os.getenv("GIT_COMMITTER_NAME")
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL") 
REVIEWER_USERNAMES = [
    u.strip() for u in os.getenv("REVIEWER_USERNAMES", "").split(",") if u.strip()
]
# ============================================================
# LOGGING CONFIGURATION
# ============================================================
def setup_logging():
    """Configure logging with both file and console handlers"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=log_date_format,
        handlers=[
            logging.FileHandler(f'{log_dir}/onboarding.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================
# CONFIGURATION VALIDATION
# ============================================================
def validate_config() -> None:
    """
    Validate all required environment variables and paths are properly configured.
    
    Raises:
        Exception: If any required configuration is missing or invalid
    """
    required_vars = [
        "REPO_PATH", "YAML_DIR", "GITLAB_URL", 
        "PROJECT_ID", "GITLAB_TOKEN", "GIT_COMMITTER_EMAIL"
    ]
    
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise Exception(f"Missing required env vars: {', '.join(missing)}")
    
    # Validate paths exist
    repo_path = os.getenv("REPO_PATH")
    yaml_dir = os.getenv("YAML_DIR")
    
    if not os.path.exists(repo_path):
        raise Exception(f"REPO_PATH does not exist: {repo_path}")
    
    if not os.path.exists(yaml_dir):
        raise Exception(f"YAML_DIR does not exist: {yaml_dir}")
    
    logger.info("✅ Configuration validation passed")
    logger.info(f"REPO_PATH: {repo_path}")
    logger.info(f"YAML_DIR: {yaml_dir}")
    logger.info(f"GITLAB_URL: {os.getenv('GITLAB_URL')}")

# ============================================================
# RETRY DECORATOR FOR API CALLS
# ============================================================
def retry_api_call(max_attempts: int = 3, delay: int = 1):
    """
    Decorator to retry failed API calls with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds (doubles each retry)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    sleep(wait_time)
            return None
        return wrapper
    return decorator

# ============================================================
# GIT OPERATIONS VALIDATION
# ============================================================
def validate_git_repo(repo_path: str) -> None:
    """
    Ensure the repository path is a valid git repository and is in a clean state.
    
    Args:
        repo_path: Path to the git repository
        
    Raises:
        Exception: If the path is not a valid git repository or has critical issues
    """
    if not os.path.exists(os.path.join(repo_path, ".git")):
        raise Exception(f"{repo_path} is not a git repository")
    
    os.chdir(repo_path)
    
    # Check for merge conflicts
    status_output = subprocess.run(
        "git status --porcelain", 
        shell=True, 
        capture_output=True, 
        text=True
    ).stdout
    
    conflict_patterns = ["UU", "AA", "DD", "AU", "UA", "DU", "UD"]
    has_conflicts = any(
        line[:2] in conflict_patterns 
        for line in status_output.splitlines() 
        if line
    )
    
    if has_conflicts:
        raise Exception("Repository has merge conflicts that need manual resolution")
    
    logger.info("✅ Git repository validation passed")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def run_command(cmd: str, capture_output: bool = True) -> str:
    """
    Run a shell command and return its output.
    
    Args:
        cmd: Shell command to execute
        capture_output: Whether to capture and return stdout
        
    Returns:
        Command output as string if capture_output is True
        
    Raises:
        Exception: If command returns non-zero exit code
    """
    logger.debug(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = f"Command failed: {cmd}\nError: {result.stderr}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    if capture_output:
        return result.stdout.strip()
    return ""

def configure_git_identity() -> None:
    """
    Set git committer email and name locally in .git/config.
    
    Raises:
        Exception: If GIT_COMMITTER_EMAIL is not set
    """
    git_committer_email = os.getenv("GIT_COMMITTER_EMAIL")
    git_committer_name = os.getenv("GIT_COMMITTER_NAME")
    
    if not git_committer_email:
        raise Exception("❌ GIT_COMMITTER_EMAIL not set in .env")

    run_command(f'git config user.email "{git_committer_email}"')

    if git_committer_name:
        run_command(f'git config user.name "{git_committer_name}"')

    logger.info(f"🔧 Git identity set → {git_committer_name} <{git_committer_email}>")

def cleanup_repo_state() -> None:
    """
    Clean up any dirty or conflicted git state before branching operations.
    """
    status = run_command("git status --porcelain", capture_output=True)
    
    conflict_patterns = ["UU", "AA", "DD", "AU", "UA", "DU", "UD"]
    has_conflicts = any(
        line[:2] in conflict_patterns 
        for line in status.splitlines() 
        if line
    )

    if has_conflicts or status.strip():
        logger.warning("Dirty/conflicted index detected — resetting...")
        run_command("git reset HEAD", capture_output=False)
        run_command("git checkout -- .", capture_output=False)

    stash_list = run_command("git stash list", capture_output=True)

    if stash_list:
        logger.info(f"🗑️ Dropping leftover stash: {stash_list.splitlines()[0]}")
        run_command("git stash drop", capture_output=False)

def check_branch_exists(branch_name: str) -> bool:
    """
    Check if a branch exists locally or remotely.
    
    Args:
        branch_name: Name of the branch to check
        
    Returns:
        True if branch exists locally or remotely, False otherwise
    """
    # Check local branches
    local_branches = run_command(f"git branch --list {branch_name}", capture_output=True)
    if local_branches:
        return True
    
    # Check remote branches
    remote_branches = run_command(
        f"git ls-remote --heads origin {branch_name}", 
        capture_output=True
    )
    return bool(remote_branches)

# ============================================================
# CORE FUNCTIONALITY
# ============================================================
@retry_api_call(max_attempts=3, delay=1)
def get_user_id(username: str) -> int:
    """
    Get GitLab user ID from username.
    
    Args:
        username: GitLab username
        
    Returns:
        GitLab user ID
        
    Raises:
        Exception: If user not found or API call fails
    """
    gitlab_url = os.getenv("GITLAB_URL")
    gitlab_token = os.getenv("GITLAB_TOKEN")
    
    url = f"{gitlab_url}/users?username={username}"
    headers = {"PRIVATE-TOKEN": gitlab_token}

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200 or not response.json():
        raise Exception(f"User not found: {username}")

    user_id = response.json()[0]["id"]
    logger.debug(f"Found user {username} with ID {user_id}")
    return user_id

def parse_file(file_path: str) -> Tuple[str, str, str]:
    """
    Extract ticket number, environment, and project name from YAML file.
    
    Expected filename format: {TICKET}_{ENVIRONMENT}_{PROJECT}.yaml
    Example: CSDCS-17521_dev_avst-01.yaml
    
    The function first attempts to read from YAML content. If fields are not 
    present in YAML, it falls back to parsing from filename.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Tuple of (ticket_number, environment, project_name)
        
    Raises:
        ValueError: If filename format is invalid and YAML has no required fields
    """
    filename = os.path.basename(file_path)
    
    # Try to read from YAML first
    ticket = None
    env = None
    project = None
    
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        
        if data:
            ticket = data.get("ticket") or data.get("jira_ticket")
            env = data.get("environment")
            project = data.get("project") or data.get("project_name")
            
        if ticket and env and project:
            logger.info(f"Using data from YAML content: {ticket}, {env}, {project}")
            return ticket, env.lower(), project
            
    except Exception as e:
        logger.debug(f"Could not parse YAML from {filename}: {e}")

    # Fallback to filename parsing
    # Pattern: TICKET_ENVIRONMENT_PROJECT.yaml
    pattern = r"^([A-Z]+-\d+)_([a-zA-Z]+)_(.+)\.yaml$"
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            f"Expected format: TICKET_ENVIRONMENT_PROJECT.yaml"
        )
    
    ticket = match.group(1)
    env = match.group(2)
    project = match.group(3)
    
    logger.info(f"Parsed from filename: Ticket={ticket}, Environment={env}, Project={project}")
    return ticket, env.lower(), project

def create_branch_and_commit(file_path: str, branch_name: str, target_branch: str, 
                             project_name: str) -> None:
    """
    Create a git branch, copy YAML file, commit, and push to remote.
    
    Args:
        file_path: Path to the YAML file to onboard
        branch_name: Name of the branch to create
        target_branch: Target branch to branch from (usually environment name)
        project_name: Name of the project being onboarded
        
    Raises:
        Exception: If branch already exists or any git operation fails
    """
    repo_path = os.getenv("REPO_PATH")
    os.chdir(repo_path)

    logger.info(f"🚀 Working on branch: {branch_name} for project: {project_name}")

    # Check for branch existence to avoid race conditions
    if check_branch_exists(branch_name):
        raise Exception(f"Branch {branch_name} already exists! Skipping to avoid conflicts.")

    # Configure git identity and cleanup
    configure_git_identity()
    cleanup_repo_state()

    # Stash any local changes before switching branches
    stash_result = subprocess.run(
        "git stash", shell=True, capture_output=True, text=True
    )
    stashed = "Saved working directory" in stash_result.stdout
    if stashed:
        logger.info(f"📦 Stashed local changes: {stash_result.stdout.strip()}")

    original_branch = None
    try:
        # Save current branch for rollback
        original_branch = run_command("git branch --show-current", capture_output=True)
        
        # Switch to target branch and update
        run_command(f"git checkout {target_branch}", capture_output=False)
        run_command("git pull", capture_output=False)

        # Create new branch
        run_command(f"git checkout -b {branch_name}", capture_output=False)
        logger.info(f"✅ Created branch: {branch_name}")

        # Copy YAML into repo
        filename = os.path.basename(file_path)
        run_command(f"cp {file_path} {repo_path}/{filename}", capture_output=False)
        
        # Parse for commit message
        ticket, env, project = parse_file(file_path)
        commit_msg = f"{ticket}: onboard {project} to {env} environment"

        # Commit and push
        run_command(f"git add {filename}", capture_output=False)
        run_command(f'git commit -m "{commit_msg}"', capture_output=False)
        run_command(f"git push origin {branch_name}", capture_output=False)

        logger.info(f"✅ Successfully pushed branch: {branch_name}")

    except Exception as e:
        logger.error(f"Failed during branch operations: {e}")
        # Rollback: delete local branch if it was created
        if branch_name and check_branch_exists(branch_name):
            try:
                run_command(f"git checkout {original_branch or target_branch}", capture_output=False)
                run_command(f"git branch -D {branch_name}", capture_output=False)
                logger.info(f"🔄 Rolled back: deleted branch {branch_name}")
            except Exception as rollback_error:
                logger.warning(f"Rollback failed: {rollback_error}")
        raise

    finally:
        # Restore stashed changes if any
        if stashed:
            try:
                run_command(f"git checkout {target_branch}", capture_output=False)
                pop_result = subprocess.run(
                    "git stash pop", shell=True, capture_output=True, text=True
                )
                if pop_result.returncode == 0:
                    logger.info("📦 Restored stashed changes")
                else:
                    logger.warning(f"Stash pop failed: {pop_result.stderr.strip()}")
            except Exception as e:
                logger.warning(f"Could not restore stash: {e}")

@retry_api_call(max_attempts=3, delay=1)
def create_merge_request(branch_name: str, target_branch: str, ticket: str, 
                         project_name: str) -> Dict[str, Any]:
    """
    Create a merge request in GitLab.
    
    Args:
        branch_name: Source branch name
        target_branch: Target branch name
        ticket: Jira ticket number
        project_name: Name of the project being onboarded
        
    Returns:
        Dictionary containing merge request data from GitLab API
        
    Raises:
        Exception: If MR creation fails
    """
    gitlab_url = os.getenv("GITLAB_URL")
    project_id = os.getenv("PROJECT_ID")
    gitlab_token = os.getenv("GITLAB_TOKEN")
    jira_base_url = os.getenv("JIRA_BASE_URL")
    
    # Get reviewer usernames from env
    reviewer_usernames_raw = os.getenv("REVIEWER_USERNAMES", "")
    reviewer_usernames = [u.strip() for u in reviewer_usernames_raw.split(",") if u.strip()]
    
    url = f"{gitlab_url}/projects/{project_id}/merge_requests"

    logger.info(f"👥 Reviewers: {reviewer_usernames}")

    # Convert usernames to GitLab user IDs
    reviewer_ids = []
    for username in reviewer_usernames:
        try:
            user_id = get_user_id(username)
            if user_id:
                reviewer_ids.append(user_id)
                logger.debug(f"Found reviewer: {username} (ID: {user_id})")
        except Exception as e:
            logger.warning(f"Could not find GitLab user: {username} - {e}")

    # Create mentions for MR description
    mentions = " ".join([f"@{u}" for u in reviewer_usernames])

    # Jira link handling
    jira_link = f"{jira_base_url}/{ticket}" if jira_base_url else ticket

    mr_labels = ["project-onboarding", "auto-generated", "SRE Autonomous AI Agent"]

    # Prepare payload with project name in title and description
    mr_title = f"{ticket}: Onboard {project_name} to {target_branch}"
    
    mr_description = f"""
## 📋 Project Onboarding Request

**Project:** `{project_name}`
**Environment:** `{target_branch}`
**Jira Ticket:** {jira_link}

### Details
This merge request was triggered by the Autonomous SRE Project Onboarding AI Agent to onboard the **{project_name}** project to the **{target_branch}** environment.

### Reviewers
{mentions}

### Actions Required
- [ ] Review the project configuration
- [ ] Validate environment compatibility
- [ ] Approve and merge when ready

---
*Automated MR generated by SRE Project Onboarding Agent*
"""

    payload = {
        "source_branch": branch_name,
        "target_branch": target_branch,
        "title": mr_title,
        "description": mr_description,
        "reviewer_ids": reviewer_ids,
        "assignee_ids": reviewer_ids,
        "labels": ",".join(mr_labels)
    }

    headers = {"PRIVATE-TOKEN": gitlab_token}

    logger.info(f"📬 Creating Merge Request for project {project_name}...")

    response = requests.post(url, headers=headers, json=payload, timeout=10)

    if response.status_code != 201:
        raise Exception(f"MR creation failed: {response.text}")
    
    mr_data = response.json()
    logger.info(f"🎉 MR Created: {mr_data['web_url']}")
    return mr_data

def send_mr_notification_to_teams(mr_data: Dict[str, Any], ticket: str, env: str, 
                                   branch_name: str, target_branch: str, 
                                   project_name: str) -> None:
    """
    Send Microsoft Teams notification for created merge request.
    
    Args:
        mr_data: Merge request data from GitLab API
        ticket: Jira ticket number
        env: Environment name
        branch_name: Source branch name
        target_branch: Target branch name
        project_name: Name of the project being onboarded
    """
    teams_webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
    jira_base_url = os.getenv("JIRA_BASE_URL")
    git_committer_name = os.getenv("GIT_COMMITTER_NAME")
    
    reviewer_usernames_raw = os.getenv("REVIEWER_USERNAMES", "")
    reviewer_usernames = [u.strip() for u in reviewer_usernames_raw.split(",") if u.strip()]
    mr_labels = ["project-onboarding", "auto-generated", "SRE Autonomous AI Agent"]

    if not teams_webhook_url:
        logger.warning("TEAMS_WEBHOOK_URL not set — skipping Teams notification")
        return

    mr_url = mr_data.get("web_url", "N/A")
    mr_id = mr_data.get("iid", "N/A")
    mr_title = mr_data.get("title", "N/A")
    created_at = mr_data.get("created_at", "")[:19].replace("T", " ")
    author = mr_data.get("author", {}).get("name", git_committer_name or "N/A")
    jira_link = f"{jira_base_url}/{ticket}" if jira_base_url else ticket

    reviewers_display = ", ".join(reviewer_usernames) if reviewer_usernames else "None"

    message = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "summary": f"New MR: {project_name} onboarding to {env}",
        "themeColor": "0078D4",
        "title": f"🚀 Project Onboarding: {project_name}",
        "sections": [
            {
                "activityTitle": f"📋 **{mr_title}**",
                "activitySubtitle": f"Project **{project_name}** is being onboarded to **{env.upper()}** environment",
                "activityImage": "https://about.gitlab.com/images/press/logo/png/gitlab-icon-rgb.png",
                "facts": [
                    {"name": "🎯 Project:", "value": f"**{project_name}**"},
                    {"name": "🎫 Jira Ticket:", "value": f"[{ticket}]({jira_link})"},
                    {"name": "🔢 MR ID:", "value": f"!{mr_id}"},
                    {"name": "🌿 Source Branch:", "value": branch_name},
                    {"name": "🎯 Target Branch:", "value": target_branch},
                    {"name": "🌍 Environment:", "value": env.upper()},
                    {"name": "👤 Author:", "value": author},
                    {"name": "👥 Reviewers:", "value": reviewers_display},
                    {"name": "🏷️ Labels:", "value": ", ".join(mr_labels)},
                    {"name": "📅 Created At:", "value": created_at},
                    {"name": "📌 Status:", "value": "🟡 Open — Awaiting Review"},
                ],
                "markdown": True
            },
            {
                "title": "📝 What's Being Onboarded?",
                "text": f"""
**Project Configuration for {project_name}**
- Environment: {env.upper()}
- Configuration file has been added to the repository
- Ready for review and validation
                """,
                "markdown": True
            }
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name": "🔍 View Merge Request",
                "targets": [{"os": "default", "uri": mr_url}]
            },
            {
                "@type": "OpenUri",
                "name": "🎫 View Jira Ticket",
                "targets": [{"os": "default", "uri": jira_link}]
            }
        ]
    }

    try:
        response = requests.post(teams_webhook_url, json=message, timeout=10)
        response.raise_for_status()
        logger.info(f"📣 Teams notification sent for {project_name} ({ticket})")
    except Exception as e:
        logger.warning(f"Teams notification failed: {e}")

# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================
def process_single_file(file: str) -> None:
    """
    Process a single YAML file: create branch, commit, MR, and notify.
    
    Args:
        file: YAML filename to process
    """
    yaml_dir = os.getenv("YAML_DIR")
    
    logger.info(f"\n📄 Processing: {file}")
    full_path = os.path.join(yaml_dir, file)
    
    # Parse file to get all metadata
    ticket, env, project_name = parse_file(full_path)
    target_branch = env
    branch_name = f"{env}-{ticket}-{project_name}"
    
    # Create branch and commit with project name
    create_branch_and_commit(full_path, branch_name, target_branch, project_name)
    
    # Create MR with project name in title and description
    mr_data = create_merge_request(branch_name, target_branch, ticket, project_name)
    
    # Send Teams notification with project details
    send_mr_notification_to_teams(mr_data, ticket, env, branch_name, target_branch, project_name)

def process_yaml_files() -> None:
    """
    Main function to process all YAML files in the configured directory.
    """
    gitlab_token = os.getenv("GITLAB_TOKEN")
    yaml_dir = os.getenv("YAML_DIR")
    
    if not gitlab_token:
        raise Exception("❌ GITLAB_TOKEN not set")

    # Validate configuration first
    validate_config()
    
    # Validate git repository
    repo_path = os.getenv("REPO_PATH")
    validate_git_repo(repo_path)

    # Get all YAML files
    files = [f for f in os.listdir(yaml_dir) if f.endswith(".yaml")]

    if not files:
        logger.warning("⚠️ No YAML files found")
        return

    logger.info(f"Found {len(files)} YAML file(s) to process")
    
    # Process files sequentially
    success_count = 0
    failure_count = 0
    
    for file in files:
        try:
            process_single_file(file)
            success_count += 1
        except Exception as e:
            logger.error(f"❌ Failed to process {file}: {e}")
            failure_count += 1
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing complete: {success_count} succeeded, {failure_count} failed")
    logger.info(f"{'='*50}")

# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    try:
        process_yaml_files()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
