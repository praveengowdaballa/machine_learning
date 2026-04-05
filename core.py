import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
from collections import defaultdict

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_TOKEN = os.getenv("JIRA_TOKEN")
PROJECT_KEY = os.getenv("PROJECT_KEY", "CSDCS")
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("LLAMA_MODEL", "llama3.1:latest")
YAML_DOWNLOAD_DIR = os.getenv("YAML_DOWNLOAD_DIR", "./onboarding_yamls")

# === ONBOARDING DETECTION ===
ONBOARDING_KEYWORDS = [
    "onboarding", "onboard", "new environment", "new project", "new gcp",
    "gcp setup", "gcp onboard", "cloud onboard", "new tenant", "provision",
    "new account", "environment request", "env request", "new env",
    "cloud provider", "new cloud", "infrastructure setup", "infra setup",
    "project setup", "gcp project", "gcp provisioning"
]

ONBOARDING_MAX_AGE_DAYS = 2

CLOSED_STATUSES = {"done", "closed", "resolved", "won't fix", "wontfix", "cancelled", "canceled"}

# === ENHANCED PRIORITY MAPPING ===
PRIORITY_EMOJI = {
    "Blocker": "🚨",
    "Critical": "🔥",
    "Major": "⚡",
    "High": "🌡️",
    "Medium": "🟡",
    "Low": "🟢",
    "Lowest": "🍃",
}

PRIORITY_COLOR = {
    "Blocker": "D83B01",
    "Critical": "E81123",
    "Major": "FF8C00",
    "High": "FFA500",
    "Medium": "FFD700",
    "Low": "107C10",
    "Lowest": "CCCCCC",
}

GIF_LIBRARY = {
    "unassigned": "https://media.giphy.com/media/xTiTnGeUsWOEwsGoG4/giphy.gif",
    "reopened":   "https://media.giphy.com/media/l2JIdnF6aJnAqzDgY/giphy.gif",
    "onboarding": "https://media.giphy.com/media/3oKIPa2TdahY8LAAxy/giphy.gif",
}

def validate_jira_connection():
    print("🔍 Validating Jira configuration...")

    missing = []
    for var in ["JIRA_URL", "JIRA_USER", "JIRA_TOKEN"]:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("👉 Check your .env file")
        exit(1)

    print("✅ Environment variables loaded")

    # Test Jira API
    test_url = f"{JIRA_URL}/rest/api/2/myself"

    try:
        response = requests.get(
            test_url,
            auth=HTTPBasicAuth(JIRA_USER, JIRA_TOKEN),
            timeout=10
        )

        if response.status_code == 200:
            user = response.json().get("displayName", "Unknown")
            print(f"✅ Jira connection successful (User: {user})")

        elif response.status_code == 401:
            print("❌ Authentication failed (401)")
            print("👉 Check JIRA_USER or JIRA_TOKEN")
            exit(1)

        else:
            print(f"❌ Jira connection failed: {response.status_code}")
            print(response.text)
            exit(1)

    except Exception as e:
        print(f"❌ Jira connection error: {e}")
        exit(1)



def get_member_emails():
    """Return list of allowed member emails from .env (comma-separated)."""
    members_raw = os.getenv("MEMBER_EMAILS", "")
    members = [m.strip().lower() for m in members_raw.split(",") if m.strip()]
    print(f"✅ Loaded {len(members)} member emails from .env")
    return members


def get_jira_issues():
    """Fetch all non-closed issues for the project."""
    jql = (
        f"project={PROJECT_KEY} AND statusCategory!=Done "
        f"ORDER BY priority DESC, updated DESC"
    )
    url = f"{JIRA_URL}/rest/api/2/search"
    params = {
        "jql": jql,
        "maxResults": 1000,
        "fields": (
            "summary,description,status,priority,assignee,"
            "created,updated,duedate,issuelinks,comment,"
            "labels,components,attachment"
        )
    }
    response = requests.get(url, params=params, auth=HTTPBasicAuth(JIRA_USER, JIRA_TOKEN))
    response.raise_for_status()
    return response.json()["issues"]


def has_yaml_attachment(issue):
    """Return True if the issue has at least one .yaml or .yml attachment."""
    attachments = issue["fields"].get("attachment") or []
    return any(
        att.get("filename", "").lower().endswith((".yaml", ".yml"))
        for att in attachments
    )


def is_onboarding_ticket(issue):
    """
    Return True when ALL three conditions are met:
      1. Summary / description / labels / components contain an onboarding keyword
      2. At least one .yaml / .yml file is attached
      3. Ticket is within ONBOARDING_MAX_AGE_DAYS (if that limit is set)
    """
    fields = issue["fields"]

    text = " ".join(filter(None, [
        fields.get("summary", ""),
        str(fields.get("description") or ""),
        " ".join(fields.get("labels", [])),
        " ".join(c.get("name", "") for c in (fields.get("components") or [])),
    ])).lower()

    if not any(kw in text for kw in ONBOARDING_KEYWORDS):
        return False

    if not has_yaml_attachment(issue):
        return False

    if ONBOARDING_MAX_AGE_DAYS is not None:
        created_raw = fields.get("created")
        if created_raw:
            created_date = datetime.strptime(created_raw.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            if (datetime.now() - created_date).days > ONBOARDING_MAX_AGE_DAYS:
                return False

    return True


def classify_and_group_issues(issues):
    """
    Split issues into three buckets:
      - assignee_issues  : dict  email -> [issue, ...]
      - unassigned_issues: list
      - onboarding_issues: list
    """
    assignee_issues   = defaultdict(list)
    unassigned_issues = []
    onboarding_issues = []

    for issue in issues:
        try:
            fields = issue["fields"]
            status = fields.get("status", {}).get("name", "")

            if status.strip().lower() in CLOSED_STATUSES:
                continue

            if is_onboarding_ticket(issue):
                onboarding_issues.append(issue)
                continue

            assignee_obj = fields.get("assignee")
            if not assignee_obj:
                unassigned_issues.append(issue)
            else:
                assignee_email = assignee_obj.get(
                    "emailAddress", assignee_obj.get("name", "unknown")
                )
                assignee_issues[assignee_email].append(issue)

        except Exception as e:
            print(f"⚠️ Error processing issue {issue.get('key', 'unknown')}: {e}")
            continue

    return assignee_issues, unassigned_issues, onboarding_issues


def generate_enhanced_ai_analysis(issue):
    """Generate comprehensive AI analysis with context."""
    fields         = issue["fields"]
    summary        = fields.get("summary", "No summary")
    description    = fields.get("description") or "No description provided"
    labels         = ", ".join(fields.get("labels", [])) or "None"
    components     = ", ".join([c.get("name", "") for c in (fields.get("components") or [])]) or "None"
    comments_count = len((fields.get("comment") or {}).get("comments", []))
    yaml_files     = [
        att["filename"] for att in (fields.get("attachment") or [])
        if att.get("filename", "").lower().endswith((".yaml", ".yml"))
    ]
    yaml_note = f"YAML files attached: {', '.join(yaml_files)}" if yaml_files else ""
    description_text = str(description)[:500]

    prompt = f"""
You are a technical analyst and Principal SRE Engineer reviewing a JIRA issue. Provide a focused analysis:

**Issue:** {summary}
**Description:** {description_text}
**Labels:** {labels}
**Components:** {components}
**Comments:** {comments_count}
{yaml_note}

Provide:
1. Root Cause Analysis (1 sentence)
2. Impact Assessment (1 sentence)
3. Recommended Action (1 sentence with specific steps)
4. Estimated Effort (1 sentence: complexity & time estimate)

Be concise, technical, and actionable. Use markdown bullet points.
"""

    try:
        response = requests.post(
            LLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200}
            },
            timeout=90
        )
        response.raise_for_status()
        ai_response = response.json().get("response", "No AI analysis available.")
        return f"🤖 **SRE AI Agent Analysis:**\n\n{ai_response}"
    except Exception as e:
        return f"⚠️ AI analysis unavailable: {str(e)}"


def calculate_metrics(issue):
    """Calculate useful metrics for each issue."""
    try:
        fields   = issue["fields"]
        created  = fields.get("created")
        updated  = fields.get("updated")

        if not created or not updated:
            return {"age": "Unknown", "idle": "Unknown", "urgency_score": 0}

        created_date = datetime.strptime(created.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        updated_date = datetime.strptime(updated.split(".")[0], "%Y-%m-%dT%H:%M:%S")

        return {
            "age":           f"{(datetime.now() - created_date).days}d old",
            "idle":          f"{(datetime.now() - updated_date).days}d idle",
            "urgency_score": calculate_urgency_score(issue)
        }
    except Exception:
        return {"age": "Error", "idle": "Error", "urgency_score": 0}


def calculate_urgency_score(issue):
    """Calculate urgency score (0-100)."""
    try:
        score         = 0
        priority      = issue["fields"].get("priority")
        priority_name = priority["name"] if priority else "Medium"
        priority_scores = {
            "Blocker": 40, "Critical": 30, "Major": 20, "High": 15,
            "Medium": 10,  "Low": 5,       "Lowest": 2
        }
        score += priority_scores.get(priority_name, 10)

        created = issue["fields"].get("created")
        if created:
            created_date = datetime.strptime(created.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            if (datetime.now() - created_date).days > 30:
                score += 30

        updated = issue["fields"].get("updated")
        if updated:
            updated_date = datetime.strptime(updated.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            if (datetime.now() - updated_date).days > 7:
                score += 20

        if not issue["fields"].get("assignee"):
            score += 10

        return min(score, 100)
    except Exception:
        return 50


def build_issue_section(issue, category_config, extra_facts=None):
    """Build a single MessageCard section dict for one issue."""
    fields       = issue["fields"]
    issue_key    = issue.get("key", "UNKNOWN")
    summary      = fields.get("summary", "No summary")
    status       = fields.get("status", {}).get("name", "Unknown")
    priority_obj = fields.get("priority")
    priority     = priority_obj["name"] if priority_obj else "Medium"
    link         = f"{JIRA_URL}/browse/{issue_key}"
    emoji        = PRIORITY_EMOJI.get(priority, "🟡")
    metrics      = calculate_metrics(issue)
    ai_analysis  = generate_enhanced_ai_analysis(issue)

    yaml_files = [
        att["filename"] for att in (fields.get("attachment") or [])
        if att.get("filename", "").lower().endswith((".yaml", ".yml"))
    ]
    yaml_value = ", ".join(yaml_files) if yaml_files else "None"

    facts = (extra_facts or []) + [
        {"name": "🏷️ Status:",     "value": status},
        {"name": "📊 Priority:",   "value": f"{emoji} {priority}"},
        {"name": "⏱️ Age:",        "value": metrics["age"]},
        {"name": "💤 Idle:",       "value": metrics["idle"]},
        {"name": "🎯 Urgency:",    "value": f"{metrics['urgency_score']}/100"},
        {"name": "📎 YAML files:", "value": yaml_value},
    ]

    return {
        "activityTitle":    f"{category_config['emoji']} **[{issue_key}]({link})**",
        "activitySubtitle": summary,
        "facts":            facts,
        "text":             ai_analysis,
        "markdown":         True
    }


def download_yaml_attachments(issue):
    """Download all YAML attachments from an issue to local directory."""
    os.makedirs(YAML_DOWNLOAD_DIR, exist_ok=True)

    issue_key   = issue.get("key", "UNKNOWN")
    attachments = issue["fields"].get("attachment") or []
    downloaded  = []

    for att in attachments:
        filename = att.get("filename", "")
        if not filename.lower().endswith((".yaml", ".yml")):
            continue

        url     = att.get("content")
        save_as = os.path.join(YAML_DOWNLOAD_DIR, f"{issue_key}_{filename}")

        try:
            resp = requests.get(url, auth=HTTPBasicAuth(JIRA_USER, JIRA_TOKEN), timeout=30)
            resp.raise_for_status()
            with open(save_as, "wb") as f:
                f.write(resp.content)
            print(f"  ✅ Downloaded: {save_as}")
            downloaded.append(save_as)
        except Exception as e:
            print(f"  ⚠️ Failed to download {filename} from {issue_key}: {e}")

    return downloaded


def send_onboarding_issues_to_teams(issues_list):
    """
    Dedicated Teams card for GCP / cloud onboarding tickets with YAML attachment.
    """
    config   = {"emoji": "☁️", "color": "0078D4", "gif": GIF_LIBRARY["onboarding"]}
    sections = []

    for issue in issues_list[:20]:
        try:
            fields        = issue["fields"]
            assignee_obj  = fields.get("assignee")
            assignee_name = assignee_obj["displayName"] if assignee_obj else "⚠️ Unassigned"
            extra = [{"name": "👤 Assignee:", "value": assignee_name}]
            sections.append(build_issue_section(issue, config, extra_facts=extra))
        except Exception as e:
            print(f"⚠️ Error processing onboarding issue: {e}")

    if not sections:
        sections.append({"text": "No onboarding issues found."})

    age_note = (
        f"Tickets created within the last {ONBOARDING_MAX_AGE_DAYS} days."
        if ONBOARDING_MAX_AGE_DAYS
        else "No date restriction — showing all open onboarding tickets."
    )

    message = {
        "@type":      "MessageCard",
        "@context":   "https://schema.org/extensions",
        "summary":    "GCP Onboarding / New Environment Requests",
        "themeColor": config["color"],
        "title":      f"☁️ GCP Onboarding & New Environment Requests ({len(issues_list)})",
        "heroImage":  {"image": config["gif"]},
        "text":       f"**Filter:** OCP GCP Project Onboarding Tickets. {age_note}",
        "sections":   sections,
        "potentialAction": [{
            "@type": "OpenUri",
            "name":  "View Onboarding Issues in Jira",
            "targets": [{"os": "default", "uri": (
                f"{JIRA_URL}/issues/?jql=project={PROJECT_KEY}+AND+statusCategory!=Done"
            )}]
        }]
    }

    resp = requests.post(TEAMS_WEBHOOK_URL, json=message)
    resp.raise_for_status()


def main():
    """Main entry point — called by CLI command `ocp-onboarder`."""
    print("Validating Jira Connectivity....")
    validate_jira_connection()
    
    print("🔍 Fetching Jira issues...")
    issues = get_jira_issues()

    print(f"📊 Found {len(issues)} issues. Classifying and grouping...")
    assignee_issues, unassigned_issues, onboarding_issues = classify_and_group_issues(issues)

    member_emails = get_member_emails()

    # ----------------------------------------------------
    # Process onboarding issues
    # ----------------------------------------------------
    if onboarding_issues:
        print(f"☁️  Sending Onboarding Issues: {len(onboarding_issues)} issues")
        print(f"📥 Downloading YAML attachments to {YAML_DOWNLOAD_DIR}/")
        
        downloaded_files = []
        for issue in onboarding_issues:
            files = download_yaml_attachments(issue)
            downloaded_files.extend(files)

        send_onboarding_issues_to_teams(onboarding_issues)

    else:
        print("ℹ️  No onboarding tickets with YAML attachments found.")
        downloaded_files = []

    print("✅ Done!")

    # ====================================================
    # ✅ 🔥 ADD THIS RETURN BLOCK (CRITICAL FOR AGENT)
    # ====================================================
    return {
        "total_issues": len(issues),
        "onboarding_count": len(onboarding_issues),
        "onboarding_tickets": [issue["key"] for issue in onboarding_issues],
        "downloaded_files": downloaded_files,
        "status": "success"
    }


if __name__ == "__main__":
    main()

