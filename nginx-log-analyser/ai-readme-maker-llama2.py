import os
import ollama

def get_files_by_extension(path, extensions):
    files = []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(extensions):
                files.append(os.path.join(root, f))
    return files

def summarize_file(file_path):
    print(f"\n🔍 Summarizing: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content.strip():
            print("⚠️ Skipped: File is empty.")
            return None

        prompt = (
            f"You are a CCoE Platform Engineer. Summarize the following file for a README.md:\n\n"
            f"{content[:1500]}"
        )

        response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': prompt}])
        summary = response['message']['content']

        print("✅ Summary generated:\n")
        print(summary)
        print("-" * 80)

        return f"### `{os.path.basename(file_path)}`\n{summary}\n"

    except Exception as e:
        print(f"❌ Error summarizing {file_path}: {e}")
        return None

def generate_readme(directory):
    extensions = ('.tf', '.py', '.ps1', '.sh')
    files = get_files_by_extension(directory, extensions)

    print(f"🧠 Found {len(files)} code files to summarize...")

    file_summaries = [summarize_file(f) for f in files if summarize_file(f)]

    intro_prompt = (
        "You are a CCoE platform engineer. "
        "Write a professional README.md for this infrastructure project. "
        "Include a short project overview, sections for Installation, Usage, Folder Structure, and Contact. "
        "Assume it's used by engineers automating cloud provisioning and ops using Terraform, Python, PowerShell, and Bash."
    )
    intro = ollama.chat(model='llama2:latest', messages=[{'role': 'user', 'content': intro_prompt}])['message']['content']

    full_readme = f"# Platform Engineering Project\n\n{intro}\n\n## 📂 File Summaries\n" + "\n".join(file_summaries)

    readme_path = os.path.join(directory, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(full_readme)

    print(f"✅ README.md generated at {readme_path}")

# Replace with your project folder
generate_readme('/mnt/c/Users/pha/ot_workloads/gss-citrix/')

