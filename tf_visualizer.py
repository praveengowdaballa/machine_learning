#!/usr/bin/env python3
"""
Terraform Plan Diff HTML Generator with LLM Explanation
Minimal implementation - no cost, no extra features
"""

import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import html

# Try to import requests for LLM, but make it optional
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

class TerraformDiffGenerator:
    def __init__(self, terraform_path=".", llm_model="llama3.1", ollama_url="http://localhost:11434"):
        self.terraform_path = Path(terraform_path)
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self.plan_file = None
        
    def check_dependencies(self):
        """Check if terraform is available"""
        try:
            subprocess.run(["terraform", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Error: terraform not found in PATH", file=sys.stderr)
            sys.exit(1)
    
    def generate_plan(self):
        """Generate Terraform plan"""
        print("📝 Generating Terraform plan...", file=sys.stderr)
        
        # Change to terraform directory
        os.chdir(self.terraform_path)
        
        # Initialize if needed
        if not Path(".terraform").exists():
            subprocess.run(["terraform", "init", "-input=false"], 
                          capture_output=True, check=False)
        
        # Create plan file
        with tempfile.NamedTemporaryFile(suffix=".tfplan", delete=False) as tmp:
            self.plan_file = tmp.name
        
        try:
            subprocess.run(
                ["terraform", "plan", "-out=" + self.plan_file, "-input=false", "-no-color"],
                capture_output=True,
                check=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print("❌ Terraform plan failed", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            sys.exit(1)
        
        return self.plan_file
    
    def extract_changes(self):
        """Extract and parse Terraform changes"""
        try:
            # Get JSON output from terraform show
            result = subprocess.run(
                ["terraform", "show", "-json", self.plan_file],
                capture_output=True,
                text=True,
                check=True
            )
            
            tf_json = json.loads(result.stdout)
            
            # Extract resource changes
            changes = []
            for resource in tf_json.get("resource_changes", []):
                actions = resource.get("change", {}).get("actions", [])
                if actions:
                    changes.append({
                        "address": resource.get("address", "unknown"),
                        "type": resource.get("type", "unknown"),
                        "action": actions[0],  # Primary action
                        "provider": resource.get("provider_name", "unknown"),
                        "before": resource.get("change", {}).get("before"),
                        "after": resource.get("change", {}).get("after")
                    })
            
            # Create summary
            summary = {
                "total": len(changes),
                "create": sum(1 for c in changes if c["action"] == "create"),
                "update": sum(1 for c in changes if c["action"] == "update"),
                "delete": sum(1 for c in changes if c["action"] == "delete"),
                "providers": list(set(c["provider"] for c in changes))
            }
            
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": summary,
                "changes": changes
            }
            
        except (json.JSONDecodeError, subprocess.CalledProcessError) as e:
            print(f"❌ Failed to parse terraform output: {e}", file=sys.stderr)
            sys.exit(1)
    
    def get_llm_explanation(self, changes_data):
        """Get LLM explanation of the changes"""
        if not HAS_REQUESTS:
            return "LLM explanation unavailable (requests library not installed)"
        
        # Prepare context for LLM
        providers = ", ".join(changes_data["summary"]["providers"])
        changes_summary = f"""
Total changes: {changes_data['summary']['total']}
- Create: {changes_data['summary']['create']}
- Update: {changes_data['summary']['update']}
- Delete: {changes_data['summary']['delete']}
Providers: {providers}

Resources changed:
{json.dumps(changes_data['changes'][:20], indent=2)}  # Limit to first 20 for brevity
"""
        
        prompt = f"""As a DevOps expert, analyze this Terraform plan and provide a brief, clear explanation of:
1. What infrastructure changes are being made
2. Potential impact of these changes
3. Any risks or considerations

Changes:
{changes_summary}

Provide a concise response (2-3 paragraphs maximum). Focus on actionable insights.
"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=360
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No analysis generated")
            else:
                return f"LLM analysis failed (HTTP {response.status_code})"
                
        except requests.exceptions.RequestException as e:
            return f"LLM analysis unavailable: {str(e)}"
    
    def generate_html_report(self, changes_data, llm_explanation=""):
        """Generate HTML report with diff visualization"""
        
        # Color mapping for actions
        action_colors = {
            "create": "#28a745",
            "update": "#ffc107",
            "delete": "#dc3545",
            "read": "#17a2b8"
        }
        
        action_icons = {
            "create": "➕",
            "update": "✏️",
            "delete": "🗑️",
            "read": "📖"
        }
        
        # Generate HTML table rows for changes
        changes_rows = []
        for change in changes_data["changes"][:50]:  # Limit to 50 for performance
            action = change["action"]
            color = action_colors.get(action, "#6c757d")
            icon = action_icons.get(action, "•")
            
            # Generate diff view for updates
            diff_html = ""
            if action == "update" and change["before"] and change["after"]:
                diff_html = '<div class="diff-view">'
                before_str = json.dumps(change["before"], indent=2)
                after_str = json.dumps(change["after"], indent=2)
                
                # Simple diff highlighting
                before_lines = before_str.split('\n')
                after_lines = after_str.split('\n')
                
                diff_html += '<details><summary>View changes</summary>'
                diff_html += '<div class="diff-container"><div class="diff-before"><strong>Before:</strong><pre>'
                diff_html += html.escape(before_str)
                diff_html += '</pre></div><div class="diff-after"><strong>After:</strong><pre>'
                diff_html += html.escape(after_str)
                diff_html += '</pre></div></div></details>'
                diff_html += '</div>'
            
            changes_rows.append(f"""
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px; color: {color}; font-weight: 600;">
                    {icon} {action.upper()}
                </td>
                <td style="padding: 12px; font-family: monospace;">{html.escape(change['address'])}</td>
                <td style="padding: 12px;">{html.escape(change['type'])}</td>
                <td style="padding: 12px;">{html.escape(change['provider'])}</td>
                <td style="padding: 12px;">{diff_html}</td>
            </tr>
            """)
        
        changes_table = ''.join(changes_rows) if changes_rows else '<tr><td colspan="5" style="padding: 40px; text-align: center;">No changes detected</td></tr>'
        
        # Summary statistics
        summary = changes_data["summary"]
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terraform Plan Analysis - {changes_data['timestamp']}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
        }}
        
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .header .timestamp {{
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .card .number {{
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .card .label {{
            color: #6c757d;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .card.create .number {{ color: #28a745; }}
        .card.update .number {{ color: #ffc107; }}
        .card.delete .number {{ color: #dc3545; }}
        .card.total .number {{ color: #667eea; }}
        
        .llm-section {{
            background: #f8f9fa;
            padding: 30px 40px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .llm-section h2 {{
            color: #667eea;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .llm-content {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            white-space: pre-wrap;
            font-family: inherit;
        }}
        
        .changes-section {{
            padding: 30px 40px;
        }}
        
        .changes-section h2 {{
            color: #667eea;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        .diff-view {{
            margin-top: 10px;
        }}
        
        .diff-view details {{
            cursor: pointer;
        }}
        
        .diff-view summary {{
            color: #667eea;
            font-weight: 500;
        }}
        
        .diff-container {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }}
        
        .diff-before, .diff-after {{
            flex: 1;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        
        .diff-before pre, .diff-after pre {{
            font-size: 12px;
            margin: 0;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #6c757d;
            font-size: 12px;
            border-top: 1px solid #dee2e6;
        }}
        
        @media (max-width: 768px) {{
            .summary-cards {{
                grid-template-columns: 1fr 1fr;
                padding: 20px;
            }}
            
            .changes-section, .llm-section {{
                padding: 20px;
            }}
            
            .diff-container {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏗️ Terraform Plan Analysis</h1>
            <div class="timestamp">Generated: {changes_data['timestamp']}</div>
        </div>
        
        <div class="summary-cards">
            <div class="card total">
                <div class="number">{summary['total']}</div>
                <div class="label">Total Changes</div>
            </div>
            <div class="card create">
                <div class="number">{summary['create']}</div>
                <div class="label">➕ Create</div>
            </div>
            <div class="card update">
                <div class="number">{summary['update']}</div>
                <div class="label">✏️ Update</div>
            </div>
            <div class="card delete">
                <div class="number">{summary['delete']}</div>
                <div class="label">🗑️ Delete</div>
            </div>
        </div>
        
        {f'''
        <div class="llm-section">
            <h2>
                <span>🤖</span>
                AI-Powered Analysis
            </h2>
            <div class="llm-content">
                {html.escape(llm_explanation).replace(chr(10), '<br>')}
            </div>
        </div>
        ''' if llm_explanation else ''}
        
        <div class="changes-section">
            <h2>📋 Resource Changes</h2>
            {'<p><small>Showing first 50 changes</small></p>' if len(changes_data['changes']) > 50 else ''}
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Action</th>
                            <th>Resource Address</th>
                            <th>Type</th>
                            <th>Provider</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {changes_table}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Terraform Plan Diff Generator</p>
            <p>Provider{'' if len(summary['providers']) == 1 else 's'}: {', '.join(summary['providers'])}</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.plan_file and Path(self.plan_file).exists():
            Path(self.plan_file).unlink()
    
    def run(self, enable_llm=True, output_file="terraform-plan-report.html"):
        """Main execution flow"""
        try:
            self.check_dependencies()
            self.generate_plan()
            changes = self.extract_changes()
            
            llm_explanation = ""
            if enable_llm and HAS_REQUESTS:
                print("🤖 Generating LLM explanation...", file=sys.stderr)
                llm_explanation = self.get_llm_explanation(changes)
            elif enable_llm and not HAS_REQUESTS:
                print("⚠️ LLM explanation disabled: requests library not installed", file=sys.stderr)
            
            print("📄 Generating HTML report...", file=sys.stderr)
            html_report = self.generate_html_report(changes, llm_explanation)
            
            # Write HTML file
            output_path = Path(output_file)
            output_path.write_text(html_report, encoding='utf-8')
            
            print(f"\n✅ Report generated successfully!", file=sys.stderr)
            print(f"📊 HTML report saved to: {output_path.absolute()}", file=sys.stderr)
            
            # Also print a summary to console
            print(f"\n📈 Summary:", file=sys.stderr)
            print(f"   Total: {changes['summary']['total']}", file=sys.stderr)
            print(f"   Create: {changes['summary']['create']}", file=sys.stderr)
            print(f"   Update: {changes['summary']['update']}", file=sys.stderr)
            print(f"   Delete: {changes['summary']['delete']}", file=sys.stderr)
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            return False
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Terraform plan diff HTML report with LLM explanation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--path",
        default=".",
        help="Path to Terraform configuration directory (default: current directory)"
    )
    
    parser.add_argument(
        "--output",
        default="terraform-plan-report.html",
        help="Output HTML file name (default: terraform-plan-report.html)"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM explanation (generate report without AI analysis)"
    )
    
    parser.add_argument(
        "--llm-model",
        default="llama3.1",
        help="Ollama model to use (default: llama3.1)"
    )
    
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = TerraformDiffGenerator(
        terraform_path=args.path,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url
    )
    
    # Run
    success = generator.run(
        enable_llm=not args.no_llm,
        output_file=args.output
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
