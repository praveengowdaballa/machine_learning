#!/usr/bin/env python3
"""
Terraform Fixer - One script to fix Checkov issues
Usage: python tf_fixer.py [terraform_directory]
"""

import json
import os
import re
import requests
import subprocess
import sys

def main():
    print("=" * 60)
    print("TERRAFORM FIXER")
    print("=" * 60)
    
    # Get directory from command line or use current
    if len(sys.argv) > 1:
        terraform_dir = sys.argv[1]
    else:
        terraform_dir = "."
    
    print(f"📁 Analyzing: {terraform_dir}")
    print()
    
    # STEP 1: Run Checkov
    print("1️⃣  RUNNING CHECKOV...")
    try:
        cmd = ["checkov", "-d", terraform_dir, "-o", "json", "--quiet"]
        print(f"   Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            with open("checkov_results.json", "w") as f:
                f.write(result.stdout)
            print("   ✅ Results saved to: checkov_results.json")
        else:
            print("   ❌ Checkov produced no output")
            return
    except Exception as e:
        print(f"   ❌ Checkov failed: {e}")
        return
    
    # STEP 2: Parse results
    print("\n2️⃣  PARSING RESULTS...")
    try:
        with open("checkov_results.json", "r") as f:
            data = json.load(f)
        
        failed_checks = data.get("results", {}).get("failed_checks", [])
        print(f"   Found {len(failed_checks)} failed checks")
        
        if not failed_checks:
            print("   🎉 No failed checks found!")
            return
            
        # Save just failed checks
        failed_data = {
            "failed_checks": failed_checks,
            "summary": data.get("summary", {})
        }
        
        with open("checkov_failed.json", "w") as f:
            json.dump(failed_data, f, indent=2)
        print("   ✅ Failed checks saved to: checkov_failed.json")
        
    except Exception as e:
        print(f"   ❌ Failed to parse results: {e}")
        return
    
    # STEP 3: Download fixes
    print("\n3️⃣  DOWNLOADING FIXES...")
    
    # Create output directory
    fixes_dir = "terraform_fixes"
    os.makedirs(fixes_dir, exist_ok=True)
    
    success_count = 0
    
    for i, check in enumerate(failed_checks, 1):
        check_id = check.get("check_id", "UNKNOWN")
        check_name = check.get("check_name", "No name")
        guideline = check.get("guideline")
        
        print(f"\n   [{i}/{len(failed_checks)}] {check_id}")
        print(f"      {check_name}")
        
        if not guideline:
            print(f"      ⚠️  No guideline URL, skipping")
            continue
        
        # Convert to raw GitHub URL
        try:
            # Extract path from URL
            path = guideline.replace(
                "https://docs.prismacloud.io/en/enterprise-edition/policy-reference/",
                ""
            )
            raw_url = f"https://raw.githubusercontent.com/hlxsites/prisma-cloud-docs/refs/heads/main/docs/en/enterprise-edition/policy-reference/{path}.adoc"
            
            # Download the documentation
            print(f"      📥 Downloading documentation...")
            response = requests.get(raw_url, timeout=10)
            
            if response.status_code != 200:
                print(f"      ❌ Failed to download (HTTP {response.status_code})")
                continue
            
            # Extract Terraform code
            content = response.text
            pattern = r'\[source,[^\]]+\]\s*----\s*(.*?)\s*----'
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if not matches:
                print(f"      ⚠️  No Terraform code found")
                continue
            
            # Find the first block that looks like Terraform
            terraform_code = None
            for match in matches:
                if 'resource "' in match or 'module "' in match:
                    terraform_code = match.strip()
                    break
            
            if not terraform_code:
                print(f"      ⚠️  No Terraform code in documentation")
                continue
            
            # Save the fix
            filename = f"{fixes_dir}/{check_id}.tf"
            with open(filename, "w") as f:
                f.write(terraform_code)
            
            print(f"      ✅ Fix saved: {filename}")
            success_count += 1
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)[:100]}...")
            continue
    
    # STEP 4: Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total failed checks: {len(failed_checks)}")
    print(f"Fixes downloaded: {success_count}")
    print(f"Output directory: {fixes_dir}/")
    print()
    
    if success_count > 0:
        print("Generated files:")
        for file in os.listdir(fixes_dir):
            if file.endswith('.tf'):
                print(f"  - {fixes_dir}/{file}")
    
    print("\n✅ Done! Review the fixes in the 'terraform_fixes' folder.")
    print("=" * 60)

if __name__ == "__main__":
    main()
