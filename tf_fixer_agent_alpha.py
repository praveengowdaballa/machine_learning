#!/usr/bin/env python3
"""
Intelligent Terraform Checkov Auto-Fixer
Uses Checkov JSON to identify fixes, then applies them intelligently with LLM.
Preserves all Terraform references, variables, and dependencies.
"""

import json
import subprocess
import argparse
import re
import shutil
import sys
import logging
from pathlib import Path
from typing import Optional, List, Set, Dict, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class FixStatus(Enum):
    """Status codes for fix operations."""
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    VALIDATION_ERROR = "validation_error"


@dataclass
class FixResult:
    """Result of a fix operation."""
    check_id: str
    resource: str
    file_path: Path
    status: FixStatus
    message: str = ""


class TerraformCheckovFixer:
    """
    Intelligent Terraform Checkov auto-fixer.
    Uses JSON to identify what needs fixing, LLM to apply fixes intelligently.
    """

    def __init__(
        self,
        tf_root: str,
        failed_json: str,
        fixes_dir: str = "terraform_fixes",
        output_dir: str = "terraform_fixes_review",
        model: str = "llama3",
        dry_run: bool = False,
        backup: bool = True,
        merge_directly: bool = False,
        max_retries: int = 2,
        log_level: str = "INFO",
    ):
        self.tf_root = Path(tf_root).resolve()
        self.failed_json = Path(failed_json).resolve()
        self.fixes_dir = Path(fixes_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.model = model
        self.dry_run = dry_run
        self.backup = backup
        self.merge_directly = merge_directly
        self.max_retries = max_retries
        self.generated_dependencies = set()
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Validation
        self._validate_inputs()
        
        # Check Ollama
        self._check_ollama()
        
        # Initialize
        if not self.merge_directly and not self.dry_run:
            self.output_dir.mkdir(exist_ok=True, parents=True)
        self.failed_checks = self._load_failed_checks()
        self.results: List[FixResult] = []

    def _setup_logging(self, log_level: str) -> None:
        """Configure logging."""
        log_file = 'terraform_fixer.log'
        
        logging.getLogger().handlers.clear()
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode='w')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_inputs(self) -> None:
        """Validate input paths."""
        if not self.tf_root.exists():
            raise ValueError(f"Terraform root not found: {self.tf_root}")
        
        if not self.failed_json.exists():
            raise ValueError(f"Failed checks JSON not found: {self.failed_json}")
        
        if not self.fixes_dir.exists():
            self.logger.warning(f"Fixes directory not found: {self.fixes_dir}")
            self.logger.warning("Creating fixes directory...")
            self.fixes_dir.mkdir(parents=True, exist_ok=True)

    def _check_ollama(self) -> None:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Ollama command failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError(
                "Ollama not found. Install from: https://ollama.ai\n"
                f"Then run: ollama pull {self.model}"
            )

    def _load_failed_checks(self) -> List[Dict]:
        """Load and parse Checkov failed checks JSON."""
        try:
            data = json.loads(self.failed_json.read_text())
            checks = []
            
            # Handle different JSON formats
            if isinstance(data, list):
                for entry in data:
                    checks.extend(entry.get("results", {}).get("failed_checks", []))
                    checks.extend(entry.get("failed_checks", []))
            else:
                checks.extend(data.get("results", {}).get("failed_checks", []))
                checks.extend(data.get("failed_checks", []))
            
            # Remove duplicates
            seen = set()
            unique_checks = []
            for check in checks:
                key = (check.get('check_id'), check.get('resource'), check.get('file_path'))
                if key not in seen:
                    seen.add(key)
                    unique_checks.append(check)
            
            self.logger.info(f"Loaded {len(unique_checks)} failed checks from JSON")
            return unique_checks
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.failed_json}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading failed checks: {e}")

    def run(self) -> Dict[str, int]:
        """Execute the fixing process."""
        self.logger.info("=" * 70)
        self.logger.info("TERRAFORM CHECKOV INTELLIGENT AUTO-FIXER")
        self.logger.info("=" * 70)
        self.logger.info(f"Terraform root:  {self.tf_root}")
        self.logger.info(f"Failed checks:   {len(self.failed_checks)}")
        self.logger.info(f"Fixes directory: {self.fixes_dir}")
        self.logger.info(f"Model:           {self.model}")
        self.logger.info(f"Mode:            {'DIRECT MERGE' if self.merge_directly else 'REVIEW'}")
        self.logger.info(f"Dry run:         {self.dry_run}")
        self.logger.info("=" * 70)
        
        all_changes: Dict[Path, List[Tuple]] = {}
        
        # Process each failed check from JSON
        for idx, check in enumerate(self.failed_checks, 1):
            check_id = check.get('check_id', 'UNKNOWN')
            resource = check.get('resource', 'UNKNOWN')
            file_abs = check.get('file_abs_path', check.get('file_path', 'UNKNOWN'))
            
            # Determine if file is in a module
            is_module = 'module.' in resource or '/module/' in str(file_abs)
            module_indicator = " [MODULE]" if is_module else ""
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"[{idx}/{len(self.failed_checks)}] Check: {check_id}{module_indicator}")
            self.logger.info(f"Resource: {resource}")
            self.logger.info(f"File: {Path(file_abs).name if file_abs != 'UNKNOWN' else 'UNKNOWN'}")
            if is_module:
                self.logger.info(f"Full Path: {file_abs}")
            self.logger.info(f"{'='*70}")
            
            result = self._process_check(check)
            
            if result:
                file_path, original_block, fixed_block, check_id = result
                if file_path not in all_changes:
                    all_changes[file_path] = []
                all_changes[file_path].append((original_block, fixed_block, check_id))
                
                self.results.append(FixResult(
                    check_id=check_id,
                    resource=check['resource'],
                    file_path=file_path,
                    status=FixStatus.SUCCESS
                ))

        # Apply changes
        if not self.dry_run and all_changes:
            if self.merge_directly:
                self._apply_direct_merges(all_changes)
            else:
                self._generate_review_files(all_changes)
        
        # Generate report
        stats = self._generate_report()
        
        return stats

    def _process_check(self, check: Dict) -> Optional[Tuple[Path, str, str, str]]:
        """Process a single Checkov failed check."""
        check_id = check["check_id"]
        check_name = check.get("check_name", check_id)
        resource = check["resource"]
        
        # Use absolute path if available, otherwise construct from relative path
        file_path = None
        if "file_abs_path" in check and check["file_abs_path"]:
            file_path = Path(check["file_abs_path"])
        else:
            file_path_str = check.get("file_path", "").lstrip("/")
            file_path = self.tf_root / file_path_str

        # Check if fix file exists
        fix_file = self.fixes_dir / f"{check_id}.tf"
        if not fix_file.exists():
            self.logger.warning(f"  ⏭️  No fix file found: {fix_file.name}")
            self.results.append(FixResult(
                check_id=check_id,
                resource=resource,
                file_path=file_path,
                status=FixStatus.SKIPPED,
                message="Fix file not found"
            ))
            return None

        # Validate Terraform file exists
        if not file_path.exists():
            self.logger.error(f"  ❌ Terraform file not found: {file_path}")
            self.results.append(FixResult(
                check_id=check_id,
                resource=resource,
                file_path=file_path,
                status=FixStatus.FAILED,
                message="Terraform file not found"
            ))
            return None

        # Parse resource identifier - handle module paths
        resource_parts = resource.split(".")
        
        # Remove module prefix(es) if present
        if resource_parts[0] == "module":
            # Resource type and name are the last two parts
            if len(resource_parts) >= 3:
                resource_type = resource_parts[-2]
                resource_name = resource_parts[-1]
            else:
                self.logger.error(f"  ❌ Invalid module resource format: {resource}")
                return None
        else:
            # Direct resource: resource_type.resource_name
            try:
                resource_type = resource_parts[0]
                resource_name = resource_parts[1]
            except (ValueError, IndexError):
                self.logger.error(f"  ❌ Invalid resource format: {resource}")
                return None

        # Extract original resource block
        tf_content = file_path.read_text()
        self.logger.info(f"  🔍 Looking for: {resource_type}.{resource_name} in {file_path.name}")
        
        original_block = self._extract_resource(tf_content, resource_type, resource_name)

        if not original_block:
            self.logger.error(f"  ❌ Resource not found in file: {resource_type}.{resource_name}")
            self.results.append(FixResult(
                check_id=check_id,
                resource=resource,
                file_path=file_path,
                status=FixStatus.FAILED,
                message=f"Resource {resource_type}.{resource_name} not found in file"
            ))
            return None

        # Load fix snippet from fixes directory
        fix_snippet = fix_file.read_text()
        self.logger.info(f"  📋 Using fix: {fix_file.name}")

        # Use LLM to intelligently merge fix while preserving all references
        fixed_block = None
        for attempt in range(self.max_retries):
            try:
                prompt = self._build_intelligent_prompt(
                    check_id,
                    check_name,
                    resource,
                    resource_type,
                    resource_name,
                    original_block,
                    fix_snippet,
                    tf_content,
                    attempt
                )
                
                self.logger.info(f"  🤖 Generating fix with {self.model} (attempt {attempt + 1}/{self.max_retries})...")
                fixed_block = self._run_ollama(prompt)
                
                # Validate
                is_valid, error = self._validate_block(fixed_block, resource_type, resource_name)
                if not is_valid:
                    self.logger.warning(f"  ⚠️ Validation failed: {error}")
                    fixed_block = None
                    continue

                semantic_ok, semantic_error = self._semantic_validate(fixed_block)
                if not semantic_ok:
                    self.logger.warning(f"  ⚠️ Semantic validation failed: {semantic_error}")
                    fixed_block = None
                    continue
                
                self.logger.info("  ✅ Fix validated successfully")
                break
                    
            except Exception as e:
                self.logger.error(f"  ❌ Error generating fix: {e}")
                fixed_block = None

        if not fixed_block:
            self.logger.error(f"  ❌ Failed to generate valid fix after {self.max_retries} attempts")
            self.results.append(FixResult(
                check_id=check_id,
                resource=resource,
                file_path=file_path,
                status=FixStatus.VALIDATION_ERROR,
                message="Could not generate valid Terraform"
            ))
            return None

        # Add context comment
        fixed_block_with_comment = self._add_context_comment(
            fixed_block, check_id, check_name, resource,
            self._get_safe_relative_path(file_path)
        )

        return (file_path, original_block, fixed_block_with_comment, check_id)

    def _get_safe_relative_path(self, file_path: Path) -> Path:
        """Get relative path safely, handling files outside tf_root."""
        try:
            return file_path.relative_to(self.tf_root)
        except ValueError:
            return Path(*file_path.parts[-3:]) if len(file_path.parts) >= 3 else file_path.name

    def _validate_block(self, block: str, resource_type: str = None, resource_name: str = None) -> Tuple[bool, str]:
        """Validate Terraform block structure."""
        if not block.strip():
            return False, "Empty block"
        
        if resource_type and resource_name:
            pattern = rf'resource\s+"{re.escape(resource_type)}"\s+"{re.escape(resource_name)}"'
            if not re.search(pattern, block):
                return False, "Resource declaration mismatch"
        
        if not block.strip().startswith("resource"):
            return False, "Block must start with 'resource'"
        
        # Check brace matching
        open_braces = block.count('{')
        close_braces = block.count('}')
        
        if open_braces == 0:
            return False, "No opening braces"
        
        if open_braces != close_braces:
            return False, f"Unmatched braces: {open_braces} open, {close_braces} close"
        
        return True, ""

    def _semantic_validate(self, block: str) -> Tuple[bool, str]:
        """Rejects semantically incomplete Terraform values introduced by the LLM."""
        placeholder_patterns = [
            r'"\s*test\s*"',
            r'"\s*example\s*"',
            r'"\s*dummy\s*"',
            r'"\s*todo\s*"',
            r'"\s*changeme\s*"',
            r'"\s*replace_me\s*"',
            r'"\s*your_.*?\s*"',
            r'"\s*<.*?>\s*"',
            r'""'
        ]

        for pattern in placeholder_patterns:
            if re.search(pattern, block, re.IGNORECASE):
                return False, f"Placeholder value detected: {pattern}"

        return True, ""

    def _build_intelligent_prompt(
        self,
        check_id: str,
        check_name: str,
        resource: str,
        resource_type: str,
        resource_name: str,
        original_block: str,
        fix_snippet: str,
        tf_content: str,
        attempt: int = 0,
    ) -> str:
        """Build intelligent LLM prompt that preserves all Terraform references."""
        
        retry_note = ""
        if attempt > 0:
            retry_note = f"\n\n⚠️ RETRY ATTEMPT {attempt + 1}: Previous output was invalid."
        
        return f"""You are an expert Terraform security engineer. Apply this Checkov security fix while preserving ALL existing configuration.

CRITICAL REQUIREMENTS:
1. Output ONLY the complete Terraform resource block
2. Start with: resource "{resource_type}" "{resource_name}"
3. Preserve EVERY existing attribute and reference
4. Add or modify ONLY what's required by the security fix
5. Maintain proper HCL syntax with balanced braces
6. NO explanations or markdown fences

FAILED SECURITY CHECK:
Check ID: {check_id}
Check Name: {check_name}
Resource: {resource}

ORIGINAL RESOURCE BLOCK:
{original_block}

REQUIRED SECURITY FIX:
{fix_snippet}

MERGE STRATEGY:
1. Start with the complete original resource structure
2. Add the missing security attributes
3. Keep ALL existing attributes intact{retry_note}

OUTPUT THE COMPLETE MERGED RESOURCE BLOCK NOW:"""

    def _run_ollama(self, prompt: str) -> str:
        """Execute Ollama model with prompt."""
        try:
            proc = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=90,
            )

            if proc.returncode != 0:
                raise RuntimeError(f"Ollama error: {proc.stderr}")

            output = proc.stdout.strip()
            
            # Clean markdown code fences if present
            output = re.sub(r'^```(?:hcl|terraform|tf)?\n?', '', output, flags=re.MULTILINE)
            output = re.sub(r'\n?```$', '', output)
            
            # Remove any leading/trailing explanations
            lines = output.split('\n')
            start_idx = 0
            end_idx = len(lines)
            
            for i, line in enumerate(lines):
                if line.strip().startswith('resource '):
                    start_idx = i
                    break
            
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == '}':
                    end_idx = i + 1
                    break
            
            output = '\n'.join(lines[start_idx:end_idx])
            
            return output.strip()
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama timeout (>90s)")
        except FileNotFoundError:
            raise RuntimeError("Ollama not found. Install from: https://ollama.ai")
        except Exception as e:
            raise RuntimeError(f"Ollama execution error: {e}")

    def _extract_resource(self, content: str, resource_type: str, resource_name: str) -> Optional[str]:
        """Extract a Terraform resource block from content."""
        escaped_type = re.escape(resource_type)
        escaped_name = re.escape(resource_name)
        
        # Find resource declaration
        pattern = rf'resource\s+"{escaped_type}"\s+"{escaped_name}"\s*{{'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        
        if not match:
            return None
        
        start = match.start()
        brace_start = match.end() - 1
        
        # Track brace depth to find matching closing brace
        depth = 1
        pos = brace_start + 1
        
        while pos < len(content) and depth > 0:
            if content[pos] == '{':
                depth += 1
            elif content[pos] == '}':
                depth -= 1
            pos += 1
        
        if depth == 0:
            return content[start:pos]
        
        return None

    def _apply_direct_merges(self, all_changes: Dict[Path, List[Tuple]]) -> None:
        """Apply fixes directly to Terraform files."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("APPLYING DIRECT MERGES")
        self.logger.info("=" * 70)
        
        for file_path, changes in all_changes.items():
            try:
                # Create backup
                if self.backup:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak.{timestamp}")
                    shutil.copy(file_path, backup_path)
                    self.logger.info(f"  💾 Backup: {backup_path.name}")
                
                # Apply all changes to this file
                current_content = file_path.read_text()
                
                for original_block, fixed_block, check_id in changes:
                    if original_block in current_content:
                        current_content = current_content.replace(original_block, fixed_block, 1)
                        self.logger.info(f"  ✓ Applied {check_id}")
                    else:
                        self.logger.warning(f"  ⚠️  Original block not found for {check_id}")
                
                # Write updated content
                file_path.write_text(current_content)
                self.logger.info(f"✅ Updated: {self._get_safe_relative_path(file_path)}\n")
                
            except Exception as e:
                self.logger.error(f"❌ Error updating {file_path}: {e}")

    def _generate_review_files(self, all_changes: Dict[Path, List[Tuple]]) -> None:
        """Generate review files for manual approval."""
        import difflib
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("GENERATING REVIEW FILES")
        self.logger.info("=" * 70)
        
        for file_path, changes in all_changes.items():
            try:
                # Generate merged content
                merged_content = file_path.read_text()
                for original_block, fixed_block, _ in changes:
                    if original_block in merged_content:
                        merged_content = merged_content.replace(original_block, fixed_block, 1)
                
                # Create output file path
                relative_path = self._get_safe_relative_path(file_path)
                output_file = self.output_dir / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(merged_content)
                
                # Generate diff
                original_content = file_path.read_text()
                diff = list(difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    merged_content.splitlines(keepends=True),
                    fromfile=str(relative_path),
                    tofile=str(relative_path) + " (fixed)",
                    lineterm='\n'
                ))
                
                if diff:
                    diff_file = output_file.with_suffix(output_file.suffix + '.diff')
                    diff_file.write_text(''.join(diff))
                    self.logger.info(f"  📊 {diff_file.relative_to(self.output_dir)}")
                
                self.logger.info(f"  ✅ {output_file.relative_to(self.output_dir)}\n")
                
            except Exception as e:
                self.logger.error(f"  ❌ Error generating review: {e}")

    def _generate_report(self) -> Dict[str, int]:
        """Generate final statistics report."""
        stats = {
            "total": len(self.failed_checks),
            "success": sum(1 for r in self.results if r.status == FixStatus.SUCCESS),
            "skipped": sum(1 for r in self.results if r.status == FixStatus.SKIPPED),
            "failed": sum(1 for r in self.results if r.status == FixStatus.FAILED),
            "validation_error": sum(1 for r in self.results if r.status == FixStatus.VALIDATION_ERROR),
        }
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FINAL REPORT")
        self.logger.info("=" * 70)
        self.logger.info(f"Total Failed Checks:    {stats['total']}")
        self.logger.info(f"✅ Successfully Fixed:    {stats['success']}")
        self.logger.info(f"⏭️  Skipped (no fix):      {stats['skipped']}")
        self.logger.info(f"❌ Failed:                {stats['failed']}")
        self.logger.info(f"⚠️  Validation Error:      {stats['validation_error']}")
        self.logger.info("=" * 70)
        
        return stats

    def _add_context_comment(
        self, resource_block: str, check_id: str, check_name: str,
        resource: str, file_path: Path
    ) -> str:
        """Add context comment header to fixed resource."""
        lines = resource_block.strip().split('\n')
        resource_declaration = lines[0]
        rest_of_block = '\n'.join(lines[1:])
        
        comment = f"""# Checkov Security Fix Applied: {check_id}
# Check: {check_name}
# Resource: {resource}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        return f"{resource_declaration}\n{comment}{rest_of_block}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Terraform Checkov Auto-Fixer")
    
    parser.add_argument("--tf-root", required=True, help="Root directory of Terraform code")
    parser.add_argument("--failed-json", required=True, help="Checkov failed checks JSON file")
    parser.add_argument("--fixes-dir", default="terraform_fixes", help="Directory with fix snippets")
    parser.add_argument("--output-dir", default="terraform_fixes_review", help="Output directory")
    parser.add_argument("--model", default="llama3", help="Ollama model")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backups")
    parser.add_argument("--merge-directly", action="store_true", help="Apply fixes directly")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retry attempts")
    parser.add_argument("--log-level", default='INFO', help="Logging level")
    
    args = parser.parse_args()

    try:
        fixer = TerraformCheckovFixer(
            tf_root=args.tf_root,
            failed_json=args.failed_json,
            fixes_dir=args.fixes_dir,
            output_dir=args.output_dir,
            model=args.model,
            dry_run=args.dry_run,
            backup=args.backup,
            merge_directly=args.merge_directly,
            max_retries=args.max_retries,
            log_level=args.log_level,
        )

        stats = fixer.run()
        
        if stats['failed'] > 0 or stats['validation_error'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
