import boto3
import argparse
from datetime import datetime
import time
import requests
import json
import os
import re
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
def parse_llm_response(llm_response):
    print("=== RAW LLM RESPONSE ===")
    print(llm_response)

    try:
        # Attempt strict JSON parse
        parsed = json.loads(llm_response)
        return parsed
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parsing failed: {e}")

        # Fallback: Try to extract JSON object using regex
        match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                print("[INFO] Recovered JSON from fallback regex.")
                return parsed
            except json.JSONDecodeError as e:
                print(f"[ERROR] Fallback JSON also failed: {e}")

        # Fallback 2: Try to extract "intent" manually
        if "list" in llm_response.lower():
            print("[INFO] Fallback to 'list' intent based on keyword.")
            return {
                "intent": "list",
                "region": "us-west-2",  # or default/fallback
                "parameters": {}
            }

        print("[ERROR] Could not parse LLM response.")
        return None

class WAFMigrationAssistant:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.region = self.determine_region()
        self.account_id = self.get_account_id()
        self.session = boto3.Session(region_name=self.region)
        self.initialize_clients()
        
    def determine_region(self):
        """Determine AWS region from environment or default"""
        return os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    def get_account_id(self):
        """Get AWS account ID"""
        try:
            sts_client = boto3.client('sts')
            return sts_client.get_caller_identity()["Account"]
        except (NoCredentialsError, BotoCoreError) as e:
            print(f"❌ AWS credentials error: {str(e)}")
            print("Please configure your AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
            exit(1)
    
    def initialize_clients(self):
        """Initialize AWS clients with proper region"""
        self.waf_regional = self.session.client('waf-regional')
        self.cf_client = self.session.client('cloudformation')
        self.s3_client = self.session.client('s3')
    
    def verify_credentials(self):
        """Verify AWS credentials are valid"""
        try:
            self.account_id  # This will trigger credential check in get_account_id
            return True
        except Exception as e:
            return False
    
    def create_s3_bucket(self, bucket_name):
        """Create S3 bucket for migration templates with proper WAF permissions"""
        try:
            # Check if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                print(f"✅ Using existing S3 bucket: {bucket_name}")
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    pass  # Bucket doesn't exist, we'll create it
                else:
                    raise
            
            # Create new bucket
            print(f"Creating S3 bucket: {bucket_name}")
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Add the correct WAF migration bucket policy
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "apiv2migration.waf-regional.amazonaws.com"
                        },
                        "Action": "s3:PutObject",
                        "Resource": f"arn:aws:s3:::{bucket_name}/AWSWAF/{self.account_id}/*"
                    }
                ]
            }
            
            self.s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            print(f"✅ Created S3 bucket with WAF migration permissions: {bucket_name}")
            print(f"Bucket Policy Resource ARN: arn:aws:s3:::{bucket_name}/AWSWAF/{self.account_id}/*")
            return True
        except Exception as e:
            print(f"\n❌ Error creating S3 bucket: {str(e)}")
            return False
    
    def generate_bucket_name(self):
        """Generate default bucket name if not provided"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"aws-waf-migration-{self.account_id}-{self.region}-{timestamp}"
    
    def query_llama(self, prompt):
        """Send prompt to Ollama/Llama2 and get response"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama2:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3  # More deterministic responses
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error querying Ollama: {str(e)}")
            return None
    

    def interpret_natural_language(self, user_input):
        normalized_input = user_input.lower()

        # Check if the input is relevant to WAF/WebACL
        if not any(keyword in normalized_input for keyword in ["waf", "webacl", "web acl", "aws waf", "acl" , "acls"]):
            return {
                "intent": "irrelevant",
                "web_acl_id": None,
                "web_acl_name": None,
                "region": None,
                "parameters": {},
                "response": "Your request does not appear to be related to AWS WAF or WebACLs."
            }
        prompt = f"""
        You are an AWS WAF migration assistant. Analyze the following user request and respond ONLY with valid JSON, and only related to WAF WEB ACLs if any other services example s3 and other dont do ,no additional text.

        User request: "{user_input}"

        Current AWS region: {self.region}

        Required JSON format:
        {{
            "intent": "list|migrate|migrate_all|help|unknown",
            "web_acl_id": "id or null",
            "web_acl_name": "name or null",
            "region": "region or null",
            "parameters": {{
                "ignore_unsupported": true|false,
                "auto_deploy": true|false,
                "create_bucket": true|false
            }},
            "response": "helpful response to user"
        }}
        Ensure there is a comma between the "parameters" object and the "response" field.
        """

        llama_response = self.query_llama(prompt)
        try:
            return json.loads(llama_response)
        except json.JSONDecodeError:
            try:
                # Extract largest JSON object from string
                json_match = re.search(r'\{.*\}', llama_response, re.DOTALL)
                if json_match:
                    raw_json = json_match.group(0)

                    # Try to sanitize missing comma errors
                    sanitized = re.sub(r'(\})(\s*"response")', r'},\2', raw_json)

                    return json.loads(sanitized)

                raise ValueError("No JSON object found")
            except Exception as e:
                print(f"Error parsing LLM response: {str(e)}")
                print(f"Original response:\n{llama_response}")
                return {
                    "intent": "unknown",
                    "web_acl_id": None,
                    "web_acl_name": None,
                    "region": None,
                    "parameters": {},
                    "response": "I couldn't understand your request. Please try again or use specific commands."
                }

    
    def list_classic_web_acls(self):
        """List all Classic WAF WebACLs in the specified region"""
        try:
            response = self.waf_regional.list_web_acls(Limit=100)
            return response.get('WebACLs', [])
        except Exception as e:
            print(f"Error listing Classic WebACLs: {str(e)}")
            return []
    
    def create_migration_stack(self, web_acl_id, bucket_name, ignore_unsupported=True):
        """Create a CloudFormation migration stack for the WebACL"""
        print(f"\nStarting migration for WebACL ID: {web_acl_id}")
        print(f"Using S3 bucket: {bucket_name}")
        print(f"Ignore unsupported types: {ignore_unsupported}")
        
        try:
            response = self.waf_regional.create_web_acl_migration_stack(
                WebACLId=web_acl_id,
                S3BucketName=bucket_name,
                IgnoreUnsupportedType=ignore_unsupported
            )
            print("\n✅ Migration template created successfully!")
            return response['S3ObjectUrl']
        except Exception as e:
            print(f"\n❌ Error creating migration template: {str(e)}")
            return None
    
    def deploy_cloudformation_stack(self, template_url, stack_name):
        """Deploy the CloudFormation stack and wait for completion"""
        print(f"\n🚀 Deploying CloudFormation stack: {stack_name}")
        print(f"Using template: {template_url}")
        
        try:
            response = self.cf_client.create_stack(
                StackName=stack_name,
                TemplateURL=template_url,
                Capabilities=['CAPABILITY_IAM'],
                OnFailure='ROLLBACK'
            )
            stack_id = response['StackId']
            print(f"\n🔹 Stack deployment initiated. Stack ID: {stack_id}")
            
            print("\n⏳ Waiting for stack creation to complete...")
            waiter = self.cf_client.get_waiter('stack_create_complete')
            waiter.wait(StackName=stack_name)
            
            stack = self.cf_client.describe_stacks(StackName=stack_name)['Stacks'][0]
            print(f"\n✅ Stack creation completed with status: {stack['StackStatus']}")
            
            if 'Outputs' in stack:
                print("\n🔹 Stack Outputs:")
                for output in stack['Outputs']:
                    print(f"{output['OutputKey']}: {output['OutputValue']}")
            
            return True
        except Exception as e:
            print(f"\n❌ Error deploying CloudFormation stack: {str(e)}")
            return False
    
    def generate_stack_name(self, web_acl_id):
        """Generate a unique stack name"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"waf-migration-{web_acl_id[:8]}-{timestamp}"
    
    def migrate_all_web_acls(self, bucket_name, ignore_unsupported=True, auto_deploy=False):
        """Migrate all Classic WebACLs in the current region"""
        web_acls = self.list_classic_web_acls()
        if not web_acls:
            print("No Classic WebACLs found to migrate in this region.")
            return
        
        print(f"\nStarting migration for {len(web_acls)} WebACLs in {self.region}")
        
        # Create bucket if needed
        if not bucket_name:
            bucket_name = self.generate_bucket_name()
            if not self.create_s3_bucket(bucket_name):
                return
        
        successful_migrations = 0
        failed_migrations = 0
        
        for acl in web_acls:
            print(f"\n=== Processing WebACL: {acl['Name']} ({acl['WebACLId']}) ===")
            
            template_url = self.create_migration_stack(
                web_acl_id=acl['WebACLId'],
                bucket_name=bucket_name,
                ignore_unsupported=ignore_unsupported
            )
            
            if not template_url:
                failed_migrations += 1
                continue
            
            stack_name = self.generate_stack_name(acl['WebACLId'])
            
            if auto_deploy:
                success = self.deploy_cloudformation_stack(
                    template_url=template_url,
                    stack_name=stack_name
                )
                
                if success:
                    successful_migrations += 1
                else:
                    failed_migrations += 1
            else:
                print("\n🔹 Migration template created but not deployed (auto_deploy=False)")
                print(f"Template URL: {template_url}")
                print(f"Stack name: {stack_name}")
                successful_migrations += 1
        
        print(f"\n🎉 Migration summary for {self.region}:")
        print(f"Successful migrations: {successful_migrations}")
        print(f"Failed migrations: {failed_migrations}")
        
        if successful_migrations > 0 and auto_deploy:
            print("\nNext steps:")
            print("1. Test the new WAFv2 WebACLs")
            print("2. Update your resources to use the new WAFv2 WebACLs")
            print("3. Delete the old Classic WebACLs after verification")
    
    def handle_natural_language_request(self, user_input):
        """Handle natural language migration request"""
        print(f"\n=== EXECUTION START ===\nProcessing: '{user_input}'")
        
        if not self.verify_credentials():
            print("❌ AWS credentials not properly configured")
            return
        
        print("\n=== STAGE 1: Interpretation ===")
        interpretation = self.interpret_natural_language(user_input)
        
        print("\n=== STAGE 2: Execution ===")
        print(f"Action: {interpretation['intent']}")
        print(f"Response: {interpretation.get('response', '[No response provided]')}")

        
        if interpretation.get('region'):
            self.region = interpretation['region']
            os.environ['AWS_DEFAULT_REGION'] = self.region  # Set the environment variable
            self.initialize_clients()
            print(f"Set region to: {self.region}")
        
        if interpretation['intent'] == "list":
            print("Executing list operation...")
            self.handle_list_request()
        elif interpretation['intent'] == "migrate_all":
            print("Executing migrate_all operation...")
            # Generate bucket name if needed
            bucket_name = None
            if interpretation['parameters'].get('create_bucket', True):
                bucket_name = self.generate_bucket_name()
                if not self.create_s3_bucket(bucket_name):
                    return

            self.migrate_all_web_acls(
                bucket_name=bucket_name,
                ignore_unsupported=interpretation['parameters'].get('ignore_unsupported', True),
                auto_deploy=interpretation['parameters'].get('auto_deploy', False)
            )
        elif interpretation['intent'] == "migrate":
            print("Executing single migration...")
            # Generate bucket name if needed
            bucket_name = None
            if interpretation['parameters'].get('create_bucket', True):
                bucket_name = self.generate_bucket_name()
                if not self.create_s3_bucket(bucket_name):
                    return
            
            # Determine WebACL ID
            web_acl_id = interpretation['web_acl_id']
            web_acl_name = interpretation['web_acl_name']
            
            if web_acl_name and not web_acl_id:
                web_acls = self.list_classic_web_acls()
                matching_acls = [acl for acl in web_acls if acl['Name'] == web_acl_name]
                
                if not matching_acls:
                    print(f"\n❌ Error: No WebACL found with name '{web_acl_name}'")
                    return
                if len(matching_acls) > 1:
                    print(f"\n❌ Error: Multiple WebACLs found with name '{web_acl_name}'")
                    return
                
                web_acl_id = matching_acls[0]['WebACLId']
            
            if not web_acl_id:
                print("\n❌ Error: Could not determine which WebACL to migrate")
                return
            
            self.handle_migration_request(
                web_acl_id=web_acl_id,
                s3_bucket=bucket_name,
                ignore_unsupported=interpretation['parameters'].get('ignore_unsupported', True),
                auto_deploy=interpretation['parameters'].get('auto_deploy', False)
            )
        else:
            print("No valid action determined")
        
        print("\n=== EXECUTION COMPLETE ===")
    
    def handle_list_request(self):
        """Handle request to list WebACLs"""
        print(f"\nListing Classic WebACLs in {self.region}:")
        web_acls = self.list_classic_web_acls()
        if not web_acls:
            print("No Classic WebACLs found in this region.")
            return

        for acl_summary in web_acls:
            web_acl_id = acl_summary['WebACLId']
            web_acl_name = acl_summary['Name']
            print(f"\nName: {web_acl_name}")
            print(f"ID: {web_acl_id}")

            try:
                # Fetch full WebACL details to get DefaultAction
                response = self.waf_regional.get_web_acl(WebACLId=web_acl_id)
                acl_details = response.get('WebACL')
                if acl_details and 'DefaultAction' in acl_details:
                    print(f"Default Action: {acl_details['DefaultAction']['Type']}")
                else:
                    print("Default Action: Not available (details could not be retrieved or key missing)")
            except ClientError as e:
                print(f"Error retrieving details for {web_acl_name} ({web_acl_id}): {e}")
            except Exception as e:
                print(f"An unexpected error occurred for {web_acl_name} ({web_acl_id}): {e}")
    
    def handle_migration_request(self, web_acl_id, s3_bucket, ignore_unsupported=True, auto_deploy=False):
        """Handle migration request"""
        if not s3_bucket:
            s3_bucket = self.generate_bucket_name()
            if not self.create_s3_bucket(s3_bucket):
                return
        
        template_url = self.create_migration_stack(
            web_acl_id=web_acl_id,
            bucket_name=s3_bucket,
            ignore_unsupported=ignore_unsupported
        )
        
        if not template_url:
            return
        
        stack_name = self.generate_stack_name(web_acl_id)
        
        if auto_deploy:
            success = self.deploy_cloudformation_stack(
                template_url=template_url,
                stack_name=stack_name
            )
            
            if success:
                print("\n🎉 Migration completed successfully!")
                print("\nNext steps:")
                print("1. Test the new WAFv2 WebACL")
                print("2. Update your resources to use the new WAFv2 WebACL")
                print("3. Delete the old Classic WebACL after verification")
        else:
            print("\n🔹 Next steps to complete the migration:")
            print(f"1. Deploy the CloudFormation stack using this template URL: {template_url}")
            print(f"2. Stack name: {stack_name}")
            print(f"3. AWS CLI command to deploy:")
            print(f"   aws cloudformation create-stack \\")
            print(f"     --stack-name {stack_name} \\")
            print(f"     --template-url {template_url} \\")
            print(f"     --capabilities CAPABILITY_IAM \\")
            print(f"     --region {self.region}")

def main():
    parser = argparse.ArgumentParser(
        description='Migrate AWS WAF Classic to WAFv2 using CloudFormation with natural language support',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--region', help='AWS region (e.g., us-west-2)')
    parser.add_argument('--list', action='store_true', help='List all Classic WebACLs')
    parser.add_argument('--web-acl-id', help='Specific WebACL ID to migrate')
    parser.add_argument('--web-acl-name', help='Specific WebACL name to migrate')
    parser.add_argument('--s3-bucket', help='S3 bucket name for migration template')
    parser.add_argument('--stack-name', help='Custom CloudFormation stack name')
    parser.add_argument('--ignore-unsupported', action='store_true', 
                       help='Ignore unsupported types during migration')
    parser.add_argument('--auto-deploy', action='store_true',
                       help='Automatically deploy the CloudFormation stack')
    parser.add_argument('--migrate-all', action='store_true',
                       help='Migrate all Classic WebACLs in the region')
    parser.add_argument('--prompt', help='Natural language prompt for migration\n'
                        'Examples:\n'
                        '  "Migrate all WAFs in us-west-2"\n'
                        '  "List all classic WAFs in us-east-1"\n'
                        '  "Help me move all web ACLs to WAFv2 with auto deploy"')
    args = parser.parse_args()
    
    assistant = WAFMigrationAssistant()
    
    if args.prompt:
        assistant.handle_natural_language_request(args.prompt)
        return
    
    # Override region if specified
    if args.region:
        assistant.region = args.region
        assistant.initialize_clients()
    
    # Original command-line functionality
    if args.list:
        assistant.handle_list_request()
        return
    
    # Handle bulk migration
    if args.migrate_all:
        assistant.migrate_all_web_acls(
            bucket_name=args.s3_bucket,
            ignore_unsupported=args.ignore_unsupported,
            auto_deploy=args.auto_deploy
        )
        return
    
    # Handle single migration
    web_acl_id = args.web_acl_id
    if args.web_acl_name and not web_acl_id:
        web_acls = assistant.list_classic_web_acls()
        matching_acls = [acl for acl in web_acls if acl['Name'] == args.web_acl_name]
        
        if not matching_acls:
            print(f"\n❌ Error: No WebACL found with name '{args.web_acl_name}'")
            return
        if len(matching_acls) > 1:
            print(f"\n❌ Error: Multiple WebACLs found with name '{args.web_acl_name}'")
            return
        
        web_acl_id = matching_acls[0]['WebACLId']
    
    if not web_acl_id and not args.migrate_all:
        print("\n❌ Error: Either --web-acl-id or --web-acl-name must be specified (or use --migrate-all)")
        return
    
    assistant.handle_migration_request(
        web_acl_id=web_acl_id,
        s3_bucket=args.s3_bucket,
        ignore_unsupported=args.ignore_unsupported,
        auto_deploy=args.auto_deploy
    )

if __name__ == "__main__":
    main()


