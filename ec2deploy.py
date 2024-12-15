import boto3

def deploy_to_ec2(account_id, region):
    ec2 = boto3.client('ec2', region_name=region)

    # Define parameters for creating an EC2 instance
    image_id = 'ami-0ee4d25a330ac1474'  # Amazon Linux 2 AMI
    instance_type = 't3a.medium'
    key_name = 'Bootstrap-CCoE-Test'
    subnet_id = 'subnet-09756a29ad1c3d7e1'  # Replace with your subnet ID
    security_group = 'sg-0acb3e10f2cd85f5f'  # Replace with your security group ID

    user_data = '''#!/bin/bash
    yum update -y
    yum install -y httpd
    systemctl start httpd
    systemctl enable httpd
    echo "Hello, World!" > /var/www/html/index.html
    '''

    # Launch the EC2 instance
    response = ec2.run_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        KeyName=key_name,
        UserData=user_data,
        MinCount=1,
        MaxCount=1,
        SubnetId=subnet_id,
        SecurityGroupIds=[security_group]
    )

    # Wait for the instance to start up
    instance_id = response['Instances'][0]['InstanceId']
    print('Instance ID:', instance_id)

    # Get the EC2 instance public IP address
    response = ec2.describe_instances(InstanceIds=[instance_id])
    ip_address = response['Reservations'][0]['Instances'][0]['PrivateIpAddress']
    print('Private IP address:', ip_address)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: ec2deploy.py <account_id> <region>")
        sys.exit(1)

    account_id = sys.argv[1]
    region = sys.argv[2]

    # Call the deployment function
    deploy_to_ec2(account_id, region)
