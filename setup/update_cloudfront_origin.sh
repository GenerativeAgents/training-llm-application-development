#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# スタック名をパラメータとして受け取る
if [ $# -eq 0 ]; then
    echo "Usage: $0 <stack-name>"
    echo "Example: $0 code-server-stack"
    exit 1
fi

STACK_NAME="$1"

echo "Getting resources from CloudFormation stack: $STACK_NAME"

# CloudFormationスタックからリソース情報を取得
STACK_RESOURCES=$(aws cloudformation describe-stack-resources --stack-name "$STACK_NAME" 2>/dev/null || true)

if [ -z "$STACK_RESOURCES" ]; then
    echo "Error: Stack '$STACK_NAME' not found"
    exit 1
fi

# Distribution IDを取得
DISTRIBUTION_ID=$(echo "$STACK_RESOURCES" | jq -r '.StackResources[] | select(.ResourceType == "AWS::CloudFront::Distribution") | .PhysicalResourceId')

if [ -z "$DISTRIBUTION_ID" ] || [ "$DISTRIBUTION_ID" == "null" ]; then
    echo "Error: CloudFront Distribution not found in stack"
    exit 1
fi

# Instance IDを取得
INSTANCE_ID=$(echo "$STACK_RESOURCES" | jq -r '.StackResources[] | select(.ResourceType == "AWS::EC2::Instance") | .PhysicalResourceId')

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "null" ]; then
    echo "Error: EC2 Instance not found in stack"
    exit 1
fi

echo "Found Distribution ID: $DISTRIBUTION_ID"
echo "Found Instance ID: $INSTANCE_ID"

# 新しいPublic DNS名を取得
NEW_DNS=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' \
  --output text)

if [ -z "$NEW_DNS" ] || [ "$NEW_DNS" == "None" ] || [ "$NEW_DNS" == "null" ]; then
    echo "Error: Unable to get Public DNS name for instance. Instance may be stopped or terminated."
    exit 1
fi

echo "New Public DNS: $NEW_DNS"

# CloudFront設定を取得
echo "Getting current CloudFront distribution configuration..."
ETAG=$(aws cloudfront get-distribution-config --id "$DISTRIBUTION_ID" --query ETag --output text)

aws cloudfront get-distribution-config --id "$DISTRIBUTION_ID" \
  --query DistributionConfig \
  | jq ".Origins.Items[0].DomainName = \"$NEW_DNS\" | .Origins.Items[0].Id = \"$NEW_DNS\" | .DefaultCacheBehavior.TargetOriginId = \"$NEW_DNS\"" \
  > /tmp/cf-config.json

# CloudFront設定を更新
echo "Updating CloudFront distribution..."
aws cloudfront update-distribution \
  --id "$DISTRIBUTION_ID" \
  --distribution-config file:///tmp/cf-config.json \
  --if-match "$ETAG" \
  > /dev/null

echo "✅ CloudFront Origin successfully updated to: $NEW_DNS"
echo ""
echo "Note: It may take around 5 minutes for the changes to propagate globally."
