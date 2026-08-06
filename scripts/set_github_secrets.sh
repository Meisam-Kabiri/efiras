#!/usr/bin/env bash
# Sets GitHub repository secrets (GCP_PROJECT_ID and GCP_SA_KEY) using GitHub CLI (`gh`) and Password Store (`pass`).
set -euo pipefail

PASS_PROJECT_ID_PATH="efiras/prod/gcp/GCP_PROJECT_ID"
PASS_GHA_KEY_PATH="efiras/prod/gcp/efiras-gha-key"

echo "Checking required CLI tools..."
if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI ('gh') is not installed or not in PATH."
  exit 1
fi

if ! command -v pass >/dev/null 2>&1; then
  echo "Error: Password Store ('pass') is not installed or not in PATH."
  exit 1
fi

echo "Verifying gh authentication..."
if ! gh auth status >/dev/null 2>&1; then
  echo "Error: gh CLI is not logged in. Please run 'gh auth login' first."
  exit 1
fi

echo "Fetching secrets from pass store..."

if ! pass show "$PASS_PROJECT_ID_PATH" >/dev/null 2>&1; then
  echo "Error: '$PASS_PROJECT_ID_PATH' not found in pass store!"
  echo "Please store the project ID first via:"
  echo "  echo -n '<YOUR_GCP_PROJECT_ID>' | pass insert -m $PASS_PROJECT_ID_PATH"
  exit 1
fi

PROJECT_ID="$(pass show "$PASS_PROJECT_ID_PATH" | head -n1 | tr -d '\r\n')"

if ! pass show "$PASS_GHA_KEY_PATH" >/dev/null 2>&1; then
  echo "Error: '$PASS_GHA_KEY_PATH' not found in pass store!"
  echo "Please generate and insert the service account key first."
  exit 1
fi


echo "Setting GitHub Secret: GCP_PROJECT_ID (${PROJECT_ID})..."
echo -n "$PROJECT_ID" | gh secret set GCP_PROJECT_ID

echo "Setting GitHub Secret: GCP_SA_KEY from pass ($PASS_GHA_KEY_PATH)..."
pass show "$PASS_GHA_KEY_PATH" | gh secret set GCP_SA_KEY

echo "GitHub Secrets updated successfully!"
