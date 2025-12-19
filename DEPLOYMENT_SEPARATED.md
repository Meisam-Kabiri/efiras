# 🚀 EFIRAS Cloud Run Deployment - SEPARATED BUILD & DEPLOY

## Setup Account
```bash
# Login to the correct account
gcloud auth login admin@efiras.com
gcloud config set account admin@efiras.com
gcloud config get-value account
```
gcloud run deploy efiras-backend \
         --image=gcr.io/efiras-480916/efiras-backend:433f7ed0-4c95-453d-a416-7555aa01c926 \
         --region=europe-west4 \
         --project=efiras-480916 \
         --memory=4Gi \
         --cpu=2 \
         --min-instances=1 \
         --max-instances=10 \
         --port=8080 \
         --env-vars-file=env.yaml \
         --allow-unauthenticated

## BUILD ONLY (Docker Image)
```bash
cd /home/meisam/Documents/EFIRAS/backend

# Build Docker image using Cloud Build (NO CONFIG FILE)
gcloud builds submit --tag gcr.io/efiras-480916/efiras-backend:latest --project=efiras-480916

# Result: Creates gcr.io/efiras-480916/efiras-backend:latest
```

## DEPLOY ONLY (Cloud Run Service)  
```bash
# Deploy the built image to Cloud Run
gcloud run deploy efiras-backend \
    --image=gcr.io/efiras-480916/efiras-backend:latest \
    --region=europe-west4 \
    --project=efiras-480916 \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --port=8080 \
    --env-vars-file=env.yaml \
    --allow-unauthenticated
```

## DEPLOY EXISTING IMAGE (fastest)
```bash
# Deploy your current successful build
gcloud run deploy efiras-backend \
    --image=gcr.io/efiras-480916/efiras-backend:latest \
    --region=europe-west4 \
    --project=efiras-480916 \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --port=8080 \
    --env-vars-file=env.yaml \
    --set-secrets="/secrets/firebase-service-account.json=firebase-service-account:latest" \
    --allow-unauthenticated
```

## WARMUP AFTER DEPLOYMENT
```bash
# Get URL
SERVICE_URL=$(gcloud run services describe efiras-backend --region=europe-west4 --project=efiras-480916 --format='value(status.url)')

# Load indexes (2-3 minutes)
curl -X POST $SERVICE_URL/warmup

# Test
curl -X POST $SERVICE_URL/query-stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is compliance?"}'
```

## Key Benefits:
- **BUILD**: Only when code changes
- **DEPLOY**: Fast redeployment with existing image
- **NO CONFIG FILE**: Simple `gcloud builds submit`
- **SEPARATED**: Build once, deploy many times