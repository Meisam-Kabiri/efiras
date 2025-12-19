# 🚀 EFIRAS Cloud Run Deployment

## Setup Google Cloud Account
```bash
# Login to the correct account for efiras-480916 project
gcloud auth login admin@efiras.com

# Set as active account
gcloud config set account admin@efiras.com

# Verify you're using the right account
gcloud config get-value account
```

## STEP 1: BUILD Docker Image
```bash
cd /home/meisam/Documents/EFIRAS/backend

# Build Docker image using Cloud Build (NO CONFIG FILE)
gcloud builds submit --tag gcr.io/efiras-480916/efiras-backend:latest --project=efiras-480916

# Result: Creates gcr.io/efiras-480916/efiras-backend:latest
```

## STEP 2: DEPLOY to Cloud Run
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

## STEP 3: WARMUP After Deployment
```bash
# Get your Cloud Run URL
SERVICE_URL=$(gcloud run services describe efiras-backend --region=europe-west4 --project=efiras-480916 --format='value(status.url)')

# Load indexes (takes 2-3 minutes)
curl -X POST $SERVICE_URL/warmup

# Test query
curl -X POST $SERVICE_URL/query-stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is compliance?"}'
```

## Deploy Existing Image (Skip Build)

If you already have a built image, skip Step 1 and deploy directly:
```bash
# Deploy your current successful build (NO REBUILD)
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
```

## Cost: FREE for 30+ years with $2000 credits!
- Memory: 4GB, CPU: 2 vCPU, Min instances: 1, Region: Europe West 4
- Cost: ~$9/month = 22+ years free

## Endpoints:
- GET /health - Health check
- POST /warmup - Load indexes (call once after deploy)
- POST /query-stream - Public queries
- POST /auth/query-stream - Authenticated queries
