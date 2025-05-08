#!/bin/bash

# CONFIGURATION
PROJECT_ID="kalshi-456121"
REGION="us-central1"
CLUSTER_NAME="kalshi-dataproc-cluster"

# Create the cluster
gcloud dataproc clusters create $CLUSTER_NAME \
  --project=$PROJECT_ID \
  --region=$REGION \
  --single-node \
  --enable-component-gateway \
  --optional-components=JUPYTER \
  --image-version=2.1-debian11 \
  --max-idle=30m \
  --quiet

echo "Cluster '$CLUSTER_NAME' created."