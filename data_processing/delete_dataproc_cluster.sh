#!/bin/bash

# CONFIGURATION
PROJECT_ID="kalshi-456121"
REGION="us-central1"
CLUSTER_NAME="kalshi-dataproc-cluster"

# Delete the cluster
gcloud dataproc clusters delete $CLUSTER_NAME \
  --project=$PROJECT_ID \
  --region=$REGION \
  --quiet

echo "Cluster '$CLUSTER_NAME' deleted."