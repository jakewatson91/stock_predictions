from google.cloud import storage

BUCKET_NAME = "kalshi-data-lake"

def upload_to_gcs(local_path, gcs_path):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded: gs://{BUCKET_NAME}/{gcs_path}")