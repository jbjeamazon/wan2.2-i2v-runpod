"""
Object-storage helpers for returning generated videos as URLs instead of
inline base64.

RunPod caps job payloads at 10 MB on /run and 20 MB on /runsync, and base64
inflates bytes by ~33%, so a 720p clip does not fit in the response body.
When S3 credentials are present we upload the MP4 and hand back a presigned
GET URL; otherwise the caller falls back to inline base64.

Works with any S3-compatible store (AWS S3, Cloudflare R2, Backblaze B2,
MinIO) by pointing S3_ENDPOINT_URL at the provider.
"""

import os
import uuid

S3_BUCKET = os.environ.get("S3_BUCKET") or None
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL") or None
S3_REGION = os.environ.get("S3_REGION", "auto")
# Cloudflare R2 rejects requests carrying an ACL header, so allow opting out.
S3_SEND_ACL = os.environ.get("S3_SEND_ACL", "false").lower() == "true"
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID") or None
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY") or None
S3_PUBLIC_URL_BASE = os.environ.get("S3_PUBLIC_URL_BASE") or None
# Presigned URLs are unauthenticated bearer links: anyone holding one can fetch
# the video until it expires. Keep the window short.
S3_URL_EXPIRY = int(os.environ.get("S3_URL_EXPIRY", "3600"))

_client = None


def is_configured() -> bool:
    """True when enough S3 settings are present to attempt an upload."""
    return bool(S3_BUCKET and S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY)


def _get_client():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config

        _client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            region_name=S3_REGION,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
    return _client


def upload_video(local_path: str, key_prefix: str = "videos") -> str:
    """Upload an MP4 and return a URL the caller can fetch it from."""
    key = f"{key_prefix}/{uuid.uuid4().hex}.mp4"
    client = _get_client()

    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=f,
            ContentType="video/mp4",
            # Explicitly private: presigned URLs still work, but a
            # mis-scoped bucket policy alone will not expose the object.
            # R2 and some MinIO configs reject ACLs, so this is best-effort.
            **({"ACL": "private"} if S3_SEND_ACL else {}),
        )

    # S3_PUBLIC_URL_BASE serves the object from a CDN / public bucket instead.
    # That URL never expires and is readable by anyone who learns it, so only
    # set this if the videos are meant to be shareable. For a private endpoint,
    # leave it unset and keep the bucket's public access blocked.
    if S3_PUBLIC_URL_BASE:
        return f"{S3_PUBLIC_URL_BASE.rstrip('/')}/{key}"

    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=S3_URL_EXPIRY,
    )
