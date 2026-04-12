Offline build & deploy guide

Purpose
- Build Docker image on a machine with internet access, export as tar, transfer to offline host, load and (optionally) push to internal registry, then deploy to Kubernetes.

Prerequisites (online machine)
- Git clone or copy of this repo
- Docker installed and logged in to target registry (optional)

Build and export (online machine)
1. Build and save image:
   ./ci/build_and_export.sh <image_tag> <out_tar>
   Example: ./ci/build_and_export.sh vectorsphere/uploader:latest uploader_latest.tar

2. Transfer tar to offline host (scp/sftp or other secure channel):
   scp uploader_latest.tar user@offline-host:/tmp/

Load and tag (offline host)
1. Load image tar into local Docker:
   ./ci/load_and_tag.sh /tmp/uploader_latest.tar [optional_target_registry/tag]
   - If you provide a target registry tag, the script will attempt to retag and push it.

2. If pushing to an internal registry, login and push:
   docker login <internal-registry>
   docker push <internal-registry>/vectorsphere/uploader:latest

Deploy to Kubernetes
1. Update k8s/checkpoint-uploader-deployment.yaml to use the image you loaded/pushed.
2. If using a private registry, create imagePull secret and reference it in the Deployment.
3. Apply manifests:
   kubectl apply -f k8s/checkpoint-uploader-deployment.yaml
   kubectl apply -f k8s/checkpoint-uploader-service.yaml

Verify
- kubectl get pods -l app=checkpoint-uploader
- kubectl logs deployment/checkpoint-uploader
- kubectl port-forward deployment/checkpoint-uploader 8001:8001 && curl localhost:8001/metrics

Notes
- Do NOT include credentials in images. Use Kubernetes Secrets for S3/MinIO credentials.
- If nodes cannot pull from registry, load tar on each node with `docker load -i uploader_latest.tar` (not recommended).

Support
- If you want, I can generate a sample `kubectl create secret docker-registry` command or a small script to automate node-side loading.
