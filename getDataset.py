import kagglehub

# Download latest version
path = kagglehub.dataset_download("jerryqu/reddit-conversations")

print("Path to dataset files:", path)