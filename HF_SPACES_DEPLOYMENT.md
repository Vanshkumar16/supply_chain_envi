# Deployment to Hugging Face Spaces

## Option 1: Connect GitHub Repository (Recommended)
1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Set:
   - **Owner**: Your username
   - **Space name**: `supply-chain-env`
   - **License**: MIT
   - **Space SDK**: Docker
4. Select **Connect a repo from GitHub**
5. Choose `Vanshkumar16/supply_chain_envi`
6. Add secrets in Space Settings:
   - `HF_TOKEN`: Your Hugging Face API token

## Option 2: Direct Push via Hugging Face CLI
```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Create space repo
huggingface-cli repo create supply-chain-env --type space --space-sdk docker

# Push to space
cd /path/to/supply_chain_env
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/supply-chain-env
git push space master
```

## Environment Variables Required
- `HF_TOKEN`: Your Hugging Face API token (for LLM inference via router.huggingface.co)

## API Endpoints
Once deployed, your Space will expose:
- `GET /reset?task_id=<id>&difficulty=<level>` - Initialize episode
- `POST /step` - Take action step
- `POST /grade` - Score episode
- `GET /state` - Get current state

## Testing the Deployment
```bash
curl https://YOUR_USERNAME-supply-chain-env.hf.space/reset?task_id=0&difficulty=easy
```

## Logs & Debugging
Check the **Logs** tab in your Space settings to see runtime output and errors.
