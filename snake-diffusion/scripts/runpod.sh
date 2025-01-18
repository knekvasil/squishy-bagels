apt-get update
apt-get install unzip
apt-get install git-lfs
git clone https://github.com/juraam/snake-diffusion.git
mv snake-diffusion/* . && rm -r snake-diffusion
pip install -r requirements.txt
bash scripts/download-dataset.sh
git clone https://huggingface.co/juramoshkov/snake-diffusion models