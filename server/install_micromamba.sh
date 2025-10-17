# install_micromamba.sh 

# install on Linux Intel (x86_64):
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
# move the binary to the 
mkdir -p micromamba/bin 
mv bin/micromamba micromamba/bin
# move micromamba to PATH
export PATH="$HOME/micromamba/bin:$PATH"
echo "export PATH="$HOME/micromamba/bin:$PATH"" >> ~/.bashrc
# test micromamba
echo "Micromamba version: $(micromamba --version)"
# Initialize micromamba for bash
micromamba shell init --shell bash --root-prefix=~/micromamba
source ~/.bashrc
# test micromamba again
micromamba activate base 
micromamba deactivate
micromamba env list 
echo "Micromamba installation complete."
