# Usage: source /opt/real/r6/r5t/env.sh TREE   (TREE = ref | r3)
# Selects which libctranslate2 the (shared, editable) python package loads and clears
# every CT2_* flag so each run states its flags explicitly.
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/venv/bin/activate
export CTRANSLATE2_ROOT=$HOME/w/$1/install
export LD_LIBRARY_PATH=$HOME/w/$1/install/lib:/usr/local/cuda-12.8/lib64
for v in $(env | grep -o '^CT2_[A-Z0-9_]*'); do unset "$v"; done
cd /opt/real/r6/r5t
