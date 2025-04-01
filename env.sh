module load gcc/11.4.0  openmpi/4.1.4 python/3.11.4

if [ ! -d "ENV" ]; then
    python -m venv ENV
    source ENV/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install --upgrade vllm
    pip install bitsandbytes>=0.45.3
else
    source ENV/bin/activate
fi
