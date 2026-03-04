conda create -n thinkqe python=3.10
conda activate thinkqe
pip install -r requirements.txt
conda install -c conda-forge faiss-gpu openjdk=21 maven
python -m spacy download en_core_web_sm