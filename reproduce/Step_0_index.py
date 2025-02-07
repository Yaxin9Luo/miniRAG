# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import gpt_4o_mini_complete, hf_model_complete, hf_embedding,openai_embedding
from minirag.utils import EmbeddingFunc
from transformers import AutoModel,AutoTokenizer
from datetime import datetime

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse
def get_args(): 
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument('--model', type=str, default='PHI')
    parser.add_argument('--outputpath', type=str, default='./logs/arc_output.csv')
    parser.add_argument('--workingdir', type=str, default='./ARC-Challenge')
    parser.add_argument('--datapath', type=str, default='./dataset/ARC-Challenge/data/')
    parser.add_argument('--querypath', type=str, default='./dataset/ARC-Challenge/qa/query_set.json')
    args = parser.parse_args()
    return args

args = get_args()


if args.model == 'PHI':
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
elif args.model == 'GLM':
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == 'MiniCPM':
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == 'qwen':
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    # llm_model_func=gpt_4o_mini_complete,
    llm_model_max_token_size=200,
    llm_model_name= LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embedding(
            texts, 
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL)
        )
    ),
)


#Now indexing
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

# # 修改为直接读取ARC知识库文件
# with open(os.path.join(DATA_PATH, 'arc_knowledge.txt')) as f:
#     rag.insert(f.read())
WEEK_LIST = find_txt_files(DATA_PATH)
for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")
    with open(WEEK) as f:
        rag.insert(f.read())
