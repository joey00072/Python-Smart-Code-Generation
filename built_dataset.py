import os
from tqdm import tqdm

PATH = "data/raw/python"

os.system(f"cd {PATH} &&  find . -type f ! -name '*.py' -delete")

file_list=[]
for dirpath, dirnames, filenames in os.walk(PATH):
    for name in filenames:
        file_list.append(os.path.join(dirpath,name))


with open("data/input.txt",'a',encoding='utf-8') as f:
    for name in tqdm(file_list):
        try:
            with open(name,'r',encoding='utf-8') as inp_f:
                f.write(inp_f.read())
        except Exception as e:
            print(str(e))