from github import Github
import os
from datetime import datetime,timedelta

# Use WSL if you are using it on windows

with open("api_key.txt") as f:
    # save github api key in api.txt file
    API_KEY=f.read()


PATH = './data/raw/python'


def make_dir(DIR):
        if not os.path.isdir(DIR):
                os.mkdir(DIR)

def delete_git(DIR):
        for f in os.listdir(DIR):
                loc = os.path.join(DIR,f)
                if os.path.isdir(os.path.join(loc,'.git')):
                        os.system(f"rm -rf {os.path.join(DIR,'.git')}")


def delete_non_py_file(loc):
        os.system(f"cd {loc} && find . -type f ! -name '*.py' -delete")


def clone_repo(item,loc):
        os.system(f"cd {loc} && git clone {item.clone_url}")
        delete_git(loc)
        delete_non_py_file(loc)

K=600
def main():
        g = Github(API_KEY)


        for i in range(1,1000):
                start= str(datetime.today()-timedelta(K+i+1))[:-16]
                end  = str(datetime.today()-timedelta(K+i))[:-16]

                query = f'numpy tutorial language:python created:{start}..{end}'

                result =  g.search_repositories(query)

                lst = []

                for item in result:
                        print(item)
                        loc = os.path.join(PATH,item.owner.login)
                        make_dir(loc)
                        clone_repo(item,loc)


                with open('logs.txt','w') as f:
                        f.write(f"Downloading days:{i}")



if __name__ == '__main__':
        main()