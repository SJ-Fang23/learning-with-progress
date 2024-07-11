# utils for progress annotation
import json
import os

def write_to_json(data:list[dict], 
                  file_path:str):
    '''
    Write data to json file
    input: data: list of dict
           file_path: str,relative to progress_data folder
    '''
    # get the absolute path
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_path,"progress_data",file_path)

    # if the file does not exist, create it
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # write data to json file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def read_from_json(file_path:str) -> list[dict]:
    '''
    Read data from json file
    input: file_path: str, relative to progress_data folder
    output: data: list of dict
    '''
    # get the absolute path
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_path,"progress_data",file_path)

    # read data from json file
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_all_json(file_path:str = None):
    '''
    Read all json files in the folder
    input: file_path: str, relative to progress_data folder, default is None
    '''
    # get the absolute path
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if file_path is None:
        file_path = "progress_data"
    file_path = os.path.join(project_path, file_path)
    # read all json files in the folder
    data = dict()
    for file in os.listdir(file_path):
       if file.endswith(".json"):
            file_name = file.split(".")[0]
            with open(os.path.join(file_path, file), 'r') as f:
                data[file_name] = json.load(f)
    return data
        

if __name__ == "__main__":
   
    print(read_all_json())