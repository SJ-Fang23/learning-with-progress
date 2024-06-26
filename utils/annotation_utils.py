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

