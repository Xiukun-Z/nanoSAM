from mutliWebUI import WebUI
import os

def main():

    model_folder = "./model"
    os.makedirs(model_folder, exist_ok=True)
    if os.path.exists('./model/sam_vit_h_4b8939.pth'):
        WebUI()
    else:
        print('not found model')
        os.system('pause')

if __name__ == "__main__":
    main()
