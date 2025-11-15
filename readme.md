## Installation

Install env

```
conda create -n nanoSAM python=3.11
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Install Segment Anything Model (SAM)
```
git clone https://github.com/facebookresearch/segment-anything
```
### Download the SAM weights

Download the SAM weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)  
and save the file to `./model/sam_vit_h_4b8939.pth`.

## Run

```
python main.py
```