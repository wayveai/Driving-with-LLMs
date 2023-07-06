# Driving-with-LLMs
This is the PyTorch implementation for inference and training of the LLM-Driver 
described in:

> **Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving**
>


<p align="center">
     <img src="assets/main.png" alt="LLM-Driver">
     <br/> The system utilises object-level vector input from our driving simulator to predict explanable actions using pretrained Language Models, providing a robust and interpretable solution for autonomous driving. 

</p>

## Getting Started
### Prerequisites
- Python 3.x
- pip
- Minimum of 20GB VRAM for running evaluations
- Minimum of 40GB VRAM for training (default setting)

### ⚙ Setup
1. **Set up a virtual environment**  

    ```sh
    python3 -m venv env
    source env/bin/activate
    ```

2. **Install required dependencies**  

    ```sh
    pip install -r requirements.txt
    ```

3. **Set up WandB API key**  

    Set up your [WandB](https://wandb.ai/) API key for training and evaluation logging.

    ```sh
    export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```
    
### 💿 Dataset
- **Train data**:
Download the [vqa_train_8k.pkl](https://drive.google.com/file/d/1THDzNawbzivuGw04vFecRnSenr6elHTT/view?usp=sharing)

- **Evaluation data**:
Download the [test_1k.pkl](https://drive.google.com/file/d/1A03P9-Df-NPpQ6IrSrtbnZ-A5rGEvkWr/view?usp=sharing)

- **Re-collect DrivingQA data**:
While the training and evaluation datasets already include pre-collected DrivingQA data, we also offer a script that illustrates how to collect DrivingQA data using the OpenAI ChatGPT API. If you wish to re-collect the DrivingQA data, simplely run the following command with your OpenAI API key:
```sh
 python make_prompt_vqa.py -i input_dataset.pkl -o output_folder/ --openai_api xxxxxxxx
```
### 🏄 Evaluation

1. **Download the finetuned LLM-Driver model**

    You can find the model at this [link](https://drive.google.com/file/d/18gnHk4-lyH3ShxYtlH3eFmUoFjx11AR3/view?usp=sharing).

2. **Evaluate for Captioning and Action Prediction**

    Run the following command:

    ```sh
    python train.py \
        --mode eval \
        --val_set_size 10000 \
        --resume_from_checkpoint /path/to/model/weights/finetuned_model/ \
        --data_path /path/to/train/data/vqa_train_8k.pkl \
        --val_data_path /path/to/eval/data/test_1k.pkl \
        --eval_items caption,action \
        --vqa
    ```

3. **Evaluate for DrivingQA**

    Run the following command:

    ```sh
    python train.py \
        --mode eval \
        --val_set_size 10000 \
        --resume_from_checkpoint /path/to/model/weights/finetuned_model/ \
        --data_path /path/to/train/data/vqa_train_8k.pkl \
        --val_data_path /path/to/eval/data/test_1k.pkl \
        --eval_items vqa \
        --vqa
    ```

4. **View Results**  

    The results can be viewed on the WandB project "llm-driver".

### 🏊 Training
1. **Download pre-trained stage weights**

    Download the pre-trained weights from this [link](https://drive.google.com/file/d/1x-Jxwy-UOge-J89rHMsllPKQW-nEuSof/view?usp=sharing).

2. **Run LLM-Driver Training**

    Execute the following command to start training:

    ```sh
    python train.py \
        --mode train \
        --eval_steps 50 \
        --val_set_size 32 \
        --num_epochs 5 \
        --resume_from_checkpoint /path/to/model/weights/pretrained_model/ \
        --data_path /path/to/train/data/vqa_train_8k.pkl \
        --val_data_path /path/to/eval/data/test_1k.pkl \
        --vqa
    ```
3. **Follow the previous section for evaluation**

### 🙌 Acknowledgements

This project has drawn inspiration from the [Alpaca LoRA](https://github.com/tloen/alpaca-lora) repository. We would like to express our appreciation for their contributions to the open-source community.
