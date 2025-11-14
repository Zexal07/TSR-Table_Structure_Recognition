#Table Structure Recognition (TSR) using Vision Transformers
This project implements a **Table Structure Recognition (TSR)** model using a **Vision Transformer (ViT)** encoder and a BERT-based decoder. The primary goal is to take an image of a table as input and generate its corresponding structural representation as a sequence of HTML tokens.

The model is trained using the TableBank dataset, a large-scale collection of images and their corresponding HTML annotations.

#Features
**Model**: Utilizes the VisionEncoderDecoderModel from the Hugging Face transformers library.

**Encoder**: A pre-trained Vision Transformer (google/vit-base-patch16-224-in21k) processes the input table images.

**Decoder**: A pre-trained BERT model (bert-base-uncased) is configured as a decoder to generate the HTML token sequence.

**Training**: Implements efficient training and evaluation using the Seq2SeqTrainer.

**Dataset**: A custom torch.utils.data.Dataset class is provided to handle the TableBank Recognition dataset format.

**Evaluation**: Model performance is measured using the BLEU score (via sacrebleu) to compare the generated HTML with the ground-truth labels.

#Project Directory Structure
The script is designed to work with the TableBank dataset. The code specifically uses the Recognition subdirectory.
'''
TableBank/
|- Detection/
|  |- Annotations
|  |  |- tablebank_latex_test.json
|  |  |- tablebank_latex_train.json
|  |  |- tablebank_latex_val.json
|  |  |- tablebank_word_test.json
|  |  |- tablebank_word_train.json
|  |  |- tablebank_word_val.json
|  |- Images
|  |  |- %20%20%202013_2.jpg
|  |  |- ...
|- Recognition/
|  |- Annotations
|  |  |- all_test.txt
|  |  |- all_train.txt
|  |  |- all_val.txt
|  |  |- latex (test, train, val) .txt
|  |  |- src_all (test, train, val) .txt
|  |  |- src_latex (test, train, val) .txt
|  |  |- src_word (test, train, val) .txt
|  |  |- tgt_all (test, train, val) .txt
|  |  |- tgt_latex (test, train, val) .txt
|  |  |- tgt_word (test, train, val) .txt
|  |  |- word (test, train, val) .txt
|  |- Images
|  |  |- %20%d8%af.%20%d9%88%d8%a6%d8%a7%d9%85_0_3.png
|  |  |- ...
'''
**Dataset**
This model relies on the files within the Recognition/ directory.
'''
Recognition/Images/: Contains all the raw .png or .jpg image files.
'''
'''
Recognition/Annotations/: Contains the ground-truth text files.
'''
The **rc-all_[split].txt** files (e.g., src-all_train.txt) are expected to contain the list of image filenames.

The **tgt-all_[split].txt** files (e.g., tgt-all_train.txt) are expected to contain the corresponding HTML structure string for each image, line by line.

#Requirements
The project requires the following main Python libraries. You can install them via pip:

'''
pip install torch transformers pillow sacrebleu
'''
#Usage
To train the model, ensure your dataset is organized as described above and update the paths in the Config class.

Configure Paths: Modify the Config.ROOT_DIR variable in the script to point to your TableBank/Recognition directory.

Run Training: Execute the Python script.

'''
python train.py
'''
The script will:

Load the pre-trained ViT image processor and BERT tokenizer.

Initialize the **VisionEncoderDecoderModel**.

Load the TableBankDataset for the training and validation splits.

Initialize the Seq2SeqTrainer with the specified training arguments.

Begin training, validating at the end of each epoch and logging progress.

Save the best-performing model (based on eval_bleu) to the Config.MODEL_OUTPUT_DIR.

#Configuration
All training and model parameters can be adjusted in the Config class at the beginning of the script:

*Paths*:

-ROOT_DIR: Path to the TableBank/Recognition folder.

-MODEL_OUTPUT_DIR: Directory to save the final trained model.

-CHECKPOINT_DIR: Directory to save intermediate training checkpoints.

*Model Checkpoints*:

-ENCODER_MODEL: The Hugging Face checkpoint for the ViT encoder.

-DECODER_MODEL: The Hugging Face checkpoint for the BERT decoder.

*Training Hyperparameters*:

-NUM_EPOCHS: Total number of training epochs.

-BATCH_SIZE: Batch size per device.

-LEARNING_RATE: The learning rate for the optimizer.

-GRAD_ACCUMULATION_STEPS: Number of steps to accumulate gradients before an optimizer step, effectively increasing the batch size.

*Data Subsetting*:

TRAIN_SIZE, VAL_SIZE, TEST_SIZE: Allows you to limit the number of samples used from each split for faster iteration or debugging.
