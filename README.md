# Sentiment Analysis Toolkit
Sentiment Analysis Toolkit that uses BERT and has a few configurable options.
Parts of the code borrowed from [Gaurish](https://github.com/thak123/bert-twitter-sentiment).
Using the Latvian BERT model trained by [Rinalds](http://ebooks.iospress.nl/volumearticle/55524) fine-tuned on the [Latvian
Twitter Eater Corpus](https://github.com/Usprogis/Latvian-Twitter-Eater-Corpus) (LTEC) we were able to train a model with 74.33% 
percision on the LTEC evaluation set and 77.60% on the evaluation set from the
[Latvian Tweet Corpus](https://github.com/pmarcis/latvian-tweet-corpus).

### Data Format
- CSV format with a label that you want to predict and the text.
- For example, to classify 3 sentiment classes you could use 0 - neutral; 1 - positive; 2 - negative like shown in this example:
```csv
label,text
1,"@maljorka Hehe, man tad labāk garšo bez nekā, nevis šādi. :D Ai, gaumes ir tik atšķirīgas."
0,@IngaStirna Ābolu šarlote.
2,Mēs ar viņu varējām sarunāties tikai caur logu un pusdienu vietā apēdām bulciņas mašīnā. Nav ok.
```


### Usage
Fill in configuration details like training/development/evaluation files, paths to the BERT model and 
where you want to save sentiment classification models in `config.py`
- #### Fine-tune BERT/mBERT (Optional)
	- Run `run_mlm.py` from [this repo](https://github.com/huggingface/transformers/tree/master/examples/language-modeling)
	on your own data.
	- It may also be useful to find a pre-trained BERT model in your language of choice and use of fine-tune that.

- #### Training
	- Run `train.py`

- #### Tuning (domain adaptation)
	- Change the training data set in `config.py` to either only your in-domain data or perhaps a 1:1 mix of 
	in-domain and out-of-domain data.
	- Run `train.py --tune`
	- You may want to lower the learning rate, change dropout or play with other parameters - 
	use `grid_search.sh` to go over combinations.

- #### Prediction
	- Run `predict.py --input input-file.csv --output output-file.csv`
	- The input file should be only texts - one per row. The output will be label, text.

- #### Grid Search
	- Iterate over several combinations of hyperparameters
	
	
### Parameters
The following are parameters for `train.py`

| Parameter | Description                   					  | Example Value 			 	| Default Value  |
|:----------|:----------------------------------------------------|:----------------------------|:---------------|
| --tune    | Loads the model in `MODEL_PATH` for fine-tuning. 	  |                            	|                |
| --lr      | Learning rate.                                 	  | 0.00005        	            | 0.00001        |
| --drop    | Dropout.                                      	  | 0.1            	            | 0.3            |
| --save    | Save and evaluate after X examples.                 | 3000            	        | 15000          |
| --estop   | Stop training after model has not improved X times. | 10            	            | 5              |

The following are parameters for `predict.py`

| Parameter    | Description                   					  | Example Value 			 	| Default Value     |
|:-------------|:-------------------------------------------------|:----------------------------|:------------------|
| --input      | Input file for prediction - one text per line.   | 'in.csv'                  	| `EVAL_PROC`       |
| --output     | Input file for prediction - one text per line.   | 'out.csv'        	        | 'predictions.csv' |
| --model_path | Model to use for prediction.                     | 'best_model.bin'            | `MODEL_PATH`      |
