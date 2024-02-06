# Translate from reactions and reagents to products.
Modify an English-French translation transformer to a chemical reaction prediction model.

## to do 
1. Replace the greedy search with beam search
2. Test the logic of training
   ###
   2.1. pretraining tasks
   
   2.2. replace the input of decoder with different sequence
   
3. (Completed) wrap .ipynb to .py file and generate a config file for convenient use.
4. (Failed) find a way to train the model in distributed mechines to speed up.
5. Check the Teacher Foring and apply Teacher Foring Ratio.
   ###
      5.1 Found Exposure Bias problems: When generating, once the decode_input is not target sequence, model outputs repeated tokens out of control.
   While target sequence is input as the decode_input, model outputs kinda normal tokens. This is the main problem caused by Teacher Foring strategy.
   The strategy accelerates training progress but makes generative output, without target sequence, worse!!!
  
     
