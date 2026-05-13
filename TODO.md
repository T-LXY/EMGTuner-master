## TONY
- Please create a standalone python file capturing how you pre-process your data
    - It should be able to take a user_folder passed in and return a df of the "eligible" data
    - NGL I need to think through the best way to do this for compatability with FOMAML


## Justin & Erick
- AI slop code located in `finetune_fomaml.py`
- We update `finetune_fomaml.ipynb`
- Setup PROPER ipynb file for fine tuning process
    - Set up hyper-params
    - Set up model
    - Prepare data (both for inner and outer loop)
    - Apply training loops
        - inner training loop 
        - outer training loop (actual new user we are trying to adapt to)
    - Access model accuracy
- A LOT of this will be combo of screening AI Slop code and verifying against example code
    - example code can be found here: https://interactive-maml.github.io/maml.html
    - other example code            : https://github.com/cbfinn/maml/blob/master/data_generator.py
        - NGL this one is super hard to read through, try to use at own risk


## ANYONE
- Report the following **for each sensor**:
    - `norm_mean` of signal value across **ALL GESTURES**
    - `norm_std` of signal value across **ALL GESTURES**
    - Should end with **8** `norm_mean` and 8 `norm_std` 
        - one for each respective sensor
        - Normalize to ensure equal "represen
    - Dump into a json file?
- Maybe make this a standalone file so it can calculate averages EXCLUDING fine tuning subject 
    - mimick real world stuff