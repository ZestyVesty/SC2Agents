# Acknowledgement
The original author of this repository belongs to Ray Heberer (https://github.com/rayheberer).

The original code can be found at git repository: https://github.com/rayheberer/SC2Agents

# Parts of code that has been added/modified
Note: comments starting with FIXME is used to make the comment to stand out from all the other comments tha
has been made.
* deepq.py
    
    line 31-56: used to store different models to different directory
    
* dueling_DQN.py
    
    line 31-56: used to store different models to different directory
    
    Note: this file is a copy of deepq.py, with some slight changes to names of classes etc.
    
* value_estimators_dueling.py
    
    line 167-190: added two fully neural network to calculate the state value and the advantage of
    each action. Then calculates the Q value and pass it on using the original code. 
    
    Note: this file is a copy of value_estimators.py