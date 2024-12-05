In the files (both in `agent.py` and `model.py`), these arrays like `[1, 0, 0]`, `[0, 1, 0]`, `[0, 0, 1]`, and `[0, 0, 0]` represent the actions taken by the model while playing Snake. The same we find in the file `modified_labels2.cvs`.


Each array corresponds to a possible action of the snake:


1. [1, 0, 0] — move straight.
2. [0, 1, 0] — turn right.
3. [0, 0, 1] — turn left.
4. [0, 0, 0] — snake dies.


How it works in the context of the code:


1. Action selection:  
In the `get_action` method, the agent uses either random selection (for exploration) or the model's prediction to choose an action.
As a result, we get an array of three elements, where only one of them is equal to 1, representing one of the three actions (left, right, or straight), if all the elements are 0, the model is dead.


2. Screenshots and labels:  
When screenshots are saved, the action array (e.g., `[1, 0, 0]`) is used as the label, corresponding to the action that was taken when the screenshot was captured.