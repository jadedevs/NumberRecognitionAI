# Number Recognition AI
A convolutional neural network that can read handwritten numerals and accurately predict what they are.
### Please do not expect this to be perfect, it has undergone minimal updates/patches and will occasionally falsely predict the number given.

---

## Usage
### CLI Arguments
```
usage: python -m make_predicton [image file path]
```

---

## How to use logs
### You will need TensorBoard installed for this step. 
If you don't have it yet, you can install it by running `pip install tensorboard` in your terminal.

- Once you have TensorBoard installed, run your prediction, and then open up a terminal window and run `tensorboard --logdir=[log path]`
- The prediction that you just ran will have produced log files which will be in the `logs` directory, included in the repo. Use this in place of `[log path]`
- Open up a browser and go to `localhost:6006` to view your logs.
