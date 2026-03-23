**UTD AI Safety Lab Coding Challenge Submission**

Hello! This is my submission for the Coding Challenge at the UTD AI Safety Lab Summer 2026 Research. My source code is in the vehicle_classifier.py file. I trained the Convolution Neural Network (CNN) by testing various values for epochs, the learning rate (lr), amount of images used (batch_size), and repetitions. The loss is 0.018 with an accuracy of 75% The model has 10 epochs and it took ~10 minutes to train to a loss of 0.018.

**Details**
+ IDE: PyCharm by Intellij
+ Library: PyTorch
+ Average time per run: 2-4 minutes (fastest time on Adam optimizer)
+ Average time to train to < 0.5 loss: 10 minutes (fastest time on Adam optimizer)
+ How I trained the model:
  + Run the model 4 times, learning from the previous set
  + The loss should decrease signficiantly. I saw the loss was usually near 1.491 on the first training run and decreased to 0.018 after training finished.
  + Plotted the highest energy (batch 150 ) loss. 

**TL;DR**
Source Code: vehicle_classifier.py
Output: vClassifier_output.txt
Loss V. Epoch Curve: Loss V Epoch Curve - Vehicle Classification - UTD AI Safety Summer 2026 Research Lab Coding Challenge Submission


Code Written By: Wafa Jailani
Email: wafajailani407@gmail.com
