**UTD AI Safety Lab Coding Challenge Submission**

# Vehicle Classification CNN
UTD AI Safety Lab Summer 2026 Coding Challenge | Wafa Jailani

## Results
- Final loss:     0.018
- Accuracy:       75%
- Eight Classes:        Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, Van

## Files
- `vehicle_classifier.py` — source code
- `vClassifier_output.txt` — full training output
- `Loss_vs_Epoch_Curve.pdf` — loss curve graph

## Model Details
- Convolutional Neural Network (CNN)
- Optimizer:     Adam - faster than SGD optimizer
- Learning rate: 0.001
- Batch size:    128
- Epochs:        10 per training session
  
## Run
1. Set dataset path in `vehicle_classifier.py`, which is the vehicle_classification zip file
2. Run `vehicle_classifier.py`
3. Model saves automatically to `vehicle_classifier.pth`

## Author
Wafa Jailani | wafajailani407@gmail.com | @wafajailani
