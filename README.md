**UTD AI Safety Lab Coding Challenge Submission**

# Vehicle Classification CNN
UTD AI Safety Lab Summer 2026 Coding Challenge | Wafa Jailani

## Results
- Final loss:     0.018
- Accuracy:       75%
- Classes:        Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, Van

## Files
- `vehicle_classifier.py` — source code
- `vClassifier_output.txt` — full training output
- `Loss_vs_Epoch_Curve.pdf` — loss curve graph

## Model Details
- Architecture:  CNN (2 conv layers, 3 fully connected layers)
- Optimizer:     Adam  
- Learning rate: 0.001
- Batch size:    128
- Epochs:        10 per training session
- Dataset:       26,378 images split 80/20 train/test
- 
## How to Run
1. Set dataset path in `vehicle_classifier.py`
2. Run `vehicle_classifier.py`
3. Model saves automatically to `vehicle_classifier.pth`

## Author
Wafa Jailani | wafajailani407@gmail.com | @wafajailani
