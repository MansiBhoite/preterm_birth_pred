# preterm_birth_pred
This project aims to predict the risk of preterm birth in pregnant women using real-time health data collected through IoT sensors and analyzed using a Machine Learning model. 
The system uses GSR (Galvanic Skin Response), Pulse, and Motion sensors connected to a Raspberry Pi, which collects physiological data. The data is processed and passed to an LSTM (Long Short-Term Memory) model, which predicts the likelihood of preterm birth based on patterns in the input.
This helps in early detection of risks so that medical help can be provided on time, especially in rural or low-resource areas.

Tech Stack & Tools Used:
IoT Hardware: Raspberry Pi, GSR Sensor, Pulse Sensor, Motion Sensor, ADS1115

Programming Languages: Python

Machine Learning Model: LSTM (Long Short-Term Memory)

Data Handling: CSV dataset of pregnant women with sensor values

Communication: I2C Protocol

Key Features:
Collects real-time physiological data through IoT sensors

Uses ML model (LSTM) to analyze and predict preterm birth risks

Displays prediction results on Raspberry Pi output

Can be extended to alert healthcare workers remotely
