# Kalman-Filtering-on-Satellite-Orbits
Satellite Orbit data obtained from sensors is noisy and unreliable. Kalman filtering is helpful to smooth the noisy data as compared to a mathematical model. This python code implements the filter for the International Space Station. The project was programmed using Python language, and there are three files in the source code. One holds the mathematical model, the other holds the sensor data of the satellite import and adjustment for the project. The final file has the entire Extended Kalman Filter process and all the code together as well (works without the other two codes because it contains them).

The code does the perfect mathematical model of the orbit using the Runge-Kutta model. This finds the orbit of the satellite unaffected by unpredictable environmental effects. However, applied on it is atmospheric drag effects abd J2 perturbation effects to include some imperfections.

I also downloaded ISS orbit data from sensors that are filtered. I used the Skyfield API to import the data at any specific time I want. Since I wanted to demostrate the Extended Kalman Filter effects, I manually added noise to it using a random Gaussian generation method. 

Finally, I brought it all together by using the two types of data and implementing EKF to filter the bad sensor data by punishing the high error it measures in the filter. The result is a relatively well filtered orbit that also attests the environmental effects that cannot be captured by the mathematical model alone. The results of the filter are in the image below:

![6e1f41e3-1aba-407c-b182-39182d42f864](https://github.com/user-attachments/assets/b9ebe52f-3c49-434f-a87f-7265dceff6f7)
