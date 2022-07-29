# Vehicle Color Detection

First of all, here is a report about the project

* [project_report.pptx](https://github.com/alireza-kermaninejad/DeepLearning-WorkShop-Part3/files/9222287/project_report.pptx)

# How to use the API App

### Clone the project
```bash
git lfs clone https://github.com/alireza-kermaninejad/DeepLearning-WorkShop-Part3.git
```

* First go to the app project in parents folder (vehicle-coloe-detection)

* And open windows command prompt(cmd) in that directory

* you can use this command:
```bash
cd write-the-vcd-path
```

Then you can use methodes below to use the app

## /// The first method
### 1- Run TensorFlow Serving with executing command below:

```bash
docker run -p 8501:8501 --name=vcd_models -v "C:\Users/Peyton/Desktop/vehicle_color_detection/app/vcd_models:/models/vcd_models/1" -e MODEL_NAME=vcd_models tensorflow/serving
```
### 2- After that Run the app.py with this comand:
```bash
python app.py
```

!!! Be sure you are in correct directory

### 3- Then you can see the result on:
```bash
http://localhost:5000
```

## /// The second method
### 1- Run the app by Building Docker

```bash
docker compose build up -d
```
!!! Be sure you are in correct directory

### 2- Go to the url below to use the app:
```bash
http://localhost:5000
```




