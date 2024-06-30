# MLOps-Dicoding-Projects
list of projects from dicoding's MLOps class

## Student Performance Classification
create a simple ML pipeline using TensorFlow Extended (TFX)

this dataset predicts student's grade class using the student-performance-dataset

criteria:
- free dataset
- use all the components from TFX (ExampleGen, StatisticGen, SchemaGen, ExampleValidator, Transform, Trainer, Resolver, Evaluator, Pusher)
- all the TFX components must be run by using *interactive_context* and saved in a  **<username_dicoding>-pipeline** folder
- use Tuner component (optional)
- deploy model using docker (optional)
- create a test notebook to get prediction (optional)

### TF-Serving using docker:

Create docker image

```bash
docker build -t student-performance-tf-serving .
```

Run docker image and define port

```bash
docker run -p 8080:8501 student-performance-tf-serving
```

## Hotel Reservations Predictions
create a simple ML pipeline using TensorFlow Extended (TFX) and deploy it to cloud

this dataset predicts whether a customer will cancel or not cancel a hotel reservations using the Hotel Reservation dataset

criteria:
- free dataset
- use all the components from TFX (ExampleGen, StatisticGen, SchemaGen, ExampleValidator, Transform, Trainer, Resolver, Evaluator, Pusher)
- all the TFX components must be run by using Pipeline Orchestrator named **Apache Beam** and saved in a  **<username_dicoding>-pipeline** folder
- deploy the machine learning system with cloud computations (heroku/railway)
- monitor the machine learning system using prometheus
- use Tuner component (optional)
- apply clean code principal (optional)
- add notebook file to test and run prediction request to the system in cloud (optional)