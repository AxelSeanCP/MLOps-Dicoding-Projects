# MLOps-Dicoding-Projects
list of projects from dicoding's MLOps class

## First Projects
create a simple ML pipeline using TensorFlow Extended (TFX)

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