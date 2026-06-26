from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import AmlCompute, Data, Environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id="616d69b9-f6bd-4360-b844-35bceca1a09f",
    resource_group_name="default_resource_group",
    workspace_name="usedcars-pricing-workspace",
)

cpu_compute_target = "cpu-cluster"
try:
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
except Exception:
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        type="amlcompute",
        size="Standard_DS11_v2",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=180,
        tier="Dedicated",
    )
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster).result()

local_data_path = "data/used_cars.csv"
data_asset = Data(
    path=local_data_path,
    type=AssetTypes.URI_FILE,
    description="A dataset of used cars for price prediction",
    name="used-cars-data"
)
data_asset = ml_client.data.create_or_update(data_asset)

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file={
        "name": "sklearn-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8",
            "pip=21.2.4",
            "scikit-learn=0.23.2",
            "scipy=1.7.1",
            {"pip": ["mlflow==2.8.1", "azureml-mlflow==1.51.0", "azureml-inference-server-http", "azureml-core==1.49.0", "cloudpickle==1.6.0"]}
        ]
    },
    name="machine_learning_E2E",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env_docker_conda)

data_prep_job = command(
    inputs=dict(
        data=Input(type=AssetTypes.URI_FILE, path=data_asset.id),
        test_train_ratio=0.2,
    ),
    outputs=dict(
        train_data=Output(type=AssetTypes.URI_FOLDER),
        test_data=Output(type=AssetTypes.URI_FOLDER),
    ),
    code="./mlops",
    command="python prep.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}}",
    environment="machine_learning_E2E@latest",
    compute=cpu_compute_target,
    display_name="data_prep_job",
)

train_job = command(
    inputs=dict(
        train_data=Input(type=AssetTypes.URI_FOLDER),
        test_data=Input(type=AssetTypes.URI_FOLDER),
        n_estimators=100,
        max_depth=10,
    ),
    outputs=dict(
        model_output=Output(type=AssetTypes.URI_FOLDER),
    ),
    code="./mlops",
    command="python train.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} --n_estimators ${{inputs.n_estimators}} --max_depth ${{inputs.max_depth}} --model_output ${{outputs.model_output}}",
    environment="machine_learning_E2E@latest",
    compute=cpu_compute_target,
    display_name="train_job",
)

register_job = command(
    inputs=dict(
        model=Input(type=AssetTypes.URI_FOLDER),
    ),
    code="./mlops",
    command="python model_register.py --model ${{inputs.model}}",
    environment="machine_learning_E2E@latest",
    compute=cpu_compute_target,
    display_name="register_job",
)

@pipeline(
    compute=cpu_compute_target,
    description="End-to-end pipeline for used car price prediction",
)
def used_cars_pricing_pipeline(pipeline_input_data):
    prep_step = data_prep_job(
        data=pipeline_input_data,
        test_train_ratio=0.2,
    )
    train_step = train_job(
        train_data=prep_step.outputs.train_data,
        test_data=prep_step.outputs.test_data,
        n_estimators=100,
        max_depth=10,
    )
    register_step = register_job(
        model=train_step.outputs.model_output,
    )
    return {
        "pipeline_train_data": prep_step.outputs.train_data,
        "pipeline_test_data": prep_step.outputs.test_data,
        "pipeline_model_output": train_step.outputs.model_output,
    }

pipeline_job = used_cars_pricing_pipeline(
    pipeline_input_data=Input(type=AssetTypes.URI_FILE, path=f"azureml:used-cars-data:{data_asset.version}")
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="used_cars_pricing_pipeline"
)

print(f"Pipeline submitted: {pipeline_job.name}")
print(f"Status: {pipeline_job.status}")
