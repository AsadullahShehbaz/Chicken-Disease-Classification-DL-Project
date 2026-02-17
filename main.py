from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier.pipeline.stage_03_training import TrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline
from cnnClassifier import logger 

STAGE_NAME = "Data Ingestion stage"

try:
        # Log stage start
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<")

        # Create pipeline object and execute
        obj = DataIngestionTrainingPipeline()
        # obj.main()

        # Log stage completion
        logger.info(f">>>>> Stage {STAGE_NAME} ended <<<<<<")

except Exception as e:
        # Log detailed exception traceback
        logger.exception(e)
        raise

STAGE_NAME = "Prepare Base Model"

try:
        # Log stage start
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<")

        # Create pipeline object and execute
        obj = PrepareBaseModelPipeline()
        obj.main()

        # Log stage completion
        logger.info(f">>>>> Stage {STAGE_NAME} ended <<<<<<")

except Exception as e:
        # Log detailed exception traceback
        logger.exception(e)
        raise

STAGE_NAME = "Training"
try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} ended <<<<<<")

except Exception as e:
        logger.exception(e)
        raise

STAGE_NAME = "Model Evaluation"
try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} ended <<<<<<")

except Exception as e:
        logger.exception(e)
        raise

