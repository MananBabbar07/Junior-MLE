import functools
print = functools.partial(print, flush=True)

import sys
import os
import io
import pandas as pd
import re
import ast

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.agents.planner import Planner
from src.agents.coder import Coder
from src.tools.executor import Executor
from src.agents.critic import Critic
from src.agents.modeler import Modeler
from src.agents.tuner import Tuner


def clean_ai_markdown(text):
    pattern = r"```(?:python)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def main():
    DATA_PATH = "data/raw/train.csv"

    print("Initializing Agentic AutoML...")

    df_sample = pd.read_csv(DATA_PATH, nrows=5)
    schema = df_sample.dtypes.to_string()
    sample_data = df_sample.head(3).to_string()

    planner = Planner()

    print("Auditing dataset...")
    discovery = planner.identify_task(schema, sample_data)

    detected_name = discovery["target"]
    TARGET_COL = next(
        (c for c in df_sample.columns if c.lower() == detected_name.lower()),
        detected_name,
    )
    TASK_TYPE = discovery["task"]

    print(f"Target Detected: {TARGET_COL}")
    print(f"Task Type: {TASK_TYPE}")

    print("Generating strategy...")
    plan = planner.analyze_data(schema, sample_data, TARGET_COL)

    coder = Coder()

    print("Generating cleaning script...")
    raw_code = coder.generate_code(plan, schema, TARGET_COL)
    final_code = clean_ai_markdown(raw_code)

    script_dir = os.path.join(BASE_DIR, "generated_scripts")
    os.makedirs(script_dir, exist_ok=True)

    script_path = os.path.join(script_dir, "clean_data.py")

    with open(script_path, "w") as f:
        f.write(final_code)

    print(f"Script saved to: {script_path}")

    executor = Executor(script_path)
    success, logs = executor.run()

    if success:
        print("Data cleaned successfully!")
        if os.path.exists("data/processed/cleaned_data.csv"):
            print("Output saved to data/processed/cleaned_data.csv")
    else:
        print("Execution failed!")
        print(logs)

    max_retries = 5
    attempt = 0

    while not success and attempt < max_retries:
        attempt += 1

        print(f"Self-correction attempt {attempt}")

        critic = Critic()
        final_code = critic.analyze_error(final_code, logs)

        with open(script_path, "w") as f:
            f.write(clean_ai_markdown(final_code))

        success, logs = executor.run()

    if success:
        print("Data cleaned successfully after self-correction.")
    else:
        print("Critical failure: self-healing unsuccessful.")

    if success:
        print("Proceeding to model training...")

        df_cleaned_sample = pd.read_csv(
            "data/processed/cleaned_data.csv",
            nrows=5
        )
        schema = df_cleaned_sample.dtypes.to_string()

        os.makedirs("models", exist_ok=True)

        modeler = Modeler()

        print("Analyzing data and selecting model...")
        training_code = modeler.generate_training_code(
            schema,
            TARGET_COL,
            TASK_TYPE,
        )

        train_script_path = os.path.join(
            script_dir,
            "train_model.py"
        )

        with open(train_script_path, "w") as f:
            f.write(clean_ai_markdown(training_code))

        train_executor = Executor(train_script_path)
        train_success, train_logs = train_executor.run()

        train_attempt = 0

        while not train_success and train_attempt < max_retries:
            train_attempt += 1

            print(
                f"Training self-correction attempt {train_attempt}"
            )

            critic = Critic()
            training_code = critic.analyze_error(
                training_code,
                train_logs,
            )

            with open(train_script_path, "w") as f:
                f.write(clean_ai_markdown(training_code))

            train_success, train_logs = train_executor.run()

        if train_success:
            print("Model training complete.")
            print(train_logs)

            print("Starting hyperparameter optimization...")

            tuner = Tuner()

            tuning_code = tuner.generate_tuning_code(
                schema,
                TARGET_COL,
                TASK_TYPE,
            )

            tune_script_path = os.path.join(
                script_dir,
                "tune_model.py"
            )

            with open(tune_script_path, "w") as f:
                f.write(clean_ai_markdown(tuning_code))

            tune_executor = Executor(tune_script_path)
            tune_success, tune_logs = tune_executor.run()

            tune_attempt = 0

            while (
                not tune_success
                and tune_attempt < max_retries
            ):
                tune_attempt += 1

                print(
                    f"Tuning self-correction attempt {tune_attempt}"
                )

                critic = Critic()
                tuning_code = critic.analyze_error(
                    tuning_code,
                    tune_logs,
                )

                with open(tune_script_path, "w") as f:
                    f.write(clean_ai_markdown(tuning_code))

                tune_success, tune_logs = tune_executor.run()

            if tune_success:
                print("Optimization complete.")
                print(tune_logs)
            else:
                print(
                    f"Tuning failed after {max_retries} attempts: "
                    f"{tune_logs}"
                )

        else:
            print(
                f"Training failed after {max_retries} attempts: "
                f"{train_logs}"
            )


if __name__ == "__main__":
    main()