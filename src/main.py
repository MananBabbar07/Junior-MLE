


import sys
import os
import io
# import sys
# import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import re
import ast
# Path injection
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.agents.planner import Planner
from src.agents.coder import Coder
from src.tools.executor import Executor
from src.agents.critic import Critic
from src.agents.modeler import Modeler
from src.agents.tuner import Tuner  


def clean_ai_markdown(text):
    """Removes ```python ... ``` blocks if the AI includes them."""
    pattern = r"```(?:python)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def main():
    DATA_PATH = "data/raw/train.csv" # Specific name as requested
    # TARGET_COL = "Survived" 

    
    print("🚀 Initializing Agentic AutoML...")
    
    # 1. Load Metadata
    df_sample = pd.read_csv(DATA_PATH, nrows=5)
    schema = df_sample.dtypes.to_string()
    sample_data = df_sample.head(3).to_string()

    # # 2. Planning Phase
    planner = Planner()
    print("🕵️ Agent is auditing the dataset to find the target...")
    discovery = planner.identify_task(schema, sample_data)
    detected_name = discovery['target']
    TARGET_COL = next((c for c in df_sample.columns if c.lower() == detected_name.lower()), detected_name)
    TASK_TYPE = discovery['task']
    print(f"🎯 Target Detected: {TARGET_COL}")
    print(f"🧠 Task Type: {TASK_TYPE}")
    print("🧠 Planner is strategizing...")
    plan = planner.analyze_data(schema, sample_data, TARGET_COL)
    print("✅ Strategy acquired.")

    # 3. Coding Phase
    coder = Coder()
    print("💻 Coder is writing the cleaning script...")
    raw_code = coder.generate_code(plan, schema, TARGET_COL)
    final_code = clean_ai_markdown(raw_code)
    
    # 4. Save the generated script
    script_dir = os.path.join(BASE_DIR, "generated_scripts")
    os.makedirs(script_dir, exist_ok=True)
    
    script_path = os.path.join(script_dir, "clean_data.py")
    with open(script_path, "w") as f:
        f.write(final_code)
    
    print(f"📂 SUCCESS! Script saved to: {script_path}")

    # 5. Execution Phase
    executor = Executor(script_path)
    success, logs = executor.run()

    if success:
        print("✅ Data cleaned successfully!")
        if os.path.exists("data/processed/cleaned_data.csv"):
            print("📂 File found in data/processed/cleaned_data.csv")
    else:
        print("❌ Execution Failed!")
        print(f"Error Details: {logs}")
        # This is where the CRITIC will eventually step in!

        # ... inside main() after Phase 5 ...
    
    max_retries = 5
    attempt = 0
    
    while not success and attempt < max_retries:
        attempt += 1
        print(f"🔄 Attempt {attempt}: Self-correction in progress...")
        
        critic = Critic()
        # The Critic fixes the code based on the error logs
        final_code = critic.analyze_error(final_code, logs)
        
        # Save the "Fixed" code
        with open(script_path, "w") as f:
            f.write(clean_ai_markdown(final_code))
            
        # Try running it again
        success, logs = executor.run()

    if success:
        print("✅ Data cleaned successfully after self-correction!")
    else:
        print("🛑 Critical Failure: Could not self-heal after 3 attempts.")

    if success:
        print("✅ Data is cleaned. Proceeding to Model Training...")
        
        df_cleaned_sample = pd.read_csv("data/processed/cleaned_data.csv", nrows=5)
        schema = df_cleaned_sample.dtypes.to_string()

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # 6. Modeling Phase
        modeler = Modeler()
        print("🤖 Modeler is analyzing data to choose an algorithm...")
        training_code = modeler.generate_training_code(schema, TARGET_COL, TASK_TYPE)
        
        # Path for the training script
        train_script_path = os.path.join(script_dir, "train_model.py")
        
        with open(train_script_path, "w") as f:
            # We use the same cleaner to remove any AI chatter
            f.write(clean_ai_markdown(training_code))

        # 7. Execute Training with Critic Loop
        train_executor = Executor(train_script_path)
        train_success, train_logs = train_executor.run()
        

        # --- THE MODELER CRITIC LOOP ---
        train_attempt = 0
        while not train_success and train_attempt < max_retries:
            train_attempt += 1
            print(f"🔄 Modeler Attempt {train_attempt}: Self-correcting training script...")
            
            critic = Critic()
            training_code = critic.analyze_error(training_code, train_logs)
            
            with open(train_script_path, "w") as f:
                f.write(clean_ai_markdown(training_code))
            
            train_success, train_logs = train_executor.run()
        # -------------------------------

        if train_success:
            print("🏆 Model Training Complete!")
            print("--- EXECUTION OUTPUT ---")
            print(train_logs)
        
            print("💎 Starting Hyper-parameter Optimization...")
            tuner = Tuner()
            tuning_code = tuner.generate_tuning_code(schema, TARGET_COL, TASK_TYPE)
            
            tune_script_path = os.path.join(script_dir, "tune_model.py")
            with open(tune_script_path, "w") as f:
                f.write(clean_ai_markdown(tuning_code))

            # 9. Execute Tuning with Critic Loop
            tune_executor = Executor(tune_script_path)
            tune_success, tune_logs = tune_executor.run()

            tune_attempt = 0
            while not tune_success and tune_attempt < max_retries:
                tune_attempt += 1
                print(f"🔄 Tuner Attempt {tune_attempt}: Self-correcting tuning script...")
                
                critic = Critic()
                tuning_code = critic.analyze_error(tuning_code, tune_logs)
                
                with open(tune_script_path, "w") as f:
                    f.write(clean_ai_markdown(tuning_code))
                
                tune_success, tune_logs = tune_executor.run()

            if tune_success:
                print("✨ OPTIMIZATION COMPLETE!")
                print("--- TUNING RESULTS ---")
                print(tune_logs)
            else:
                print(f"🛑 Tuning Failed after {max_retries} attempts: {tune_logs}")

        else:
            print(f"🛑 Training Failed after {max_retries} attempts: {train_logs}")
if __name__ == "__main__":
    main()