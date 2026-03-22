import subprocess
import os

class Executor:
    def __init__(self, script_path):
        self.script_path = script_path

    def run(self):
        print(f"⚙️ Executing: {self.script_path}")
        try:
            # Runs the script and captures output/errors
            result = subprocess.run(
                ["python", self.script_path],
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            # If the script crashes, we get the error message here
            return False, e.stderr