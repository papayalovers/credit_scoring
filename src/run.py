from pathlib import Path
from pipeline.pipeline import run_training_pipeline
from utils.logger import setup_logger
from tqdm import tqdm
from colorama import Fore, init

init(autoreset=True)

###################################################
# Running end-to-end training models until finished
###################################################
def main():
    setup_logger()

    print(Fore.CYAN + "Starting Training Pipeline...\n")

    pbar = None
    current_step = None

    def update_progress(step: str, message: str, inc: int):
        nonlocal current_step, pbar

        # Create new progress bar if has new step
        if current_step != step:
            if pbar:
                pbar.close()

            current_step = step
            pbar = tqdm(total=100, desc=f"[{step}]")

        tqdm.write(f"{Fore.YELLOW}[{step}] ⏳ {message}")
        pbar.update(inc)

    def step_done(step: str):
        nonlocal pbar

        if pbar:
            pbar.n = 100
            pbar.refresh()
            pbar.close()
            pbar = None

        tqdm.write(f"{Fore.GREEN}[{step}] Completed 100% ✅\n")

    run_training_pipeline(
        progress_callback=update_progress,
        step_done_callback=step_done
    )

    print(Fore.GREEN + "Training Finished Successfully!")

if __name__ == "__main__":
    main()