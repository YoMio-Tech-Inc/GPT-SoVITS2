import multiprocessing
import subprocess


def run_script(script_name):
    subprocess.run(["python", script_name])


if __name__ == "__main__":
    scripts = ["GPT_SoVITS/prepare_datasets/1-get-text.py", "GPT_SoVITS/prepare_datasets/2-get-semantic.py", "GPT_SoVITS/prepare_datasets/3-get-wav32k.py"]
    processes = []

    for script in scripts:
        p = multiprocessing.Process(target=run_script, args=(script,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
