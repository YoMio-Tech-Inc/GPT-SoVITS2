import multiprocessing
import subprocess
import time


def run_script(script_name, return_dict):
    start_time = time.time()
    subprocess.run(["python3", script_name])
    end_time = time.time()
    return_dict[script_name] = end_time - start_time


if __name__ == "__main__":
    start_time = time.time()
    # scripts = ["GPT_SoVITS/prepare_datasets/1-get-text.py", "GPT_SoVITS/prepare_datasets/2-get-semantic.py", "GPT_SoVITS/prepare_datasets/3-get-wav32k.py"]
    scripts = [
        "GPT_SoVITS/prepare_datasets/1-get-text-GPUs.py",
        "GPT_SoVITS/prepare_datasets/2+3.py",
    ]
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    for script in scripts:
        p = multiprocessing.Process(target=run_script, args=(script, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for script, script_time in return_dict.items():
        print(f"{script} 处理时间: {script_time} 秒")
    end_time = time.time()
    print(f"总处理时间: {end_time - start_time} 秒")
