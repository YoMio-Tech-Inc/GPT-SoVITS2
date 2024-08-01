import os
import traceback
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import torch.multiprocessing as mp
from functools import partial
import time
import json

dataset_dir = "./dataset"
progress_file = os.path.join(dataset_dir, "bert_progress.json")


def init_bert_model(gpu_id):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"进程 {os.getpid()} 正在加载模型, 设备: {device}")
    return SentenceTransformer(
        "BAAI/bge-m3", cache_folder="./pretrained_models", device=device
    ).half()


def process_batch(gpu_id, data_queue, result_queue):
    bert_model = init_bert_model(gpu_id)
    batch_size = 48

    while True:
        batch = data_queue.get()
        if batch is None:
            break

        texts = [item[2] for item in batch]
        try:
            bert_features = bert_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                output_value="token_embeddings",
                batch_size=batch_size,
            )
            result_queue.put((batch, bert_features))
        except Exception as e:
            print(f"BERT模型处理出错: {str(e)}")
            result_queue.put((batch, None))

    result_queue.put(None)


def save_results(result_queue, total_items):
    processed_items = 0
    with tqdm(total=total_items, desc="保存BERT特征") as pbar:
        while processed_items < total_items:
            result = result_queue.get()
            if result is None:
                continue
            batch, bert_features = result
            if bert_features is None:
                continue

            for (spk_name, wav_name, text, index_folder), feature in zip(
                batch, bert_features
            ):
                try:
                    bert_dir = os.path.dirname(wav_name)
                    name = os.path.basename(wav_name)
                    path_bert = os.path.join(
                        dataset_dir, index_folder, bert_dir, f"{name}.pt"
                    )
                    torch.save(feature.cpu(), path_bert)
                    processed_items += 1
                    pbar.update(1)

                    # 保存进度
                    with open(progress_file, "w") as f:
                        json.dump({"processed_items": processed_items}, f)
                except Exception as e:
                    print(f"保存BERT特征出错: {spk_name}, {wav_name}, {text}")
                    print(f"错误信息: {str(e)}")

    print(f"总共处理了 {processed_items} 个项目")


def main(todo):
    # 检查是否有保存的进度
    start_index = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
            start_index = progress.get("processed_items", 0)
        print(f"从上次处理的位置继续，已处理 {start_index} 个项目")

    num_gpus = torch.cuda.device_count()
    num_processes = max(1, num_gpus)
    print(f"将使用 {num_processes} 个进程进行并行处理")

    data_queue = mp.Queue(maxsize=100)
    result_queue = mp.Queue()

    # 启动GPU处理进程
    gpu_processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=process_batch, args=(i % num_gpus, data_queue, result_queue)
        )
        p.start()
        gpu_processes.append(p)

    # 启动保存结果的进程
    save_process = mp.Process(target=save_results, args=(result_queue, len(todo)))
    save_process.start()

    # 主进程作为生产者
    batch_size = 48
    for i in range(start_index, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        data_queue.put(batch)

    # 发送结束信号
    for _ in range(num_processes):
        data_queue.put(None)

    # 等待所有进程结束
    for p in gpu_processes:
        p.join()
    save_process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    todo = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".txt"):
                index_folder = os.path.relpath(root, dataset_dir)
                file_path = os.path.join(root, file)

                # 尝试不同的编码
                encodings = ["utf-8", "gbk", "gb2312", "utf-16"]
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            lines = f.readlines()
                        break  # 如果成功读取，跳出循环
                    except UnicodeDecodeError:
                        continue  # 如果解码失败，尝试下一个编码
                else:
                    print(f"无法解码文件 {file_path}，跳过此文件")
                    continue  # 如果所有编码都失败，跳过此文件

                for line in lines:
                    try:
                        spk_name, wav_name, text = line.split("|")
                        todo.append([spk_name, wav_name, text, index_folder])
                    except Exception:
                        print(line)

    try:
        start_time = time.time()
        main(todo)
        print(f"总处理时间: {time.time() - start_time} 秒")
        # 处理完成后删除进度文件
        if os.path.exists(progress_file):
            os.remove(progress_file)
    except Exception as e:
        print(f"处理过程中遇到错误: {str(e)}")
        print("您可以稍后重新运行程序，它将从上次处理的位置继续")
