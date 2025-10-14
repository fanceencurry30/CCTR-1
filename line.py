#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import lmdb
from tqdm import tqdm

def create_line_lmdb(input_lmdb_path, output_lmdb_path):
    """将整图LMDB转换为行文本LMDB（适配HCCDoc-WS格式）"""
    if not os.path.exists(input_lmdb_path):
        raise FileNotFoundError(f"输入LMDB路径不存在: {input_lmdb_path}")

    os.makedirs(output_lmdb_path, exist_ok=True)
    
    # 打开输入LMDB
    env_in = lmdb.open(input_lmdb_path, readonly=True, max_readers=100, lock=False)
    
    # 自动计算输出LMDB大小（预估每图10行×500KB）
    with env_in.begin() as txn:
        num_images = int(txn.get(b'num-samples').decode())
    map_size = num_images * 30 * 3 * 1024  # 30 行 每行3KB

    env_out = lmdb.open(output_lmdb_path, map_size=map_size)

    samples_processed = 0

    with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
        # 进度条显示
        pbar = tqdm(total=num_images, desc="Processing images")
        
        for key, value in txn_in.cursor():
            if not key.startswith(b'image-'):
                continue

            # 获取关联数据
            image_id = int(key.decode().split('-')[1])
            label_key = f'label-{image_id:09d}'.encode()
            label_data = txn_in.get(label_key)
            
            if not label_data:
                pbar.update(1)
                continue

            # 解码图像和标注
            img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
            label_info = json.loads(label_data.decode('utf-8'))
            
            # 提取HCCDoc-WS标注行
            text_lines = label_info['annotations'].get('HCCDoc-WS', [])    # ？？？？？？？？？
            
            # 按文本行垂直位置排序（从上到下）
            text_lines.sort(key=lambda x: np.mean([x['point'][i] for i in range(1, 8, 2)]))  # 取所有y坐标的平均值
            
            # 处理每行文本
            for line_idx, line in enumerate(text_lines, 1):
                try:
                    # 解析四边形坐标（4个点，每个点x,y交替）
                    pts = np.array(line['point'], dtype=np.float32).reshape(4, 2)
                    
                    # 计算裁剪矩形
                    x, y, w, h = cv2.boundingRect(pts)
                    line_img = img[y:y+h, x:x+w]
                    
                    # 编码为JPEG
                    _, img_enc = cv2.imencode('.jpg', line_img)
                    if img_enc is None:
                        continue
                    
                    # 生成带元数据的label
                    clean_text = line['text'].strip()
                    new_label = f"{clean_text} <image_id={image_id}_line_id={line_idx}>"
                    
                    # 存储到新LMDB
                    new_id = samples_processed + 1
                    txn_out.put(f'image-{new_id:09d}'.encode(), img_enc.tobytes())
                    txn_out.put(f'label-{new_id:09d}'.encode(), new_label.encode('utf-8'))
                    samples_processed += 1
                    
                except Exception as e:
                    print(f"处理图片 {image_id} 行 {line_idx} 时出错: {str(e)}")
                    continue
            
            pbar.update(1)
        pbar.close()

    # 写入样本总数
    with env_out.begin(write=True) as txn_final:
        txn_final.put(b'num-samples', str(samples_processed).encode())

    env_in.close()
    env_out.close()

    print(f"\n转换完成: {output_lmdb_path}")
    print(f"共处理 {num_images} 张原始图片")
    print(f"生成 {samples_processed} 个行文本样本")

if __name__ == "__main__":
    input_lmdb = "./hccdoc_train_lmdb"
    output_lmdb = "./hccdoc_line_lmdb"

    print(f"输入LMDB: {os.path.abspath(input_lmdb)}")
    print(f"输出LMDB: {os.path.abspath(output_lmdb)}")

    create_line_lmdb(input_lmdb, output_lmdb)