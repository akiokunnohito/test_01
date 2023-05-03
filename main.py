import numpy as np
import torch
import cv2
import sys
import streamlit as st
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import time
import os
from google.cloud import storage
import google.auth
import json
from google.oauth2.service_account import Credentials

# JSON形式の認証情報を環境変数から取得
service_account_info = json.loads(st.secrets["gcp"]["service_account_key"], strict=False)

# 認証情報を使ってクレデンシャルを作成
credentials = Credentials.from_service_account_info(service_account_info)

# クレデンシャルを指定してstorage.Clientを作成
storage_client = storage.Client(credentials=credentials)

# マスク生成の準備
sys.path.append("..")

# Google Cloud Storageからモデルファイルをダウンロードする関数
def download_model_from_gcs(storage_client, bucket_name, model_path, destination_file_name):
    # (この行は削除します)
    # storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename(destination_file_name)
    print("Model downloaded from Google Cloud Storage.")


# Google Cloud Storageから画像処理モデルをダウンロード
bucket_name = "kika_app"
model_path = "sam_vit_b_01ec64.pth"
local_model_path = "sam_vit_b_01ec64.pth"
download_model_from_gcs(storage_client, bucket_name, model_path, local_model_path)

model_type = "vit_b"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=local_model_path)
sam.to(device=device)



# マスクを塗分ける関数
def show_anns_brunas(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        
        # イテレーションに基づいて色を選択
        if idx % 5 == 0:
            color_mask = [234/255, 84/255, 21/255]  # Orange
        elif idx % 5 == 1:
            color_mask = [255/255, 223/255, 0/255]  # Yellow
        elif idx % 5 == 2:
            color_mask = [0/255, 81/255, 146/255]  # Blue
        elif idx % 5 == 3:
            color_mask = [29/255, 122/255, 33/255]  # Green
        elif idx % 5 == 4:
            color_mask = [255/255, 255/255, 255/255]  # White
        
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*1))) 

    return ax

st.title("testapp")

uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=["jpg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((256, 256), Image.LANCZOS)
    image_np = np.array(image)

    # マスク生成器のインスタンス化と実行
    mask_generator = SamAutomaticMaskGenerator(sam)

    start_time = time.time()  # 処理開始時刻
    masks = mask_generator.generate(image_np)
    processing_time = time.time() - start_time  # 処理時間を計算

    # 結果の表示
    fig, ax = plt.subplots(figsize=(7, 7))
    white_background = np.full((256, 256, 3), 255, dtype=np.uint8)  # 白い背景を作成
    ax.imshow(white_background)
    show_anns_brunas(masks)
    plt.axis('off')
    st.pyplot(fig)

    st.write("リサイズした画像:")
    resized_image = Image.fromarray(image_np)
    st.image(resized_image)

    minutes, seconds = divmod(processing_time, 60)
    st.write(f"処理時間は{int(minutes)}分{int(seconds)}秒でした。")
else:
    st.write("画像ファイルがアップロードされていません。")