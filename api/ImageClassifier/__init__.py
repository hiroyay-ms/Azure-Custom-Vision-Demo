import json
import logging
import requests
import os
import numpy as np
import tempfile
import cv2

import azure.functions as func
from azure.storage.blob import BlobServiceClient


def main(event: func.EventGridEvent):
    result = json.dumps({
        'id': event.id,
        'data': event.get_json(),
        'topic': event.topic,
        'subject': event.subject,
        'event_type': event.event_type,
    })

    logging.info('Python EventGrid trigger processed an event: %s', result)

    # アップロードされた Blob
    image_url = event.get_json()['url']

    connection_string = os.environ['STORAGE_CONNECTION_STRING']
    input_container = os.environ['INPUT_CLASSIFY_CONTAINER']
    output_container = os.environ['OUTPUT_CONTAINER_NAME']
    prediction_url = os.environ['CLASSIFY_URL']
    prediction_key = os.environ['PREDICTION_KEY']

    # Blob のダウンロード
    blob_name = image_url[image_url.rindex('/')+1:]
    blob_service = BlobServiceClient.from_connection_string(conn_str=connection_string)
    container = blob_service.get_container_client(input_container)

    img_bytes = container.get_blob_client(blob_name).download_blob().readall()

    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    height, width, depth = image.shape

    # 一時ファイルとして保存
    filename = tempfile.gettempdir() + "/c-" + image_url[image_url.rindex('/')+1:]
    #filename = tempfile.gettempdir() + "\c-" + image_url[image_url.rindex('/')+1:]
    logging.info("Temporary File: %s", filename)

    cv2.imwrite(filename, image)

    # 画像分類の結果を取得
    headers = {'Content-Type': 'application/json', 'Prediction-Key': prediction_key}

    response = requests.post(url=prediction_url, headers=headers, data=open(filename, 'rb'))   
    predictions = response.json()['predictions']

    logging.info("results: %s", predictions)

    # スコアが高い情報を取得
    highest_probability_index = np.argmax([p.get('probability') for p in predictions])
    prediction = predictions[highest_probability_index]

    # probability が 80% を超える場合、画像に情報を書き込み
    if float(prediction['probability']) > .8:

        if prediction['tagName'] == "Banana":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        text = "label: {}, score: {:.2%}".format(prediction['tagName'], float(prediction['probability']))
        cv2.putText(image, text, (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
 
        cv2.imwrite(filename, image)

        # 結果を Blob へアップロード
        blob_client = blob_service.get_blob_client(container=output_container, blob="c-" + blob_name)

        with open(filename, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        logging.info(text)
    else:
        logging.info("No Result")