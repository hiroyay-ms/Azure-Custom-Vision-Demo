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
    input_container = os.environ['INPUT_DETECT_CONTAINER']
    output_container = os.environ['OUTPUT_CONTAINER_NAME']
    prediction_url = os.environ['DETECT_URL']
    prediction_key = os.environ['PREDICTION_KEY']

    # Blob のダウンロード
    blob_name = image_url[image_url.rindex('/')+1:]
    blob_service = BlobServiceClient.from_connection_string(conn_str=connection_string)
    container = blob_service.get_container_client(input_container)

    img_bytes = container.get_blob_client(blob_name).download_blob().readall()

    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    height, width, depth = image.shape

    # 一時ファイルとして保存
    filename = tempfile.gettempdir() + "o-" + image_url[image_url.rindex('/'):]
    #filename = tempfile.gettempdir() + "\o-" + image_url[image_url.rindex('/')+1:]
    logging.info("Temporary File: %s", filename)

    cv2.imwrite(filename, image)

    # 物体検出の結果を取得
    headers = {'Content-Type': 'application/json', 'Prediction-Key': prediction_key}

    response = requests.post(url=prediction_url, headers=headers, data=open(filename, 'rb'))   
    analytics = response.json()

    logging.info("results: %s", analytics)

    aboveThreshold = False

    for key in analytics['predictions']:

        # probability が 50% を超えるモノをマーク
        if float(key['probability']) > .5:
            aboveThreshold = True

            x1 = key['boundingBox']['left'] * width
            y1 = key['boundingBox']['top'] * height
            x2 = key['boundingBox']['width'] * width
            y2 = key['boundingBox']['height'] * height

            if key['tagName'] == "Banana":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(image, (int(x1), int(y1), int(x2), int(y2)), color)

            text = "{} ({:.2%})".format(key['tagName'], float(key['probability']))
            cv2.putText(image, text, (int(x1), int(y1)+10), cv2.FONT_HERSHEY_SIMPLEX, .5, color, thickness=2)

            logging.info("prediction: %s", text)

    if aboveThreshold:
        cv2.imwrite(filename, image)

        # 結果を Blob へアップロード
        blob_client = blob_service.get_blob_client(container=output_container, blob="o-" + blob_name)

        with open(filename, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
    else:
        logging.info("No Result")
