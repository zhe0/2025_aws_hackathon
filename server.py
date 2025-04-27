# server.py

from flask import Flask, request, jsonify
import boto3
import uuid
import os
import requests
import json
import time

app = Flask(__name__)

# AWS 設定
region = 'us-west-2'
s3_bucket = 'sagemaker-studio-921180591197-34xdr9k8lbh'  # 改成你的 bucket
transcribe_client = boto3.client('transcribe', region_name=region)
s3_client = boto3.client('s3', region_name=region)
bedrock_client = boto3.client('bedrock-runtime', region_name=region)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    transcribed_text = None  # 預設

    # --- ✨ (1) 如果收到的是直接傳文字 (JSON模式)
    if request.is_json:
        data = request.get_json()
        transcribed_text = data.get('text')
        print(f"✅ 收到文字訊息: {transcribed_text}")

    # --- ✨ (2) 如果收到的是上傳音檔 (multipart/form-data模式)
    elif 'file' in request.files:
        file = request.files['file']
        ext = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 上傳到 S3
        s3_key = f"audio/{filename}"
        s3_client.upload_file(filepath, s3_bucket, s3_key)

        # 啟動 Transcribe 辨識
        job_name = f"job-{uuid.uuid4()}"
        file_uri = f"s3://{s3_bucket}/{s3_key}"

        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_uri},
            MediaFormat=ext,
            LanguageCode='zh-TW',
            OutputBucketName=s3_bucket
        )

        # 等待辨識完成
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(2)

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_file_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            response = requests.get(transcript_file_uri)

            if response.status_code == 200:
                transcript_json = response.json()
                transcribed_text = transcript_json['results']['transcripts'][0]['transcript']
            else:
                transcribed_text = "辨識結果無法讀取，請稍後再試。"
        else:
            transcribed_text = "辨識失敗，請重新上傳音檔。"

    else:
        return jsonify({"error": "沒有檔案或文字"}), 400

    # --- 呼叫 Claude 3 Haiku
    prompt_template = f"""
    ROLE:
    你是個 helpful 智能語音機器人。

    ENVIROMENT:
    - 你住在工廠內部。
    - 你有機械手臂可以移動物品。
    - 你有馬達可以在"工廠內部"移動。
    - 你初始狀態是待機狀況。

    STEP:
    依據 ENVIROMENT 知道你的限制。
    分辨聊天的"命令"是甚麼，可能命令有三種:
     - 聊天
     - 查詢
     - 行動
    如果"命令"是"聊天"，請使用大量emoji與使用者聊天。
    如果"命令"是"查詢"，請使用網路查詢配合。
    如果"命令"是"動作"，只能用以下八種：
    1. 從 A 走到 B
    2. 拿起 A 物體
    3. 放下 A 物體
    4. 倒 A 液體到杯子中
    5. 停止倒 A 液體到杯子中
    6. 按下 A 按鈕
    7. 放開 A 按鈕
    8. 說話，說話內容為 A

    EXAMPLE:
    user question:
    幫我送這張請購單去給工讀生
    your answer:
    1 → 2 → 1 → 3 → 8

    USER QUESTION:
    {transcribed_text}
    """

    # Bedrock Claude 3 Haiku
    bedrock_response = bedrock_client.invoke_model(
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": prompt_template}
            ],
            "max_tokens": 300
        }),
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        accept="application/json",
        contentType="application/json"
    )

    bedrock_body = json.loads(bedrock_response['body'].read())
    answer = bedrock_body['content'][0]['text']

    return jsonify({
        "transcribed_text": transcribed_text,
        "answer": answer
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
