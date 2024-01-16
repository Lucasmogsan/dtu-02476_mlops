from http import HTTPStatus

from fastapi import FastAPI, File, UploadFile
from mlops_group8.predict_model import predict
from google.cloud import storage


app = FastAPI()
# storage_client = storage.Client()
storage_client = storage.Client.create_anonymous_client()

# mlops-group8_model_released/model_latest.pt

bucket_name = "mlops-group8_model_released"
blob_name = "model_latest.pt"

source_bucket = storage_client.bucket(bucket_name)

blobs = sorted(list(source_bucket.list_blobs()), key=lambda blob: blob.updated, reverse=True)
source_blob = blobs[0]

model_name = source_blob.name

source_blob.download_to_filename(f"models/{model_name}")

model_path = f"models/{model_name}"
print(model_path)


@app.post("/predict/")
async def cv_model(data: UploadFile = File(...)):
    """Simple function using open-cv to resize an image."""
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    img = "image.jpg"
    preds = predict(model_path, img)

    response = {
        "input": data,
        "output": preds,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
