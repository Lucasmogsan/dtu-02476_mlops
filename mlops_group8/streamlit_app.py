import streamlit as st
from mlops_group8.predict_model import predict
from google.cloud import storage
import random


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


st.write(
    """
# Rice Classificator! :sunglasses:
""",
)

st.subheader("dtu-02476-mlops: Group 8")


uploaded_file = st.file_uploader("Choose a rice image...", type="jpg")
# display image
if uploaded_file is not None:
    # copy image to local
    with open("image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
        f.close()
    # display image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    img = "image.jpg"
    preds = predict(model_path, img)

    st.header(f"Predicted Class: {preds}", divider="rainbow")
    st.header("Meme of the day:")

    # random number
    number = random.randint(1, 4)

    st.image(f"assets/rice_meme{number}.jpg")
