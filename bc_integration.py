import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from web3 import Web3
import json
from util import classify, set_background

# Load environment variables
ethereum_account = '0xd7985126c2085AA65998733dE8314305E73e8BAa'
private_key = '0xc765998245b9a398e1aafffffcda0ce3278aada88c8f0bd1130cd30e1f0a62f4'

ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

def save_report_on_blockchain(report_data):
    report_data_str = json.dumps(report_data)
    nonce = web3.eth.get_transaction_count(ethereum_account)
    tx = {
        'nonce': nonce,
        'to': ethereum_account,
        'value': 0,
        'gas': 2000000,
        'gasPrice': web3.to_wei('50', 'gwei'),
        'data': web3.to_hex(text=report_data_str)  # Include report data in transaction data
    }
    signed_tx = web3.eth.account.sign_transaction(tx, private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return web3.to_hex(tx_hash)

option = st.sidebar.selectbox("Select Classification type", ("Chest Xray Diagnosis", "Brain Tumour Classification"))

def retrieve_report_from_blockchain(tx_hash):
    try:
        tx = web3.eth.get_transaction(tx_hash)
        if tx and 'input' in tx:
            data_hex = tx['input']
            data_str = web3.to_text(data_hex)
            report = json.loads(data_str)
            return report
        else:
            st.warning(f"No report data found in transaction: {tx_hash}")
            return None
    except Exception as e:
        st.error(f"Error retrieving report: {e}")
        return None

if option == 'Chest Xray Diagnosis':
    set_background('/home/vibhav/Msc_Project/bg/istockphoto-1477482163-2048x2048(1).jpg')
    st.markdown("<h1 style='text-align: center; color: white;'>Implementing CNN for Medical Imaging Classification </h1>", unsafe_allow_html=True)
    st.header('Please upload a chest X-Ray Image')
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
    model = load_model('/home/vibhav/Msc_Project/model/pneumonia_classifier.h5')

    with open('/home/vibhav/Msc_Project/model/labels.txt', 'r') as f:
        class_names = [a.strip().split(' ')[1] for a in f.readlines()]

    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)
        class_name, conf_score = classify(image, model, class_names)
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))

        if st.button("Save Report on Blockchain"):
            report_data = {"type": "Chest Xray Diagnosis", "result": class_name, "score": float(conf_score)}
            tx_hash = save_report_on_blockchain(report_data)
            st.write(f"Report saved! Transaction hash: {tx_hash}")

        tx_hash_input = st.text_input("Enter transaction hash to retrieve report")
        if st.button("Retrieve Report"):
            report = retrieve_report_from_blockchain(tx_hash_input)
            if report:
                st.write("Retrieved report:", report)
            else:
                st.write("No report found for the given transaction hash.")

elif option == 'Brain Tumour Classification':
    set_background('/home/vibhav/Msc_Project/bg/istockphoto-1477482163-2048x2048(1).jpg')
    st.markdown("<h1 style='text-align: center; color: White;'>Brain Tumor Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: White;'>Upload an MRI Scan for Classification</h3>",unsafe_allow_html=True)
    file = st.file_uploader("Please upload your MRI Scan", type=["jpg", "png"])
    model2 = load_model('/home/vibhav/BrainTumor.h5')

    def import_and_predict(image_data, model):
        size = (150, 150)
        image1 = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = ImageOps.grayscale(image1)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        img_reshape = img.reshape(1, 150, 150, 1)
        prediction = model.predict(img_reshape)
        return prediction

    if file is None:
        st.markdown("<h5 style='text-align: center; color: White;'>Please Upload a File</h5>", unsafe_allow_html=True)
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model2)
        
        # Binary classification logic
        class_names = ['Tumour', 'No Tumour']
        
        # Assuming the model outputs probabilities for each class
        if np.argmax(predictions) == 2:  # Index 2 corresponds to 'no_tumor'
            result = class_names[1]  # 'no_tumor'
        else:
            result = class_names[0]  # 'tumor'
        
        st.success(f"The patient most likely has: {result}")

        if st.button("Save Report on Blockchain"):
            report_data = {"type": "Brain Tumor Classification", "result": result}
            tx_hash = save_report_on_blockchain(report_data)
            st.write(f"Report saved! Transaction hash: {tx_hash}")

        tx_hash_input = st.text_input("Enter transaction hash to retrieve report")
        if st.button("Retrieve Report"):
            report = retrieve_report_from_blockchain(tx_hash_input)
            if report:
                st.write("Retrieved report:", report)
            else:
                st.write("No report found for the given transaction hash.")
