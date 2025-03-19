import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carregar_modelo():
    #https://drive.google.com/file/d/11jtPJsHbzFd874-XotldQ4Ad_y6aKIXm/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=11jtPJsHbzFd874-XotldQ4Ad_y6aKIXm'
    gdown.download(url, 'model_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='model_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carregar_imagem():
    upload_file=st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', type=['png','jpg','jpeg'])

    if upload_file is not None:
        image_data = upload_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Imagem carregada com sucesso')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['BlackMeasles', 'LeafBlight', 'HealthyGrapes', 'BlackRot']
    #print("Output shape:", output_data.shape)
    #print("Output[0]:", output_data[0])
    #print("Length of output_data[0]:", len(output_data[0]))
    #print("Length of classes:", len(classes))
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]

    fig = px.bar(df, y='classes', x='probabilidades (%)',
                 orientation='h', text='probabilidades (%)',
                 title='Probabilidade de classes de doenças em uvas')
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title='Leaf Classification',
		page_icon='🍇',
    )
    
    st.write('Leaf Classification 🍇')
	
    #Carregar modelo
    interpreter = carregar_modelo()

    #Carregar image
    image = carregar_imagem()

    #Classifica
    if image is not None:
        previsao(interpreter,image)

if __name__ == '__main__':
    main()
