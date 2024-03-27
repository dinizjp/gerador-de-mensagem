import streamlit as st
import pytesseract
import cv2
import numpy as np
import re


# Função para limpar o texto extraído pelo OCR
def clean_ocr_result(text):
    # Primeiro, remover os espaços extras entre as palavras
    text = re.sub(r'\s+', ' ', text)
    
    # Em seguida, vamos remover caracteres soltos que não fazem parte de uma palavra maior
    text = re.sub(r'\b\w\b', '', text)
    
    # Por fim, remover espaços extras novamente após as remoções anteriores
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Função para isolar o balão e realizar OCR
def isolate_balloon_and_ocr(opencv_image):
    # Converter para escala de cinza
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    # Inverter a imagem para que o texto fique preto e o balão branco
    inverted_gray = cv2.bitwise_not(gray)

    # Aplicar threshold para isolar o balão branco
    _, thresh = cv2.threshold(inverted_gray, 200, 255, cv2.THRESH_BINARY)

    # Encontrar contornos no threshold
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variáveis para determinar o balão mais provável
    max_area = 0
    balloon_contour = None

    # Loop pelos contornos para identificar um contorno que se pareça com um balão
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            balloon_contour = cnt

    # Se um balão for encontrado, isolar e realizar OCR
    if balloon_contour is not None:
        x, y, w, h = cv2.boundingRect(balloon_contour)
        roi = inverted_gray[y:y+h, x:x+w]
        
        # Configuração customizada do Tesseract para maximizar a precisão
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(roi, lang='por', config=custom_config)
        return text.strip()

    return "Texto não encontrado."


# Define o caminho do Tesseract OCR se não estiver no PATH do sistema
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Interface do aplicativo Streamlit
st.title('Gerador de Mensagens para Sorteio')
uploaded_files = st.file_uploader("Escolha os prints dos ganhadores", accept_multiple_files=True, type=['png', 'jpg'])

if uploaded_files:
    sorteio_nums = [st.text_input(f"Digite o número do sorteio para {uploaded_file.name}", key=uploaded_file.name) for uploaded_file in uploaded_files]

    if st.button('Gerar Mensagem'):
        mensagem = "E VAMOS AOS VENCEDORES DA NOIIITEEE!!!\n\n"
        all_data = []

        for uploaded_file, sorteio_num in zip(uploaded_files, sorteio_nums):
            if uploaded_file is not None and sorteio_num:
                # Converter a imagem carregada para o formato que a OpenCV pode entender
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                extracted_text = isolate_balloon_and_ocr(opencv_image)
                cleaned_text = clean_ocr_result(extracted_text)
                all_data.append((sorteio_num, cleaned_text))

        # Ordenar por número do sorteio
        all_data.sort(key=lambda x: int(x[0]))

        # Construir a mensagem com os dados ordenados
        for sorteio_num, name in all_data:
            mensagem += f"SORTEIO #{sorteio_num}\n{name}\n\n"

        st.text_area("Mensagem gerada", mensagem, height=300)
