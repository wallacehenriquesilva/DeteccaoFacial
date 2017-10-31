import cv2
import os
import numpy as np

import time

# ### Treinando dados
# Array de nomes, o primeiro é vazio porque não existe usuário cadastrado como 0
pessoas = ["", "Neymar", "Wallace"];

# instancia o uso da webcam
webcam = cv2.VideoCapture(0);


# ### Preparando o treinamento
# Função parar detectar uma face usando OpenCV e haarcascade
def detecta_face(img):
    # Converte a imagem para escala de cinza porque o OpenCV trabalha com escala de cinza.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

    # Carrega o haarcascade
    face_haarcascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml');

    # Detecta imagens multi escala, porque as imagens podem estar mais próximas e mais longes da camera
    faces = face_haarcascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # Caso nenhum rosto for detectado, retorna.
    if (len(faces) == 0):
        return None, None;

    # Pega a posição do rosto encontrado
    (x, y, w, h) = faces[0];

    # Retorna a parte da imagem que contem o rosto apenas.
    return gray[y:y + w, x:x + h], faces[0]


def prepara_dados_treinamento(diretorio):
    # Pega o diretorio onde estão os arquivos de treinamento
    diretorios = os.listdir(diretorio);

    # Cria a lista de faces
    listFaces = [];
    # Lista de labels com todas as pessoas
    listLabels = [];

    # Foreach de diretorios
    for nomeDiretorio in diretorios:

        # pega os diretorios que contem a inicial 't'
        # ignore any non-relevant directories if any
        if not nomeDiretorio.startswith("t"):
            continue;

        # ------STEP-2--------
        # Monta a lista, tirando o t da pasta, deixando apenas o número, que fará referência a
        # lista de pessoas lá do inicio.
        # Ex. t1 -> 1 - Wallace
        label = int(nomeDiretorio.replace("t", ""))

        # monta o diretorio
        diretorioTemporario = diretorio + "/" + nomeDiretorio

        # Pega a lista de imagens que tem no diretório
        listImagens = os.listdir(diretorioTemporario)

        # ------STEP-3--------
        # Foreach das imagens no diretorio
        for nomeImagem in listImagens:

            # Ignora os arquivos de sistema, que iniciam com ponto
            if nomeImagem.startswith("."):
                continue;

            # Pega o caminho absoluto da imagme com extensão.
            caminhoImagem = diretorioTemporario + "/" + nomeImagem

            # Le imagem
            image = cv2.imread(caminhoImagem)

            cv2.waitKey(100)

            # Detecta a face
            face, rect = detecta_face(image)

            # Ignora os rostos não detectados
            if face is not None:
                # adiciona o rosto a lista de rostos
                listFaces.append(face)
                # add label for this face
                listLabels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return listFaces, listLabels


print("Preparando dados...")
listFaces, listLabels = prepara_dados_treinamento("training-data")
print("Data prepared")

# print total faces e labels
print("Total de faces: ", len(listFaces))
print("Total de labels: ", len(listLabels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Faz o treinamento
face_recognizer.train(listFaces, np.array(listLabels))

# Salva o treinamento em um arquivo
face_recognizer.save('trainer/trainer.yml');


# ### Predicação

# Função para desenhar o retangulo na imagem
# Desenha de acordo com as posições  x e y e com a altura e largura.
# Desenha também um retangulo para exibir o nome em cima.
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (250, 100, 0), 2)
    cv2.rectangle(img, (x, y), (x + w, y - 26), (255, 100, 0), -1)


# Funcão para escrever o nome no retangulo
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)


def predicao(test_img):
    # Faz uma cópia da imagem
    img = test_img.copy()
    # Detecta a face na imagem
    face, rect = detecta_face(img)

    # Faz a predição da imagme usando o reconhecedor
    label, confidence = face_recognizer.predict(face)
    # pega o nome do responsável pela imagem na lista de responsáveis.
    label_text = pessoas[label]

    # Desenha o retangulo na face detectada
    draw_rectangle(img, rect)
    # Escreve o nome da pessoa da face reconhecida
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img


print("Realizando predicao das images...")
"""
# Carrega as imagens de teste
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")

# Faz a predicação
predicted_img1 = predicao(test_img1)
predicted_img2 = predicao(test_img2)
predicted_img3 = predicao(test_img3)
print("Predição complete")
"""

# display both images
# cv2.imshow(pessoas[1], cv2.resize(predicted_img1, (400, 500)))
# cv2.imshow(pessoas[2], cv2.resize(predicted_img2, (400, 500)))
# cv2.imshow(pessoas[3], cv2.resize(predicted_img3, (400, 500)))

while True:
    try:
        # pega efeticamente a imagem da webcam
        s, imagem = webcam.read();

        # espelha a imagem
        imagem = cv2.flip(imagem, 180);
        #cv2.imshow("Imagem da web cam", imagem);
        # mostra a imagem captura na janela
        predicted_img4 = predicao(imagem);
        # Velocidade da exibição do vídeo
        time.sleep(.05);
        cv2.imshow("Imagem da web cam", predicted_img4);

        # o trecho seguinte é apenas para parar o código e fechar a janela
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    except:
        print("NÃO ENCONTRADO");

cv2.waitKey(0);
cv2.destroyAllWindows();
cv2.waitKey(1);
cv2.destroyAllWindows();

# Dispensa o uso da webcam
webcam.release();
