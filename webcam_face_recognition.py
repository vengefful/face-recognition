import face_recognition
import cv2
import sys
import numpy as np
import simpleaudio as sa
import requests
import pickle
import os.path
from datetime import date, datetime, time
from PIL import Image, ImageOps

# Função para realizar a requisição POST para adicionar a frequência
def adicionar_frequencia(matricula_aluno, presente, data):
    url_api_frequencia = 'http://localhost:5000/api/add/frequencia'  # Substitua pela sua URL

    # Dados do aluno para a requisição POST
    dados_aluno = {
        "matricula": matricula_aluno,
        "presente": presente,
        "data": data,
    }

    # Realizar a requisição POST
    resposta = requests.post(url_api_frequencia, json=dados_aluno)

    # Verificar a resposta da requisição
    if resposta.status_code == 201:
        print("Frequência adicionada com sucesso!")
    else:
        print("Falha ao adicionar frequência. Código de status:", resposta.status_code)
        sys.exit("Erro ao adicionar frequência. Encerrando o aplicativo.")

# Função para verificar se uma imagem possui EXIF
def tem_orientacao_exif(caminho_imagem):
    try:
        imagem = Image.open(caminho_imagem)
        if hasattr(imagem, '_getexif'):
            exif_info = imagem._getexif()
            if exif_info and 274 in exif_info:
                return True
        return False
    except Exception as e:
        print(f"Ocorreu um erro ao tentar verificar o EXIF da imagem: {e}")
        return False


# Defina o horário limite para marcar os alunos como ausentes
horario_limite = time(23, 59, 0)  # Define o horário como 20:00:00

# Lista de alunos reconhecidos pela webcam
lista_frequencias = []

# Cria lista de alunos_conhecidos
alunos_conhecidos = {}

# Lista de todos os alunos matriculados
lista_alunos = []

# Definir arquivo de áudio a ser reproduzido ao ser reconhecido aluno
som = sa.WaveObject.from_wave_file('/home/fernando/Documents/Projects/face-recognition/Sons/ping.wav')

# URL da sua API para obter as frequências do dia
url_api_frequencia = 'http://localhost:5000/api/frequencia'  # Substitua pela sua URL

# Obter a data atual no formato YYYY-MM-DD
data_atual = date.today().strftime('%Y-%m-%d')

# Parâmetros da requisição GET com a data atual
params = {'data': data_atual}

# Realizar a requisição GET
resposta = requests.get(url_api_frequencia, params=params)

# Verificar a resposta da requisição
if resposta.status_code == 200:
    frequencias_do_dia = resposta.json()  # Obtém as frequências do dia especificado
    if frequencias_do_dia:
        for matricula in frequencias_do_dia['frequencias']:
            lista_frequencias.append(matricula['matricula_aluno'])
else:
    print("Falha ao obter frequências. Código de status:", resposta.status_code)
    sys.exit("Erro na requisição. Encerrando o aplicativo.")

# URL da sua API para obter as frequências do dia
url_api_frequencia = 'http://localhost:5000/api/alunos'  # Substitua pela sua URL

# Realizar a requisição GET
resposta = requests.get(url_api_frequencia)

# Verificar a resposta da requisição
if resposta.status_code == 200:
    alunosJson = resposta.json()  # Obtém as frequências do dia especificado
    if alunosJson:
        for aluno in alunosJson['alunos']:
            lista_alunos.append(aluno)
else:
    print("Falha ao obter alunos. Código de status:", resposta.status_code)
    sys.exit("Erro na requisição. Encerrando o aplicativo.")

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Create arrays of known face encodings and their names

for aluno in lista_alunos:
    print(aluno['foto'])
    codificacao = ""

    if not os.path.exists(f"Codificacoes/{aluno['matricula']}.pkl"):
        if (tem_orientacao_exif(aluno['foto'])):
            image = Image.open(aluno['foto'])
            image = ImageOps.exif_transpose(image)
            image.save(aluno['foto'])

        codificacao = face_recognition.face_encodings(face_recognition.load_image_file(aluno['foto']))[0]
        with open(f"Codificacoes/{aluno['matricula']}.pkl", 'wb') as file:
            pickle.dump(codificacao, file)
    else:
        with open(f"Codificacoes/{aluno['matricula']}.pkl", 'rb') as arquivo:
            codificacao = pickle.load(arquivo)

    matricula = aluno['matricula']
    foto = aluno['foto']
    nome = aluno['nome']
    alunos_conhecidos[matricula] = {'codificacao': codificacao, 'nome': nome}

# alunos_conhecidos = {
#     "2138743342": { "codificacao": face_recognition.face_encodings(face_recognition.load_image_file("./pessoas_conhecidas/Cristiano Ronaldo - 2138743342.jpg"))[0], "nome": "Cristiano Ronaldo" },
#     "1231413": { "codificacao": face_recognition.face_encodings(face_recognition.load_image_file("./pessoas_conhecidas/Messi - 1231413.jpg"))[0], "nome": "Messi" },
#     "21309123": { "codificacao": face_recognition.face_encodings(face_recognition.load_image_file("./pessoas_conhecidas/Fernando - 21309123.jpeg"))[0], "nome": "Fernando Jorge" },
# }

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
matricula_aluno = ''
aluno_encontrado = None
data = ""

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        code = cv2.COLOR_BGR2RGB
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            #matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconhecido"
            turma = ""
            data = ""
            for matricula, aluno_info in alunos_conhecidos.items():
                matches = face_recognition.compare_faces([aluno_info["codificacao"]], face_encoding, tolerance=0.5)

                if matches[0]:
                    name = aluno_info["nome"]
                    matricula_aluno = matricula
                    aluno_encontrado = next((aluno for aluno in lista_alunos if aluno.get("matricula") == matricula_aluno), None)
                    data = datetime.now().strftime("%Y-%m-%d")
                else:
                    aluno_encontrado = None


            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, name, (10, 30), font, 1.0, (0, 0, 255), 2)
        if (aluno_encontrado):
            cv2.putText(frame, f"RA:{aluno_encontrado['matricula']}", (10, 70), font, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, aluno_encontrado['turma'], (10, 110), font, 1.0, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Adicionando Frequencia no backend
    if matricula_aluno not in lista_frequencias and matricula_aluno != '':
        adicionar_frequencia(matricula_aluno, True, data)
        lista_frequencias.append(matricula_aluno)
        matricula_aluno = ''
        play_obj = som.play()


    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Obtém o horário atual
    horario_atual = datetime.now().time()

    # Verifica se o horário atual é posterior ao horário limite
    if horario_atual > horario_limite:
        # Itera sobre todas as matrículas da turma
        for aluno in lista_alunos:
            # Marca o aluno como ausente
            if aluno['matricula'] not in lista_frequencias:
                adicionar_frequencia(aluno['matricula'], False, data)
        sys.exit("Finalizando App")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
