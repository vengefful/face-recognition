import requests

# URL da rota para adicionar o aluno
url = 'http://localhost:5000/api/add/aluno'


for x in range(1, 150):
    x_formatted = str(x).zfill(3)
    data = {
        "matricula": f'{x_formatted}',
        "nome": "Fulano",
        "turma": "A",
        "turno": "Manhã",
        "responsavel": "Pai do Fulano",
        "telefoneResponsavel": "123456789",
        "foto": f'/home/fernando/Documents/Projects/frequencia-escolar/server/faces/{x_formatted}_a.png'
    }
    response = requests.post(url, json=data)

    if response.status_code == 201:
        print("Aluno adicionado com sucesso!")
    else:
        print("Erro ao adicionar aluno:", response.text)
