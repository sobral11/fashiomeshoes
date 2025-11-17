# Use Python 3.10 explicitamente
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro (cache mais eficiente)
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código da aplicação
COPY . .

# Expor a porta (Render usa PORT automaticamente)
EXPOSE 80

# Comando para rodar a aplicação, usando a porta da variável de ambiente ou 80 por padrão
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
