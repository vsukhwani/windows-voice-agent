# Server

```shell
cd server/

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

mv env.example .env 

# Add your service API keys to .env

python bot.py
```

# Client

```shell
cd client/

npm i

npm run dev

# Navigate to URL shown in terminal in your web browser
```