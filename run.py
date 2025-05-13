from app import create_app

app = create_app()

if __name__ == "__main__":
    # pakai host 127.0.0.1 dan port 5001
    app.run(debug=True, host="127.0.0.1", port=5001)
