from search import search_prompt

def main():
    try:
        chain = search_prompt()
    except RuntimeError as e:
        print(f"Não foi possível iniciar o chat: {e}")
        return

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat pronto. Digite 'sair' para encerrar.\n")
    while True:
        pergunta = input("PERGUNTA: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            break

        try:
            resposta = chain(pergunta)
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}\n")
            continue

        print(f"RESPOSTA: {resposta}\n")

if __name__ == "__main__":
    main()