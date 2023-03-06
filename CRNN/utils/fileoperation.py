def get_figure(path):
    with open(path, 'r', encoding='utf-8') as f:
        chinese = f.read()
        f.close()
        return chinese

