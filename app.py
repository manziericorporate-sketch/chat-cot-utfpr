# ==============================================================================
# CHATBOT CONVERSACIONAL COM ESTRATÉGIA DE RACIOCÍNIO — CHAIN-OF-THOUGHT (CoT)
# ==============================================================================
# Projeto Final — Arquitetura Transformer e Grandes Modelos de Linguagem (LLMs)
# Especialização em Inteligência Artificial Generativa Aplicada — UTFPR
# Professor: Rogério  |  Grupo: Eloi, Lucy, Winderson e Paulo
#
# O QUE ESTE PROGRAMA FAZ
#   Abre, no navegador, uma tela de chat (feita com Streamlit) que conversa com o
#   modelo de linguagem Gemini, do Google. A pergunta do usuário NÃO é enviada
#   direto ao modelo ("chamada ingênua"). Ela passa por DUAS ETAPAS ENCADEADAS:
#
#     ETAPA 1 — CADEIA DE RACIOCÍNIO: uma chamada auxiliar pede ao Gemini só o
#               "pensamento": entender o problema, planejar, calcular e conferir.
#               Essa chamada não entra na memória da conversa (não a polui).
#     ETAPA 2 — SÍNTESE FINAL: a pergunta + a cadeia de raciocínio são enviadas à
#               sessão de chat com memória, que escreve a resposta final ao usuário.
#
# COMO EXECUTAR (resumo — detalhes no arquivo LEIA-ME.txt)
#   1) pip install -r requirements.txt
#   2) Definir a chave:  set GEMINI_API_KEY=sua_chave    (Windows)
#                        export GEMINI_API_KEY=sua_chave (Mac/Linux)
#   3) streamlit run app.py
#   Atalho no Windows: dar dois cliques em INICIAR_CHAT.bat (faz tudo sozinho).
# ==============================================================================


# ------------------------------------------------------------------------------
# 0. BIBLIOTECAS (ferramentas prontas que o programa usa)
# ------------------------------------------------------------------------------
import os                                                               # Lê variáveis do sistema operacional (é onde fica guardada a chave da API)
import re                                                               # "Expressões regulares": localiza e apaga trechos de texto com um padrão
import time                                                             # Permite pausar o programa por alguns segundos (usado ao tentar de novo)
from pathlib import Path                                                # Monta caminhos de arquivos, como o da imagem do logo

import streamlit as st                                                  # Streamlit: desenha a tela do chat no navegador
from google import genai                                                # SDK oficial do Google para conversar com o Gemini
from google.genai import errors, types                                  # errors = tipos de erro da API; types = formatos de configuração e mensagem


# ------------------------------------------------------------------------------
# 1. CONFIGURAÇÕES GERAIS (valores fixos usados em todo o programa)
# ------------------------------------------------------------------------------
MODELO = "gemini-3.1-flash-lite"                                        # Nome do modelo de linguagem do Google (o mesmo usado em aula)
TEMPERATURA_ETAPA1 = 0.2                                                # Baixa "criatividade" na Etapa 1: raciocínio mais preciso e estável
TEMPERATURA_ETAPA2 = 0.3                                                # Criatividade moderada na Etapa 2: texto natural, mas fiel ao raciocínio
MAX_TENTATIVAS = 3                                                      # Quantas vezes o programa tenta de novo se a internet ou a API falhar
PASTA_DO_APP = Path(__file__).parent                                    # Pasta onde este arquivo app.py está salvo
ARQUIVO_LOGO = PASTA_DO_APP / "logo_utfpr.png"                          # Caminho completo da imagem do logo da UTFPR
TAG_INICIO = "<analise_interna>"                                        # Marcador que indica onde COMEÇA o raciocínio guardado no histórico
TAG_FIM = "</analise_interna>"                                          # Marcador que indica onde TERMINA o raciocínio guardado no histórico

# Instrução de sistema: as "regras da casa" que o modelo segue em TODA a conversa.
INSTRUCAO_SISTEMA = (                                                   # Abre o texto (as linhas abaixo são juntadas em um texto só)
    "Você é um assistente técnico, objetivo e didático, com raciocínio analítico. "  # Papel do assistente
    "Responda sempre em português do Brasil. "                          # Idioma das respostas
    "Use o histórico da conversa para entender perguntas que citam mensagens anteriores. "  # Usa a memória
    "Trechos entre <analise_interna> e </analise_interna> são anotações internas suas: "  # Explica os marcadores
    "não fazem parte da pergunta do usuário e nunca devem ser citados como tal."  # Proíbe confundir com a pergunta
)                                                                       # Fecha o texto da instrução de sistema


# ------------------------------------------------------------------------------
# 2. MEMÓRIA INICIAL (para o chat não abrir vazio)
# ------------------------------------------------------------------------------
# Esta pequena conversa já vem "carregada": aparece na tela E na memória do Gemini.
# Por isso, mais tarde, o modelo consegue responder "quem são os alunos do grupo?".
MEMORIA_INICIAL = [                                                     # Lista de mensagens que abrem a conversa
    {                                                                   # Primeira mensagem: o grupo se apresenta
        "role": "user",                                                 # "user" = mensagem escrita pelo usuário
        "content": (                                                    # Texto da mensagem
            "Olá! Somos o grupo formado por Eloi, Lucy, Winderson e Paulo, alunos da "  # Nomes dos alunos do grupo
            "Especialização em Inteligência Artificial Generativa Aplicada da UTFPR."  # Curso e universidade
        ),                                                              # Fim do texto da mensagem
    },                                                                  # Fim da primeira mensagem
    {                                                                   # Segunda mensagem: o assistente responde e explica como funciona
        "role": "assistant",                                            # "assistant" = mensagem escrita pela IA
        "content": (                                                    # Texto da mensagem
            "Olá, Eloi, Lucy, Winderson e Paulo! Sejam bem-vindos. 👋\n\n"  # Saudação aos quatro alunos
            "Eu uso a estratégia **Encadeamento de Pensamento (Chain-of-Thought)**: "  # Nome da estratégia
            "antes de responder, monto um raciocínio passo a passo (entender, planejar, "  # Como funciona (parte 1)
            "calcular e conferir) e só depois escrevo a resposta final.\n\n"  # Como funciona (parte 2)
            "Envie um problema que exija análise — por exemplo, um cálculo de ponto de "  # Convite (parte 1)
            "equilíbrio — e depois faça perguntas sobre o que já conversamos."  # Convite (parte 2)
        ),                                                              # Fim do texto da mensagem
    },                                                                  # Fim da segunda mensagem
]                                                                       # Fim da memória inicial

# Pergunta de exemplo (botão na barra lateral) — caso de uso que exige raciocínio analítico.
PERGUNTA_EXEMPLO = (                                                    # Abre o texto da pergunta de exemplo
    "A Loja Alfa vende um produto por R$ 80,00. O custo variável é de R$ 50,00 por unidade "  # Dados (parte 1)
    "e os custos fixos mensais somam R$ 24.000,00. "                    # Dados (parte 2)
    "(a) Qual é o ponto de equilíbrio em unidades? "                    # Pergunta (a)
    "(b) Quantas unidades a loja precisa vender para lucrar R$ 12.000,00 no mês? "  # Pergunta (b)
    "(c) Se o preço subir 10% e o volume de vendas cair 15% em relação ao item (b), "  # Pergunta (c), parte 1
    "o lucro aumenta ou diminui? Em quanto?"                            # Pergunta (c), parte 2
)                                                                       # Fecha o texto da pergunta de exemplo


# ------------------------------------------------------------------------------
# 3. TRATAMENTO DE ERROS (para o app não "quebrar" quando algo dá errado)
# ------------------------------------------------------------------------------
class ErroAmigavel(Exception):                                          # Cria um tipo de erro próprio, com mensagem clara em português
    pass                                                                # Não precisa de nada extra: só carrega a mensagem de erro


def traduzir_erro_da_api(codigo, erro) -> str:                          # Converte o código técnico do erro em uma frase simples
    if codigo == 429:                                                   # 429 = muitas perguntas em pouco tempo (limite da conta gratuita)
        return "Limite de requisições da API atingido (erro 429). Aguarde 1 minuto e tente de novo."  # Mensagem
    if codigo in (401, 403) or (codigo == 400 and "API key" in str(erro)):  # Chave errada ou sem permissão
        return "Chave de API inválida ou sem permissão. Confira a variável GEMINI_API_KEY."  # Mensagem
    if codigo == 404:                                                   # 404 = o modelo pedido não existe ou não está disponível
        return f"Modelo '{MODELO}' não encontrado. Ajuste a constante MODELO no app.py."  # Mensagem
    if codigo is not None and codigo >= 500:                            # 500 ou mais = problema nos servidores do Google
        return "Os servidores do Gemini estão instáveis no momento. Tente novamente em instantes."  # Mensagem
    return f"Erro da API Gemini (código {codigo}): {erro}"              # Qualquer outro erro: mostra o detalhe técnico


def executar_com_retentativa(funcao):                                   # Executa uma chamada à API e tenta de novo em falhas passageiras
    for tentativa in range(1, MAX_TENTATIVAS + 1):                      # Repete no máximo 3 vezes (tentativas 1, 2 e 3)
        try:                                                            # "Tente" executar o bloco abaixo; se der erro, vai para um dos "except"
            return funcao()                                             # Faz a chamada à API; se der certo, devolve o resultado e sai
        except errors.APIError as erro:                                 # Caso 1: a API do Google respondeu com um erro
            codigo = getattr(erro, "code", None)                        # Lê o código numérico do erro (ex.: 429, 403, 500)
            passageiro = codigo == 429 or (codigo or 0) >= 500          # Erros que costumam se resolver sozinhos
            if passageiro and tentativa < MAX_TENTATIVAS:               # Se é passageiro e ainda há tentativas...
                time.sleep(2 ** tentativa)                              # ...espera 2 s (depois 4 s) antes de tentar de novo
                continue                                                # Volta ao início do "for" para a próxima tentativa
            raise ErroAmigavel(traduzir_erro_da_api(codigo, erro)) from erro  # Erro definitivo: avisa o usuário
        except Exception as erro:                                       # Caso 2: falha de internet, tempo esgotado etc.
            if tentativa < MAX_TENTATIVAS:                              # Se ainda há tentativas...
                time.sleep(2 ** tentativa)                              # ...espera um pouco
                continue                                                # ...e tenta de novo
            raise ErroAmigavel(f"Falha de conexão com a API Gemini: {erro}") from erro  # Desiste e avisa
    raise ErroAmigavel("Não foi possível obter resposta do modelo.")    # Segurança: nunca deveria chegar aqui


# ------------------------------------------------------------------------------
# 4. CONEXÃO COM O GEMINI E MEMÓRIA DA CONVERSA
# ------------------------------------------------------------------------------
def obter_cliente() -> genai.Client:                                    # Cria (uma única vez) a conexão com o Gemini
    if "cliente_gemini" not in st.session_state:                        # st.session_state = "gaveta" que guarda dados enquanto a página está aberta
        chave = os.environ.get("GEMINI_API_KEY")                        # Busca a chave da API nas variáveis do sistema
        if not chave:                                                   # Se a chave não foi configurada...
            st.error("Variável de ambiente GEMINI_API_KEY não encontrada. Veja o LEIA-ME.txt.")  # ...mostra aviso
            st.stop()                                                   # ...e interrompe o app (sem chave não há como falar com o Gemini)
        st.session_state.cliente_gemini = genai.Client(api_key=chave)   # Cria o cliente e guarda na "gaveta"
    return st.session_state.cliente_gemini                              # Devolve o cliente já criado


def converter_memoria_inicial() -> list:                                # Converte a MEMORIA_INICIAL para o formato que o Gemini entende
    historico = []                                                      # Lista vazia que vai receber as mensagens convertidas
    for mensagem in MEMORIA_INICIAL:                                    # Percorre cada mensagem da memória inicial
        papel = "user" if mensagem["role"] == "user" else "model"       # Para o Gemini, o assistente se chama "model"
        parte = types.Part(text=mensagem["content"])                    # Embrulha o texto no formato "Part" do SDK
        historico.append(types.Content(role=papel, parts=[parte]))      # Junta papel + texto e adiciona à lista
    return historico                                                    # Devolve o histórico pronto


def obter_sessao_chat():                                                # Cria (uma única vez) a sessão de chat COM MEMÓRIA nativa do SDK
    if "sessao_chat" not in st.session_state:                           # Só cria se ainda não existir nesta página
        st.session_state.sessao_chat = obter_cliente().chats.create(    # chats.create = chat que guarda o histórico
            model=MODELO,                                               # Qual modelo usar
            config=types.GenerateContentConfig(                         # Configurações da conversa
                temperature=TEMPERATURA_ETAPA2,                         # Grau de criatividade das respostas finais
                system_instruction=INSTRUCAO_SISTEMA,                   # "Regras da casa" definidas na seção 1
            ),                                                          # Fim das configurações
            history=converter_memoria_inicial(),                        # O chat já nasce com a memória inicial (não começa vazio)
        )                                                               # Fim da criação da sessão de chat
    return st.session_state.sessao_chat                                 # Devolve a sessão de chat


def historico_em_texto(sessao_chat) -> str:                             # Transforma a memória do chat em texto simples
    linhas = []                                                         # Lista que vai receber cada fala da conversa
    for conteudo in sessao_chat.get_history():                          # get_history() = todas as mensagens guardadas na memória
        texto = "".join(parte.text or "" for parte in (conteudo.parts or []))  # Junta os pedaços de texto
        texto = re.sub(f"{TAG_INICIO}.*?{TAG_FIM}", "", texto, flags=re.S)  # Apaga raciocínios antigos (economiza)
        quem = "Usuário" if conteudo.role == "user" else "Assistente"   # Identifica quem falou
        linhas.append(f"{quem}: {texto.strip()}")                       # Adiciona a fala no formato "Quem: texto"
    return "\n".join(linhas)                                            # Junta tudo em um único texto, uma fala por linha


# ==============================================================================
# 5. [TRECHO PRINCIPAL] ESTRATÉGIA DE RACIOCÍNIO: CHAIN-OF-THOUGHT EM 2 ETAPAS
# ==============================================================================
def etapa1_gerar_cadeia_de_raciocinio(pergunta: str, historico: str) -> str:  # ETAPA 1: só o "pensamento"
    prompt = (                                                          # Monta o pedido (prompt) da Etapa 1 — as linhas abaixo viram um texto só
        "Você é um analista que pensa passo a passo.\n\n"               # Papel do modelo nesta etapa
        f"HISTÓRICO DA CONVERSA:\n{historico}\n\n"                      # Contexto: o que já foi conversado (memória)
        f"NOVA PERGUNTA DO USUÁRIO:\n{pergunta}\n\n"                    # A pergunta atual
        "TAREFA: NÃO escreva a resposta final ao usuário. Escreva SOMENTE a cadeia de raciocínio:\n"  # Regra central
        "1. ENTENDIMENTO: o que está sendo pedido e quais dados foram fornecidos.\n"  # Passo 1 do raciocínio
        "2. PLANO: lista numerada das etapas necessárias para resolver.\n"  # Passo 2 do raciocínio
        "3. EXECUÇÃO: resolva cada etapa, mostrando contas e justificativas.\n"  # Passo 3 do raciocínio
        "4. VERIFICAÇÃO: confira os resultados (refaça as contas ou teste a lógica).\n"  # Passo 4 do raciocínio
        "5. CONCLUSÃO PRELIMINAR: uma ou duas frases."                  # Passo 5 do raciocínio
    )                                                                   # Fim do prompt da Etapa 1
    configuracao = types.GenerateContentConfig(temperature=TEMPERATURA_ETAPA1)  # Temperatura baixa = mais precisão
    resposta = executar_com_retentativa(                                # Chama a API com proteção contra falhas
        lambda: obter_cliente().models.generate_content(                # generate_content = chamada AVULSA, sem memória
            model=MODELO, contents=prompt, config=configuracao          # Modelo, texto do pedido e configuração
        )                                                               # Fim da chamada
    )                                                                   # Fim da proteção contra falhas
    return (resposta.text or "").strip()                                # Devolve só o texto do raciocínio, sem espaços sobrando


def etapa2_sintetizar_resposta(pergunta: str, cadeia: str) -> str:      # ETAPA 2: resposta final, com memória
    prompt = (                                                          # Monta o pedido da Etapa 2: pergunta original + raciocínio da Etapa 1
        f"{pergunta}\n\n"                                               # A pergunta original vem primeiro (é o que fica como "pergunta" na memória)
        f"{TAG_INICIO}\n"                                               # Marca o início do raciocínio interno
        f"Raciocínio passo a passo já realizado (use como base):\n{cadeia}\n"  # O raciocínio da Etapa 1
        f"{TAG_FIM}\n\n"                                                # Marca o fim do raciocínio interno
        "Com base nesse raciocínio, responda à pergunta de forma clara e organizada, "  # Instrução (parte 1)
        "mostrando os principais passos e destacando o resultado final em negrito."  # Instrução (parte 2)
    )                                                                   # Fim do prompt da Etapa 2
    sessao_chat = obter_sessao_chat()                                   # Pega a sessão de chat que tem memória
    resposta = executar_com_retentativa(lambda: sessao_chat.send_message(prompt))  # send_message = envia e GUARDA na memória
    return (resposta.text or "").strip()                                # Devolve o texto da resposta final


def responder_com_chain_of_thought(pergunta: str) -> dict:              # Junta as duas etapas (recebe só a pergunta atual)
    historico = historico_em_texto(obter_sessao_chat())                 # Lê a memória para dar contexto à Etapa 1
    with st.spinner("🧠 Etapa 1/2 — Montando a cadeia de raciocínio passo a passo..."):  # Animação "carregando"
        cadeia = etapa1_gerar_cadeia_de_raciocinio(pergunta, historico)  # Executa a Etapa 1
    if not cadeia:                                                      # Se a Etapa 1 voltou vazia...
        raise ErroAmigavel("O modelo não retornou a cadeia de raciocínio. Tente novamente.")  # ...avisa o usuário
    with st.spinner("✍️ Etapa 2/2 — Escrevendo a resposta final com base no raciocínio..."):  # Nova animação
        resposta = etapa2_sintetizar_resposta(pergunta, cadeia)         # Executa a Etapa 2
    return {"resposta": resposta, "cadeia": cadeia}                     # Devolve a resposta final e o raciocínio usado


# ------------------------------------------------------------------------------
# 6. INTERFACE DO CHAT (o que aparece na tela)
# ------------------------------------------------------------------------------
def limpar_conversa():                                                  # Função chamada pelo botão "Limpar conversa"
    for chave in ("mensagens", "sessao_chat"):                          # Itens a apagar: histórico da tela e memória do modelo
        st.session_state.pop(chave, None)                               # Apaga o item (se não existir, ignora sem erro)


def mostrar_cadeia(cadeia: str):                                        # Mostra o raciocínio da Etapa 1 numa caixa que abre e fecha
    with st.expander("🔍 Ver cadeia de raciocínio (Etapa 1 — Chain-of-Thought)"):  # Caixa recolhida: não polui a tela
        st.markdown(cadeia)                                             # Escreve o raciocínio dentro da caixa


def main():                                                             # Função principal: desenha a tela e controla a conversa
    st.set_page_config(page_title="Chat CoT — UTFPR", page_icon="🧠", layout="centered")  # Título da aba e layout

    if ARQUIVO_LOGO.exists():                                           # Se a imagem do logo está na pasta...
        st.image(str(ARQUIVO_LOGO), width=230)                          # ...mostra o logo da UTFPR no topo da tela
    st.title("Chat com Raciocínio Chain-of-Thought")                    # Título principal da página
    st.caption("Streamlit + Gemini · cada resposta passa por 2 etapas: raciocínio → síntese final")  # Subtítulo

    with st.sidebar:                                                    # Tudo dentro deste bloco aparece na barra lateral (à esquerda)
        if ARQUIVO_LOGO.exists():                                       # Se o logo existe...
            st.image(str(ARQUIVO_LOGO), width=160)                      # ...mostra o logo também na barra lateral
        st.header("⚙️ Painel")                                          # Título da barra lateral
        st.markdown(f"**Modelo:** `{MODELO}`  \n**Estratégia:** Chain-of-Thought (2 etapas)")  # Informações
        usar_exemplo = st.button("📊 Enviar pergunta de exemplo")        # Botão que envia o caso de uso pronto
        st.button("🗑️ Limpar conversa", on_click=limpar_conversa)       # Botão que reinicia a conversa e a memória
        st.caption("Limpar volta o chat ao início, com a memória inicial.")  # Explicação do botão

    if "mensagens" not in st.session_state:                             # Na primeira abertura (ou após limpar)...
        st.session_state.mensagens = [dict(m) for m in MEMORIA_INICIAL]  # ...a tela começa com a memória inicial

    for mensagem in st.session_state.mensagens:                         # Redesenha todas as mensagens já trocadas
        with st.chat_message(mensagem["role"]):                         # Balão de chat (usuário ou assistente)
            st.markdown(mensagem["content"])                            # Texto da mensagem
            if mensagem.get("cadeia"):                                  # Se a mensagem tem um raciocínio guardado...
                mostrar_cadeia(mensagem["cadeia"])                      # ...mostra a caixa com o raciocínio

    pergunta = st.chat_input("Digite sua pergunta ou referencie interações anteriores...")  # Campo de digitação
    if usar_exemplo:                                                    # Se o botão de exemplo foi clicado...
        pergunta = PERGUNTA_EXEMPLO                                     # ...usa a pergunta de exemplo como se o usuário tivesse digitado
    if not pergunta:                                                    # Se não há pergunta nova...
        return                                                          # ...não faz mais nada e espera o usuário

    st.session_state.mensagens.append({"role": "user", "content": pergunta})  # Guarda a pergunta no histórico da tela
    with st.chat_message("user"):                                       # Abre o balão do usuário
        st.markdown(pergunta)                                           # Mostra a pergunta

    with st.chat_message("assistant"):                                  # Abre o balão do assistente
        try:                                                            # Tenta executar a estratégia de raciocínio
            resultado = responder_com_chain_of_thought(pergunta)        # Executa as 2 etapas do Chain-of-Thought
        except ErroAmigavel as erro:                                    # Se deu um erro conhecido (chave, limite, conexão...)
            st.error(str(erro))                                         # Mostra a mensagem de erro em vermelho
            st.session_state.mensagens.pop()                            # Tira da tela a pergunta que ficou sem resposta
            return                                                      # Encerra esta rodada
        except Exception as erro:                                       # Se deu um erro inesperado
            st.error(f"Erro inesperado: {type(erro).__name__}: {erro}")  # Mostra o tipo e o detalhe do erro
            st.session_state.mensagens.pop()                            # Tira da tela a pergunta que ficou sem resposta
            return                                                      # Encerra esta rodada
        st.markdown(resultado["resposta"])                              # Mostra a resposta final
        mostrar_cadeia(resultado["cadeia"])                             # Mostra a caixa com o raciocínio da Etapa 1

    st.session_state.mensagens.append(                                  # Guarda a resposta no histórico da tela (para redesenhar depois)
        {"role": "assistant", "content": resultado["resposta"], "cadeia": resultado["cadeia"]}  # Resposta + raciocínio
    )                                                                   # Fim do registro


if __name__ == "__main__":                                              # Só executa quando o arquivo é rodado diretamente (streamlit run app.py)
    main()                                                              # Chama a função principal
