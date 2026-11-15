# Exportar um modelo para o Hugging Face Hub

No **Histórico**, abra uma execução individual com checkpoint e selecione a
exportação para o Hugging Face Hub. A prévia é local e funciona sem token.
Confira o model card, a lista de arquivos e a origem dos pesos. O checkpoint
é copiado como foi gravado: pode conter classes, caminhos ou outros metadados
internos. O arquivo de métricas é uma seleção do registro, não o run.json
completo.

Para habilitar o envio, o ambiente do servidor deve ter o extra `visionforge-studio[hub]`.
Salve um token de escrita com o menor escopo possível no perfil selecionado.
Em servidor compartilhado, qualquer pessoa que possa escolher esse perfil pode
usar, substituir ou remover o token. Prefira acesso HTTPS quando a interface
for acessada pela rede. Remover o token da interface apaga a cópia local;
para revogá-lo, use a conta do Hugging Face.

Informe o destino no formato usuário/modelo ou organização/modelo. Somente o
botão de envio publica os arquivos. O servidor recusa um destino público ou
de visibilidade desconhecida. Se o pacote mudar após a prévia, abra outra
prévia e confira os arquivos antes de confirmar. A exportação não transforma
os pesos em um modelo compatível com Transformers.
