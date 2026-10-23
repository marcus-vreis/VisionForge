# Nomes de experimentos e diretórios de grupos

O nome identifica uma pasta dentro da raiz de modelos configurada. Use um único
componente, como `cafe_resnet18`, `Café trial 1` ou `resnet-18.v2`.

O backend recusa nomes vazios, `.` e `..`, separadores `/` e `\`, caracteres de
controle e os caracteres reservados `< > : " | ? *`. O nome não pode terminar
com ponto ou espaço nem usar nomes de dispositivos Windows, como `CON`, `NUL`,
`COM1` e `LPT1`, inclusive com extensão. A mesma regra vale para tarefas customizadas
e identificadores de grupos. A validação ocorre antes da submissão de um treino.

Nomes antigos que violem essas regras não são renomeados automaticamente. Seus
arquivos permanecem no disco; configurações com nomes inválidos precisam ser
corrigidas antes de uma nova execução ou retomada.

Essa regra não restringe a raiz escolhida em `output.models_dir`: ela pode estar
fora de `outputs/`, em outro disco ou em uma pasta de armazenamento ligada por
symlink ou junction. Os metadados de cada grupo e seus filhos devem permanecer
dentro da raiz autorizada depois da resolução desses aliases.

Antes de alterar qualquer filho, o writer valida todos os destinos de `run.json`
e seus arquivos temporários. Um alias que escape da raiz ou metadados com múltiplos
hard links interrompem a escrita sem tagging parcial. A checagem não constitui
uma garantia contra substituição concorrente de diretórios por outro processo.
