"""Texts the server writes for people, in the language the page asked for (ADR-116).

The interface is bilingual (ADR-110), but a message the server composes itself
used to arrive in whichever language its author was thinking in. This module is
the one place those messages live: ``key -> {pt, en}``, each a ``str.format``
template, no gettext.

The language of a request is a ``ContextVar``. The GUI binds it from the
``X-VF-Lang`` header on every API call, and a queued job re-binds the language it
was submitted under, so a message written deep inside a training thread (a health
warning, the paging-file hint) is worded for the person who started the run. Code
that never ran under a request (the command line, a script) gets Portuguese,
which is what it always printed.

It lives in ``utils`` rather than in ``gui`` because ``core`` writes some of these
messages and must not import the web layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

Lang = Literal["pt", "en"]

DEFAULT_LANG: Lang = "pt"
LANG_HEADER = "X-VF-Lang"

_SUPPORTED: tuple[Lang, ...] = ("pt", "en")

_LANG: ContextVar[Lang] = ContextVar("visionforge_lang", default=DEFAULT_LANG)


def resolve_lang(value: str | None) -> Lang:
    """The language an ``X-VF-Lang`` value asks for; absent or unknown is Portuguese.

    Only the primary subtag counts, so ``en-US`` and ``EN`` are English.
    """
    if value:
        primary = value.strip().lower().replace("_", "-").split("-", 1)[0]
        for lang in _SUPPORTED:
            if primary == lang:
                return lang
    return DEFAULT_LANG


def current_lang() -> Lang:
    """The language messages are written in right now."""
    return _LANG.get()


def set_lang(lang: Lang) -> None:
    """Bind the language for the rest of this task and the threads it starts."""
    _LANG.set(lang)


@contextmanager
def using_lang(lang: Lang) -> Iterator[None]:
    """Write messages in ``lang`` inside the ``with`` block."""
    token = _LANG.set(lang)
    try:
        yield
    finally:
        _LANG.reset(token)


# Templates are written once per language, never assembled from fragments: the
# other language wants the parts in another order. Braces that are meant
# literally are doubled.
CATALOG: dict[str, dict[Lang, str]] = {
    # ── training health (core/training_health.py) ────────────────────────────
    "health.value_label": {"pt": "valor", "en": "value"},
    "health.collapsed_predictions": {
        "pt": (
            "O modelo previu a mesma classe ({cls}) para todas as {n} imagens de "
            "validação — ele não aprendeu a distinguir as classes. A acurácia "
            "mostrada é apenas a proporção dessa classe. Causa mais comum: "
            "learning rate alto demais para esta arquitetura (VGG e AlexNet com "
            "Adam costumam precisar de 1e-4, não 1e-3). Tente reduzir o learning "
            "rate ou trocar o otimizador para SGD."
        ),
        "en": (
            "The model predicted the same class ({cls}) for all {n} validation "
            "images — it did not learn to tell the classes apart. The accuracy "
            "shown is only the share of that class. Most common cause: a learning "
            "rate too high for this architecture (VGG and AlexNet with Adam "
            "usually need 1e-4, not 1e-3). Try lowering the learning rate or "
            "switching the optimizer to SGD."
        ),
    },
    "health.stagnant_loss": {
        "pt": (
            "A loss de treino praticamente não caiu ({first:.4f} → {last:.4f} em "
            "{n} épocas). O modelo não está aprendendo: revise o learning rate "
            "(alto demais diverge, baixo demais não sai do lugar) e confira se os "
            "rótulos do dataset estão corretos."
        ),
        "en": (
            "The training loss barely moved ({first:.4f} → {last:.4f} over {n} "
            "epochs). The model is not learning: review the learning rate (too "
            "high diverges, too low goes nowhere) and check that the dataset "
            "labels are correct."
        ),
    },
    "health.constant_predictions": {
        "pt": (
            "O modelo previu praticamente o mesmo {label} ({mean:.4f}) para todas "
            "as entradas — ele está chutando a média em vez de usar a imagem. "
            "Revise o learning rate e a normalização dos alvos."
        ),
        "en": (
            "The model predicted practically the same {label} ({mean:.4f}) for "
            "every input — it is guessing the mean instead of using the image. "
            "Review the learning rate and the target normalisation."
        ),
    },
    "health.frozen_random_backbone": {
        "pt": (
            "Feature extraction congelou {frozen} pesos que nunca foram treinados "
            "(o modelo está sem pesos pré-treinados). Congelar faz sentido quando "
            "os pesos já aprenderam algo; aqui eles são aleatórios. Ative os "
            "pesos pré-treinados, ou use fine-tuning para treinar a rede inteira."
        ),
        "en": (
            "Feature extraction froze {frozen} weights that were never trained "
            "(the model has no pretrained weights). Freezing makes sense when the "
            "weights already learned something; here they are random. Turn on the "
            "pretrained weights, or use fine-tuning to train the whole network."
        ),
    },
    "health.collapsed_segmentation": {
        "pt": (
            "A segmentação previu uma única classe em todos os pixels — as outras "
            "ficaram com IoU zero. O mIoU mostrado é o dessa classe dividido pelo "
            "número de classes, não uma medida de qualidade. Revise o learning "
            "rate e verifique se as máscaras têm os índices de classe esperados."
        ),
        "en": (
            "The segmentation predicted a single class on every pixel — the others "
            "got an IoU of zero. The mIoU shown is that class divided by the "
            "number of classes, not a measure of quality. Review the learning rate "
            "and check that the masks have the expected class indices."
        ),
    },
    "health.no_detections": {
        "pt": (
            "O modelo terminou {epochs} época(s) com mAP@50 igual a zero: ele não "
            "acertou nenhuma caixa. Confira se os rótulos estão no formato YOLO "
            "esperado e se o número de classes bate com o data.yaml; depois "
            "disso, o learning rate."
        ),
        "en": (
            "The model finished {epochs} epoch(s) with mAP@50 equal to zero: it "
            "did not hit a single box. Check that the labels are in the expected "
            "YOLO format and that the number of classes matches data.yaml; after "
            "that, the learning rate."
        ),
    },
    # ── Windows paging file (WinError 1455) ──────────────────────────────────
    "winerror.paging_file": {
        "pt": (
            "Windows ficou sem espaço de paginação ao criar os processos de "
            "leitura de dados (WinError 1455). Cada worker é um processo novo que "
            "recarrega o torch e as DLLs da CUDA, ~1 GB cada.\n"
            "  - Reduza data.num_workers (2, ou 0 para desligar).\n"
            "  - Feche runs anteriores que tenham ficado presos e o que estiver "
            "ocupando memória.\n"
            "  - Ou aumente o arquivo de paginação do Windows "
            "(Sistema → Configurações avançadas → Desempenho → Memória virtual)."
        ),
        "en": (
            "Windows ran out of paging file space while starting the data-loading "
            "processes (WinError 1455). Each worker is a new process that reloads "
            "torch and the CUDA DLLs, ~1 GB each.\n"
            "  - Lower data.num_workers (2, or 0 to turn it off).\n"
            "  - Close earlier runs that got stuck and whatever is using memory.\n"
            "  - Or increase the Windows paging file "
            "(System → Advanced settings → Performance → Virtual memory)."
        ),
    },
    "winerror.workers_died": {
        "pt": (
            "Os workers do DataLoader morreram ao iniciar — no Windows isso quase "
            "sempre é o arquivo de paginação pequeno demais para {workers} "
            "workers recarregarem as DLLs CUDA do torch (WinError 1455). Reduza "
            "training.workers para 0–2, ou aumente a memória virtual do Windows "
            "(Sistema → Configurações avançadas → Desempenho → Memória virtual)."
        ),
        "en": (
            "The DataLoader workers died on startup — on Windows this is almost "
            "always a paging file too small for {workers} workers to reload "
            "torch's CUDA DLLs (WinError 1455). Lower training.workers to 0–2, or "
            "increase the Windows virtual memory (System → Advanced settings → "
            "Performance → Virtual memory)."
        ),
    },
    # ── native file and folder dialogs ───────────────────────────────────────
    "pick.title_checkpoint": {
        "pt": "Selecione um checkpoint (.pth ou .pt)",
        "en": "Select a checkpoint (.pth or .pt)",
    },
    "pick.title_yaml": {
        "pt": "Selecione o data.yaml do dataset",
        "en": "Select the dataset's data.yaml",
    },
    "pick.title_folder": {
        "pt": "Selecione o diretório base do dataset",
        "en": "Select the dataset's base directory",
    },
    "pick.failed": {
        "pt": "Falha ao abrir o seletor: {error}",
        "en": "Could not open the picker: {error}",
    },
    "pick.no_tkinter": {
        "pt": "O tkinter não está disponível nesta instalação do Python.",
        "en": "tkinter is not available on this Python installation.",
    },
    "pick.container": {
        "pt": (
            "O seletor nativo não abre dentro do container (sem display). "
            "Digite o caminho montado, por exemplo /work/datasets/meu-dataset."
        ),
        "en": (
            "The native picker does not open inside the container (no display). "
            "Type the mounted path, for example /work/datasets/my-dataset."
        ),
    },
    # ── dataset scans: split detection ───────────────────────────────────────
    "detect.path_empty": {
        "pt": "Informe o caminho do diretório base do dataset.",
        "en": "Enter the path of the dataset's base directory.",
    },
    "detect.not_found": {
        "pt": "O diretório '{path}' não foi encontrado no disco.",
        "en": "The directory '{path}' was not found on disk.",
    },
    "detect.not_a_folder": {
        "pt": "O caminho '{path}' existe mas não é uma pasta.",
        "en": "The path '{path}' exists but is not a folder.",
    },
    "detect.no_permission": {
        "pt": "Sem permissão de leitura em '{path}'.",
        "en": "No read permission on '{path}'.",
    },
    "detect.no_subfolders": {
        "pt": (
            "Nenhuma subpasta encontrada em '{path}'. Esperado pastas separando "
            "treino, validação e teste."
        ),
        "en": (
            "No subfolders found in '{path}'. Expected folders separating "
            "training, validation and test."
        ),
    },
    "detect.role_train": {"pt": "treino", "en": "training"},
    "detect.role_val": {"pt": "validação", "en": "validation"},
    "detect.role_test": {"pt": "teste", "en": "test"},
    "detect.partial": {
        "pt": (
            "Detectado parcialmente. Faltando: {missing}. Selecione manualmente "
            "as pastas restantes."
        ),
        "en": (
            "Partially detected. Missing: {missing}. Select the remaining folders "
            "by hand."
        ),
    },
    "detect.found": {
        "pt": (
            "Splits detectados: treino='{train}', validação='{val}', teste='{test}'."
        ),
        "en": (
            "Splits detected: training='{train}', validation='{val}', test='{test}'."
        ),
    },
    "detect.unrecognized": {
        "pt": (
            "Não foi possível identificar automaticamente os splits. Subpastas "
            "encontradas: {found}. Selecione manualmente qual é treino / "
            "validação / teste."
        ),
        "en": (
            "The splits could not be identified automatically. Subfolders found: "
            "{found}. Select by hand which one is training / validation / test."
        ),
    },
    # ── dataset scans: stats, samples, previews ──────────────────────────────
    "scan.base_missing": {
        "pt": "Diretório base não encontrado: {path}",
        "en": "Base directory not found: {path}",
    },
    "scan.dir_missing": {
        "pt": "Diretório não encontrado: {path}",
        "en": "Directory not found: {path}",
    },
    "scan.split_missing": {
        "pt": "Split '{split}' não encontrado.",
        "en": "Split '{split}' not found.",
    },
    "scan.split_missing_in": {
        "pt": "Split '{split}' não encontrado em {path}.",
        "en": "Split '{split}' not found in {path}.",
    },
    "scan.split_missing_dataset": {
        "pt": "Split '{split}' não encontrado neste dataset.",
        "en": "Split '{split}' not found in this dataset.",
    },
    "scan.split_no_classes": {
        "pt": "Nenhuma classe encontrada no split.",
        "en": "No class found in the split.",
    },
    "scan.class_no_images": {
        "pt": "Sem imagens no diretório {path}.",
        "en": "No images in the directory {path}.",
    },
    "scan.split_no_images": {
        "pt": "Sem imagens no split '{split}'.",
        "en": "No images in split '{split}'.",
    },
    "scan.no_boxes": {
        "pt": "Nenhuma caixa anotada encontrada neste split.",
        "en": "No annotated box found in this split.",
    },
    "scan.no_yolo_split": {
        "pt": (
            "Nenhum split YOLO encontrado (esperado 'images/<split>' "
            "ou '<split>/images')."
        ),
        "en": ("No YOLO split found (expected 'images/<split>' or '<split>/images')."),
    },
    "scan.no_seg_split": {
        "pt": "Nenhum split encontrado (esperado <split>/{{imagens,máscaras}}).",
        "en": "No split found (expected <split>/{{images,masks}}).",
    },
    "scan.anomaly_train_missing": {
        "pt": "Pasta de treino normal não encontrada: {train_dir}/{normal_dir}",
        "en": "Normal training folder not found: {train_dir}/{normal_dir}",
    },
    "scan.anomaly_train_empty": {
        "pt": "Nenhuma imagem normal no treino — o treino falharia.",
        "en": "No normal image in training — the training would fail.",
    },
    "scan.no_manifest": {
        "pt": "Nenhum CSV de manifest encontrado na pasta base.",
        "en": "No manifest CSV found in the base folder.",
    },
    # ── testing a finished run on another folder ─────────────────────────────
    "testrun.path_not_found": {
        "pt": "Caminho não encontrado: {path}",
        "en": "Path not found: {path}",
    },
    "testrun.regression_needs_csv": {
        "pt": (
            "Regressão avalia um manifesto: escolha o arquivo .csv com a coluna "
            "de imagem e a(s) coluna(s) alvo, não uma pasta."
        ),
        "en": (
            "Regression is scored on a manifest: choose the .csv file with the "
            "image column and the target column(s), not a folder."
        ),
    },
    "testrun.no_checkpoint": {
        "pt": "Run '{run}' não tem um checkpoint utilizável (artifacts.model: {model}).",
        "en": "Run '{run}' has no usable checkpoint (artifacts.model: {model}).",
    },
    "testrun.no_rows": {
        "pt": "Nenhuma linha utilizável em {path}.",
        "en": "No usable row in {path}.",
    },
    "testrun.no_pairs": {
        "pt": "Nenhum par imagem/máscara encontrado em {path}.",
        "en": "No image/mask pair found in {path}.",
    },
    "testrun.masks_missing": {
        "pt": (
            "A pasta {split} tem '{images}', mas não tem a pasta de máscaras "
            "'{masks}' ({masks_path}). Segmentação pareia cada imagem com a "
            "máscara de mesmo nome."
        ),
        "en": (
            "The folder {split} has '{images}' but not the masks folder "
            "'{masks}' ({masks_path}). Segmentation pairs each image with the "
            "mask of the same name."
        ),
    },
    "testrun.anomaly_train_missing": {
        "pt": (
            "A detecção de anomalias calibra o limiar com as imagens normais de "
            "treino, e elas não existem em {path}. Escolha uma pasta de teste que "
            "esteja ao lado da pasta '{train_dir}' (layout MVTec: "
            "{train_dir}/{normal_dir}/ e a pasta de teste na mesma pasta pai)."
        ),
        "en": (
            "Anomaly detection calibrates its threshold on the normal training "
            "images, and they do not exist in {path}. Choose a test folder that "
            "sits next to the '{train_dir}' folder (MVTec layout: "
            "{train_dir}/{normal_dir}/ and the test folder under the same parent)."
        ),
    },
    "export.torchvision_detection": {
        "pt": (
            "Export ONNX para detectores torchvision ainda não é suportado; "
            "use o backend Ultralytics."
        ),
        "en": (
            "ONNX export for torchvision detectors is not supported yet; use the "
            "Ultralytics backend."
        ),
    },
    "testrun.batch_unsupported": {
        "pt": "Inferência em lote não é suportada para runs de '{task}' ainda.",
        "en": "Batch inference is not supported for '{task}' runs yet.",
    },
    # ── runs and the queue ───────────────────────────────────────────────────
    "run.not_found": {
        "pt": "Run '{run_id}' não encontrado.",
        "en": "Run '{run_id}' not found.",
    },
    "run.still_running": {
        "pt": "O experimento ainda está em execução.",
        "en": "Experiment is still running.",
    },
    "run.failed": {
        "pt": "O experimento falhou: {error}",
        "en": "Experiment failed: {error}",
    },
    "run.no_message": {"pt": "(sem mensagem)", "en": "(no message)"},
    "run.resuming_label": {
        "pt": "{name} (retomando)",
        "en": "{name} (resuming)",
    },
    "run.nothing_to_continue": {
        "pt": "O run '{run_id}' não tem mais nada a continuar.",
        "en": "Run '{run_id}' has nothing left to continue.",
    },
    "run.cannot_delete_active": {
        "pt": "Não é possível excluir um run que está em execução.",
        "en": "Cannot delete a run that is currently executing.",
    },
    "run.group_action": {
        "pt": (
            "Esta execução é um conjunto de réplicas: abra uma das seeds para "
            "continuar, testar, prever em lote, exportar ou gerar Grad-CAM."
        ),
        "en": (
            "This run is a set of replicates: open one of the seeds to continue, "
            "test, predict in batch, export or generate Grad-CAM."
        ),
    },
    "queue.not_stoppable": {
        "pt": (
            "Esta execução não verifica pedidos de parada (uma tarefa própria que "
            "controla o próprio laço de treino, por exemplo) e vai rodar até o "
            "fim. Nada foi interrompido."
        ),
        "en": (
            "This run does not check for stop requests (a custom task that owns "
            "its own training loop, for example) and will run to the end. Nothing "
            "was interrupted."
        ),
    },
    "queue.unknown_id": {
        "pt": "Nenhum run na fila com esse id — talvez ele já tenha começado.",
        "en": "No queued run with that id — it may have already started.",
    },
    # ── "Open folder" ────────────────────────────────────────────────────────
    "reveal.foreign_origin": {
        "pt": "A pasta só pode ser aberta pela página do próprio VisionForge.",
        "en": "The folder can only be opened from VisionForge's own page.",
    },
    "reveal.remote_client": {
        "pt": "A pasta só pode ser aberta a partir da máquina que executa o servidor.",
        "en": "The folder can only be opened from the machine running the server.",
    },
    "reveal.failed": {
        "pt": "Não foi possível abrir a pasta: {error}",
        "en": "Failed to open the folder: {error}",
    },
    # ── profiles (ADR-114) ───────────────────────────────────────────────────
    "profile.invalid": {
        "pt": (
            "'{slug}' não é um nome de perfil válido: use de 1 a {max_len} letras "
            "ascii minúsculas, dígitos, '-' ou '_'."
        ),
        "en": (
            "'{slug}' is not a valid profile name: use 1-{max_len} lowercase "
            "ascii letters, digits, '-' or '_'."
        ),
    },
    "profile.unknown": {
        "pt": "O perfil '{slug}' não existe.",
        "en": "Profile '{slug}' does not exist.",
    },
    "profile.unresolvable": {
        "pt": "Não foi possível resolver o perfil '{slug}'.",
        "en": "Profile '{slug}' cannot be resolved.",
    },
    "profile.outside_root": {
        "pt": "O perfil '{slug}' está fora da pasta de perfis.",
        "en": "Profile '{slug}' is outside the profiles folder.",
    },
    "profile.linked_subfolder": {
        "pt": "O perfil '{slug}' guarda {sub} fora de si.",
        "en": "Profile '{slug}' keeps its {sub} outside itself.",
    },
    "profile.name_empty": {
        "pt": "O nome precisa ter ao menos uma letra (a-z) ou um número.",
        "en": "The name needs at least one letter (a-z) or digit.",
    },
    "profile.name_reserved": {
        "pt": "'{slug}' é um nome reservado. Escolha outro.",
        "en": "'{slug}' is a reserved name. Choose another.",
    },
    "profile.exists": {
        "pt": "Já existe um perfil '{name}' (pasta '{slug}').",
        "en": "A profile '{name}' already exists (folder '{slug}').",
    },
    "profile.create_failed": {
        "pt": "Não foi possível criar a pasta do perfil '{slug}': {reason}.",
        "en": "Could not create the folder of profile '{slug}': {reason}.",
    },
    "profile.output_locked": {
        "pt": (
            "A pasta de saída é definida pelo perfil e não pode ser alterada "
            "{where}: remova '{path}'."
        ),
        "en": (
            "The output folder is set by the profile and cannot be changed "
            "{where}: remove '{path}'."
        ),
    },
    "profile.where_comparison": {"pt": "numa comparação", "en": "in a comparison"},
    "profile.where_sweep": {"pt": "numa varredura", "en": "in a sweep"},
}


def message(lang: Lang, key: str, /, **params: object) -> str:
    """The catalog text ``key`` in ``lang`` with ``params`` filled in."""
    return CATALOG[key][lang].format(**params)


def tr(key: str, /, **params: object) -> str:
    """The catalog text ``key`` in the language of the current request."""
    return message(current_lang(), key, **params)


__all__ = [
    "CATALOG",
    "DEFAULT_LANG",
    "LANG_HEADER",
    "Lang",
    "current_lang",
    "message",
    "resolve_lang",
    "set_lang",
    "tr",
    "using_lang",
]
