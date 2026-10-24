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
which is what it always printed, unless ``VISIONFORGE_LANG`` says otherwise.

The text written into the figures a run saves is the one exception: a figure is
a publication artifact and used to be English, so ``tr_artifact`` words it in
English when no language is bound at all (a script, the command line) and in the
bound language otherwise. The command line can bind one with ``VISIONFORGE_LANG``.

It lives in ``utils`` rather than in ``gui`` because ``core`` writes some of these
messages and must not import the web layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

Lang = Literal["pt", "en"]

DEFAULT_LANG: Lang = "pt"
# What an artifact is worded in when nobody chose (see ``tr_artifact``).
ARTIFACT_DEFAULT_LANG: Lang = "en"
LANG_HEADER = "X-VF-Lang"
LANG_ENV = "VISIONFORGE_LANG"

_SUPPORTED: tuple[Lang, ...] = ("pt", "en")

# ``None`` is "nobody chose", which is not the same as choosing Portuguese: the
# figures of an unbound run are English, those of a Portuguese page are not.
_LANG: ContextVar[Lang | None] = ContextVar("visionforge_lang", default=None)


def parse_lang(value: str | None) -> Lang | None:
    """The language a header or environment value names, or ``None`` if it names none.

    Only the primary subtag counts, so ``en-US`` and ``EN`` are English.
    """
    if value:
        primary = value.strip().lower().replace("_", "-").split("-", 1)[0]
        for lang in _SUPPORTED:
            if primary == lang:
                return lang
    return None


def resolve_lang(value: str | None) -> Lang:
    """The language an ``X-VF-Lang`` value asks for; absent or unknown is Portuguese."""
    return parse_lang(value) or DEFAULT_LANG


def lang_from_env(environ: Mapping[str, str]) -> Lang | None:
    """The language ``VISIONFORGE_LANG`` selects; unset, empty or unknown selects none."""
    return parse_lang(environ.get(LANG_ENV))


def current_lang() -> Lang:
    """The language messages are written in right now."""
    return _LANG.get() or DEFAULT_LANG


def bound_lang() -> Lang | None:
    """The language somebody bound, or ``None`` when nobody did."""
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
    "gpu.context_broken": {
        "pt": (
            "A GPU deste servidor parou de aceitar trabalho depois de um erro de CUDA "
            "num treino anterior (veja o erro dele no histórico). Feche e abra o "
            "VisionForge de novo para continuar treinando."
        ),
        "en": (
            "This server's GPU stopped accepting work after a CUDA error in an "
            "earlier run (see its error in the history). Close and reopen "
            "VisionForge to keep training."
        ),
    },
    "segmentation.mask_id_out_of_range": {
        "pt": (
            "Há máscaras com o id de classe {found}, mas o modelo tem num_classes = "
            "{num_classes} (ids de 0 a {last}). Aumente num_classes para pelo menos "
            "{needed} ou marque esses pixels com o ignore_index ({ignore})."
        ),
        "en": (
            "Some masks contain class id {found}, but the model has num_classes = "
            "{num_classes} (ids 0 to {last}). Raise num_classes to at least "
            "{needed}, or mark those pixels with the ignore_index ({ignore})."
        ),
    },
    "scan.no_seg_split": {
        "pt": "Nenhum split encontrado (esperado <split>/{{imagens,máscaras}}).",
        "en": "No split found (expected <split>/{{images,masks}}).",
    },
    # ── dataset checks at submit time (the run is refused before it is queued) ──
    "submit.detection_empty_train": {
        "pt": "O split de treino do dataset de detecção não existe ou não tem nenhuma imagem.",
        "en": "The detection dataset's train split does not exist or has no images.",
    },
    "submit.segmentation_no_train_dir": {
        "pt": "Pasta de treino da segmentação não encontrada: {path}",
        "en": "Segmentation train folder not found: {path}",
    },
    "submit.segmentation_no_pairs": {
        "pt": (
            "O split de treino não tem nenhum par imagem↔máscara (imagens: {images}, "
            "máscaras: {masks}); os nomes dos arquivos, sem a extensão, precisam coincidir."
        ),
        "en": (
            "The train split has no image-mask pair (images: {images}, masks: {masks}); "
            "file names, without the extension, must match."
        ),
    },
    "submit.regression_no_train_csv": {
        "pt": "CSV de treino da regressão não encontrado: {path}",
        "en": "Regression train CSV not found: {path}",
    },
    "submit.regression_missing_columns": {
        "pt": "Colunas ausentes em {csv}: {missing}. Colunas encontradas: {found}.",
        "en": "Columns missing from {csv}: {missing}. Columns found: {found}.",
    },
    "submit.regression_empty_manifest": {
        "pt": "O CSV de treino {csv} está sem linhas de dados.",
        "en": "The train CSV {csv} has no data rows.",
    },
    "submit.num_classes_mismatch": {
        "pt": (
            "num_classes={num_classes}, mas a pasta de treino tem {found} pastas de "
            "classe ({names}). Ajuste num_classes para {found} ou corrija as pastas."
        ),
        "en": (
            "num_classes={num_classes}, but the train folder has {found} class folders "
            "({names}). Set num_classes to {found} or fix the folders."
        ),
    },
    "submit.binary_folders": {
        "pt": (
            "A tarefa binária precisa de exatamente 2 pastas de classe no treino "
            "(1 saída para 2 pastas), mas há {found}: {names}."
        ),
        "en": (
            "A binary task needs exactly 2 class folders in train (1 output for 2 "
            "folders), but there are {found}: {names}."
        ),
    },
    # ── regression manifest rows (raised while the dataset loads) ──
    "regdata.manifest_not_found": {
        "pt": "Manifest de regressão não encontrado: {path}",
        "en": "Regression manifest not found: {path}",
    },
    "regdata.missing_columns": {
        "pt": "O manifest '{path}' não tem a(s) coluna(s): {missing}. Colunas encontradas: {found}.",
        "en": "Manifest '{path}' is missing column(s): {missing}. Found: {found}.",
    },
    "regdata.empty_image": {
        "pt": "Manifest '{path}', linha {line}: valor vazio em '{column}'.",
        "en": "Manifest '{path}' row {line}: empty '{column}' value.",
    },
    "regdata.non_numeric_target": {
        "pt": "Manifest '{path}', linha {line}: alvo não numérico ({error}).",
        "en": "Manifest '{path}' row {line}: non-numeric target ({error}).",
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
    "export.no_checkpoint": {
        "pt": "Run '{run}' não tem checkpoint registrado em artifacts.model.",
        "en": "Run '{run}' has no checkpoint recorded in artifacts.model.",
    },
    # ── Grad-CAM (gui/api/torch_gradcam.py, core/gradcam.py) ─────────────────
    "gradcam.unsupported_task": {
        "pt": (
            "Grad-CAM não está disponível para a tarefa '{task}' (suportadas: "
            "classification, regression, segmentation)."
        ),
        "en": (
            "Grad-CAM is not available for task '{task}' (supported: "
            "classification, regression, segmentation)."
        ),
    },
    "gradcam.no_conv_layer": {
        "pt": "O modelo não tem uma camada nn.Conv2d para o Grad-CAM usar.",
        "en": "Model has no nn.Conv2d layer to target for Grad-CAM.",
    },
    "testrun.batch_unsupported": {
        "pt": "Inferência em lote não é suportada para runs de '{task}' ainda.",
        "en": "Batch inference is not supported for '{task}' runs yet.",
    },
    # ── runs and the queue ───────────────────────────────────────────────────
    "evidence.audit_failed": {
        "pt": "Não foi possível auditar as imagens: {error}",
        "en": "Could not audit images: {error}",
    },
    "evidence.bundle_failed": {
        "pt": "Não foi possível exportar o experimento: {error}",
        "en": "Could not export the experiment: {error}",
    },
    "evidence.run_busy": {
        "pt": "Este run está em execução ou aguarda retomada. Exporte após o término.",
        "en": "This run is active or queued for continuation. Export after it finishes.",
    },
    "evidence.export_busy": {
        "pt": "O experimento está sendo exportado. Aguarde o término antes de continuar.",
        "en": "The experiment is being exported. Wait for it to finish before continuing.",
    },
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
    "profile.save_failed": {
        "pt": "Não foi possível salvar o perfil '{slug}': {reason}.",
        "en": "Could not save profile '{slug}': {reason}.",
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
    # ── sample dataset of the guides (ADR-115, ADR-117) ──────────────────────
    "sample.unknown_task": {
        "pt": "Tarefa desconhecida '{task}'.",
        "en": "Unknown task '{task}'.",
    },
    "sample.exists": {
        "pt": (
            "{path} já existe e não está vazia: use-a como dataset ou remova-a "
            "para gerar um novo."
        ),
        "en": (
            "{path} already exists and is not empty; use it as the dataset or "
            "remove it to generate a new one."
        ),
    },
    "sample.write_failed": {
        "pt": "Não foi possível gravar o dataset de exemplo: {error}",
        "en": "Could not write the sample dataset: {error}",
    },
    # ── the rest of the run routes' own refusals ─────────────────────────────
    "run.bad_json": {
        "pt": "Não foi possível ler o run.json: {error}",
        "en": "Failed to parse run.json: {error}",
    },
    "run.cannot_resume_task": {
        "pt": "A tarefa '{task}' não pode ser retomada.",
        "en": "Task '{task}' cannot be resumed.",
    },
    "run.stored_config_invalid": {
        "pt": "A configuração guardada de '{run_id}' não é mais válida: {error}",
        "en": "The stored config of '{run_id}' is no longer valid: {error}",
    },
    "run.delete_outside": {
        "pt": "Exclusão recusada: o caminho está fora de {models_dir}: {path}",
        "en": "Refused to delete path outside {models_dir}: {path}",
    },
    "run.dir_missing": {
        "pt": "A pasta do run '{path}' não existe.",
        "en": "Run directory '{path}' does not exist.",
    },
    "run.custom_action": {
        "pt": (
            "'{action}' não está disponível para a tarefa própria '{task}': esta "
            "ação reconstrói uma arquitetura pronta a partir da config do run, e "
            "uma tarefa própria define o próprio modelo. Avalie-a executando a "
            "tarefa de novo sobre outro split."
        ),
        "en": (
            "'{action}' is not available for the researcher-defined task "
            "'{task}': this action rebuilds a built-in architecture from the "
            "run's config, and a custom task owns its own model. Evaluate it by "
            "re-running the task against another split."
        ),
    },
    "compare.seeds_unique": {
        "pt": "As seeds da comparação precisam ser distintas.",
        "en": "Comparison seeds must be unique.",
    },
    "replicates.seeds_unique": {
        "pt": "As seeds das réplicas precisam ser distintas.",
        "en": "Replicate seeds must be unique.",
    },
    # ── files served to the page (image preview, artifacts) ──────────────────
    "file.traversal": {
        "pt": "O caminho contém segmentos de navegação ('..').",
        "en": "Path contains traversal segments.",
    },
    "file.not_absolute": {
        "pt": "O caminho precisa ser absoluto.",
        "en": "Path must be absolute.",
    },
    "file.not_found": {"pt": "Arquivo não encontrado.", "en": "File not found."},
    "file.not_image": {
        "pt": "Só é possível pré-visualizar arquivos de imagem.",
        "en": "Only image files are previewable.",
    },
    "file.not_in_dataset": {
        "pt": (
            "O caminho não está em nenhuma pasta de dataset consultada nesta sessão."
        ),
        "en": "Path is not inside any dataset directory probed this session.",
    },
    "file.access_denied": {"pt": "Acesso negado.", "en": "Access denied."},
    # ── updating the install from the page (gui/api/routes.py, ADR-120) ──────
    "update.foreign_origin": {
        "pt": "Atualização recusada: o pedido veio de uma página de outro site.",
        "en": "Update refused: the request came from a page of another site.",
    },
    "update.none": {
        "pt": "Não há atualização disponível.",
        "en": "No update is available.",
    },
    "update.blocked_source": {
        "pt": "Esta instalação roda do código-fonte; atualize com git pull.",
        "en": "This install runs from a source checkout; update it with git pull.",
    },
    "update.blocked_no_installer": {
        "pt": "Este ambiente não tem pip nem uv para instalar a atualização.",
        "en": "This environment has neither pip nor uv to install the update.",
    },
    "update.blocked_remote": {
        "pt": "Só quem está na máquina do servidor pode atualizá-lo.",
        "en": "Only someone at the server's own machine can update it.",
    },
    "update.blocked_busy": {
        "pt": "Há um treino rodando ou na fila; atualize quando a fila esvaziar.",
        "en": "A training is running or queued; update once the queue is empty.",
    },
    "update.spawn_failed": {
        "pt": "Não foi possível iniciar a atualização: {error}",
        "en": "Could not start the update: {error}",
    },
    # ── researcher-defined tasks: hide, restore, delete ──────────────────────
    "custom.hidden": {
        "pt": "Aba ocultada. O arquivo continua em user_tasks/.",
        "en": "Tab hidden. The file is still in user_tasks/.",
    },
    "custom.already_hidden": {
        "pt": "Já estava oculta.",
        "en": "It was already hidden.",
    },
    "custom.unhidden": {"pt": "Aba restaurada.", "en": "Tab restored."},
    "custom.deleted": {
        "pt": "Removido do disco: {removed}",
        "en": "Removed from disk: {removed}",
    },
    "custom.confirm_mismatch": {
        "pt": (
            "A confirmação precisa repetir exatamente a chave da tarefa: "
            "esperado '{key}'."
        ),
        "en": "Confirmation must repeat the task key exactly: expected '{key}'.",
    },
    "custom.no_file": {
        "pt": "Nenhum arquivo de tarefa encontrado para '{key}'.",
        "en": "No task file found for '{key}'.",
    },
    "custom.not_registered": {
        "pt": (
            "A tarefa customizada '{key}' não está registrada (disponíveis: "
            "{available}). Defina-a com @register_task num arquivo .py em {where}/."
        ),
        "en": (
            "Custom task '{key}' is not registered (available: {available}). "
            "Define it with @register_task in a .py file under {where}/."
        ),
    },
    "custom.none_available": {"pt": "nenhuma", "en": "none"},
    # ── dataset download (gui/api/dataset_download.py) ───────────────────────
    "download.unknown_torchvision": {
        "pt": "Dataset do torchvision desconhecido: '{name}'. Disponíveis: {available}.",
        "en": "Unknown torchvision dataset '{name}'. Available: {available}.",
    },
    "download.no_images": {
        "pt": (
            "{provider} terminou, mas nenhuma imagem chegou em '{root}'. A "
            "exportação em si deu certo, então a causa mais comum é uma versão "
            "vazia ou sem anotações de '{dataset}' — abra-a no provedor e confira "
            "se a versão pedida tem imagens."
        ),
        "en": (
            "{provider} finished but no images landed in '{root}'. The export "
            "itself succeeded, so the usual cause is an empty or unannotated "
            "version of '{dataset}' — open it in the provider and check that the "
            "version you asked for has images."
        ),
    },
    "download.stray_extensions": {
        "pt": "Arquivos encontrados com estas extensões: {extensions}.",
        "en": "Found files with these extensions: {extensions}.",
    },
    "download.roboflow_format": {
        "pt": (
            "O dataset do Roboflow precisa ser 'workspace/project' — cole a URL "
            "do projeto ou o par workspace/project."
        ),
        "en": (
            "Roboflow dataset must be 'workspace/project' — paste the project "
            "URL or the workspace/project pair."
        ),
    },
    "download.roboflow_api_key": {
        "pt": "O Roboflow precisa de uma api_key.",
        "en": "Roboflow requires an api_key.",
    },
    "download.roboflow_version": {
        "pt": "O Roboflow precisa de um número de versão.",
        "en": "Roboflow requires a version number.",
    },
    "download.kaggle_format": {
        "pt": "O dataset do Kaggle precisa ser 'owner/dataset-slug'.",
        "en": "Kaggle dataset must be 'owner/dataset-slug'.",
    },
    "download.kaggle_old_pair": {
        "pt": (
            "Isto parece o antigo par 'username:key' do Kaggle, que o cliente "
            "atual ignora. Gere um token de API em kaggle.com/settings/api "
            "('Create New Token') e salve esse valor — ele começa com 'KGAT_'."
        ),
        "en": (
            "This looks like the old Kaggle 'username:key' pair, which the "
            "current client ignores. Generate an API token at "
            "kaggle.com/settings/api ('Create New Token') and save that value "
            "instead — it starts with 'KGAT_'."
        ),
    },
    "download.kaggle_no_credentials": {
        "pt": (
            "Credenciais do Kaggle não encontradas. Crie um token de API em "
            "kaggle.com/settings/api e salve-o no campo de credencial, ou "
            "grave-o em ~/.kaggle/access_token: {error}"
        ),
        "en": (
            "Kaggle credentials not found. Create an API token at "
            "kaggle.com/settings/api and save it in the credential field, or "
            "write it to ~/.kaggle/access_token: {error}"
        ),
    },
    "download.hf_no_image_label": {
        "pt": (
            "O dataset '{dataset}' do Hugging Face não tem colunas de imagem e "
            "rótulo para materializar como ImageFolder."
        ),
        "en": (
            "Hugging Face dataset '{dataset}' has no image+label features to "
            "materialize into an ImageFolder."
        ),
    },
    # ── stored provider keys (utils/credentials.py) ──────────────────────────
    "credentials.empty": {
        "pt": "O valor da credencial não pode ficar vazio.",
        "en": "Credential value cannot be empty.",
    },
    # ── the note of /api/model/defaults (model_notes.py, routes.py) ──────────
    "model.collapse_unmeasured": {
        "pt": (
            "{architecture}: Adam a 1e-3 não foi medido para esta família; "
            "sugerimos {optimizer} a {rate:g}, o mesmo das famílias de atenção "
            "medidas."
        ),
        "en": (
            "{architecture}: Adam at 1e-3 was not measured for this family; we "
            "suggest {optimizer} at {rate:g}, the same as the attention families "
            "that were measured."
        ),
    },
    "model.accuracy_on_classes": {
        "pt": "acurácia {accuracy:.2f} em {classes} classes",
        "en": "accuracy {accuracy:.2f} on {classes} classes",
    },
    "model.failure_collapse": {
        "pt": "previu uma classe só",
        "en": "predicted a single class",
    },
    "model.failure_learned_little": {"pt": "aprendeu pouco", "en": "learned little"},
    "model.measured_itself": {
        "pt": "{architecture} {failure} com Adam a 1e-3 ({accuracy}).",
        "en": "{architecture} {failure} with Adam at 1e-3 ({accuracy}).",
    },
    "model.measured_sibling": {
        "pt": (
            "{architecture}: o {measured_on}, da mesma família, {failure} com "
            "Adam a 1e-3 ({accuracy}); este modelo não foi medido."
        ),
        "en": (
            "{architecture}: {measured_on}, from the same family, {failure} with "
            "Adam at 1e-3 ({accuracy}); this model was not measured."
        ),
    },
    "model.remedy_measured": {
        "pt": (
            "Com {optimizer} a {rate:g}, a acurácia foi {recovered:.2f} nas "
            "mesmas condições."
        ),
        "en": (
            "With {optimizer} at {rate:g}, the accuracy was {recovered:.2f} under "
            "the same conditions."
        ),
    },
    "model.remedy_suggested": {
        "pt": "Sugerimos {optimizer} a {rate:g}.",
        "en": "We suggest {optimizer} at {rate:g}.",
    },
    "model.small_images": {
        "pt": (
            "As imagens têm cerca de {median}px de lado; treinar acima disso "
            "amplia a imagem sem acrescentar detalhe."
        ),
        "en": (
            "The images are about {median}px on a side; training above that "
            "enlarges the image without adding detail."
        ),
    },
    # ── config validation and loading (utils/*config.py) ─────────────────────
    # A field name is an identifier and stays as written in both languages.
    "config.path_missing": {
        "pt": "{field} não existe: {path}",
        "en": "{field} does not exist: {path}",
    },
    "config.not_a_file": {
        "pt": "{field} precisa ser um arquivo, recebido: {path}",
        "en": "{field} must be a file, got: {path}",
    },
    "config.not_a_dir": {
        "pt": "{field} precisa ser uma pasta, recebido: {path}",
        "en": "{field} must be a directory, got: {path}",
    },
    "config.custom_or_timm": {
        "pt": "Defina só um entre custom_model e timm_model, não os dois.",
        "en": "Set only one of custom_model or timm_model, not both.",
    },
    "config.batch_power_of_two": {
        "pt": "batch_size precisa ser uma potência de 2, recebido {value}.",
        "en": "batch_size must be a power of 2, got {value}.",
    },
    "config.gpu_ids_negative": {
        "pt": "gpu_ids precisam ser todos >= 0, recebido {ids}",
        "en": "gpu_ids must all be >= 0, got {ids}",
    },
    "config.gpu_ids_unique": {
        "pt": "gpu_ids não podem se repetir, recebido {ids}",
        "en": "gpu_ids must be unique, got {ids}",
    },
    "config.multi_cuda_gpus": {
        "pt": "multi_cuda precisa de pelo menos 2 gpu_ids, recebido {ids}",
        "en": "multi_cuda requires at least 2 gpu_ids, got {ids}",
    },
    "config.schema_newer": {
        "pt": (
            "A config tem schema_version {version}, escrita por uma versão mais "
            "nova do VisionForge (esta suporta até {supported}). Atualize o "
            "VisionForge."
        ),
        "en": (
            "Config schema_version {version} was written by a newer version of "
            "VisionForge (this build supports up to {supported}). Please upgrade "
            "VisionForge."
        ),
    },
    "config.binary_num_classes": {
        "pt": "A tarefa binária exige num_classes=1, recebido {n}.",
        "en": "Binary task requires num_classes=1, got {n}.",
    },
    "config.multiclass_num_classes": {
        "pt": "A tarefa multiclasse exige num_classes>=2, recebido {n}.",
        "en": "Multiclass task requires num_classes>=2, got {n}.",
    },
    "config.auc_multiclass": {
        "pt": (
            "A métrica auc_roc não é definida para tarefas multiclasse. Use "
            "'accuracy' ou 'f1'."
        ),
        "en": (
            "auc_roc metric is not defined for multiclass tasks. Use 'accuracy' "
            "or 'f1' instead."
        ),
    },
    "config.unsafe_path_component": {
        "pt": "'{value}' não é um nome de pasta válido: use um único componente, sem separadores ou caracteres reservados.",
        "en": "'{value}' is not a valid directory name: use a single component without separators or reserved characters.",
    },
    "group.path_outside_root": {
        "pt": "O caminho '{path}' não pertence à raiz de modelos autorizada '{root}'.",
        "en": "Path '{path}' is outside the authorized model root '{root}'.",
    },
    "audit.manifest_outside_root": {
        "pt": "O manifesto de grupos '{path}' está fora da raiz do dataset '{root}'.",
        "en": "Group manifest '{path}' is outside the dataset root '{root}'.",
    },
    "group.linked_metadata": {
        "pt": "O arquivo de metadados '{path}' tem múltiplos hard links e não pode ser alterado como parte de um grupo.",
        "en": "Metadata file '{path}' has multiple hard links and cannot be changed as part of a group.",
    },
    "config.file_not_found": {
        "pt": "Arquivo de config não encontrado: {path}",
        "en": "Config file not found: {path}",
    },
    "config.path_not_file": {
        "pt": "O caminho da config não é um arquivo: {path}",
        "en": "Config path is not a file: {path}",
    },
    "config.yaml_not_mapping": {
        "pt": "O arquivo de config precisa conter um mapeamento YAML, recebido: {kind}",
        "en": "Config file must contain a YAML mapping, got: {kind}",
    },
    "config.detector_invalid": {
        "pt": (
            "O modelo '{name}' não é um detector {backend} válido. Escolha um "
            "destes: {allowed}."
        ),
        "en": (
            "Model '{name}' is not a valid {backend} detector. Choose one of: "
            "{allowed}."
        ),
    },
    "config.detection_needs_data": {
        "pt": "DetectionDataConfig precisa de data_yaml ou de base_dir.",
        "en": "DetectionDataConfig requires either data_yaml or base_dir.",
    },
    "config.target_columns_empty": {
        "pt": "target_columns precisa listar pelo menos uma coluna.",
        "en": "target_columns must list at least one column.",
    },
    "config.target_columns_unique": {
        "pt": "target_columns não podem se repetir, recebido {columns}",
        "en": "target_columns must be unique, got {columns}",
    },
    "config.num_targets_mismatch": {
        "pt": (
            "model.num_targets ({targets}) precisa ser igual ao número de "
            "data.target_columns ({columns})."
        ),
        "en": (
            "model.num_targets ({targets}) must equal the number of "
            "data.target_columns ({columns})."
        ),
    },
    "config.ignore_index_collides": {
        "pt": (
            "data.ignore_index ({index}) coincide com o id de uma classe real "
            "(0..{last}); use um valor fora desse intervalo (por exemplo, 255 "
            "ou -1)."
        ),
        "en": (
            "data.ignore_index ({index}) collides with a real class id "
            "(0..{last}); use a value outside that range (e.g. 255 or -1)."
        ),
    },
    "config.unknown_preprocessing": {
        "pt": "Etapa de pré-processamento desconhecida: {kind}. Disponíveis: {available}.",
        "en": "Unknown preprocessing step: {kind}. Available: {available}.",
    },
    # ── hints of the researcher-task form (GET /api/custom/{key}/schema) ─────
    "schema.early_stopping_patience": {
        "pt": (
            "Épocas seguidas sem melhora antes de encerrar o treino. 0 (padrão) "
            "desliga: roda todas as épocas configuradas."
        ),
        "en": (
            "Consecutive epochs without improvement before training stops. 0 "
            "(default) turns it off: every configured epoch runs."
        ),
    },
    "schema.deterministic": {
        "pt": (
            "Quando True (padrão), força os algoritmos determinísticos do cuDNN e "
            "desliga o auto-tuning, para que a mesma config e a mesma seed "
            "reproduzam os mesmos números. Use False para deixar o cuDNN se "
            "ajustar sozinho, o que pode compensar em muitas épocas, mas torna "
            "impossível reproduzir um run exatamente."
        ),
        "en": (
            "When True (default), forces cuDNN deterministic algorithms and "
            "disables auto-tuning, so the same config and seed reproduce the same "
            "numbers. Set False to let cuDNN auto-tune, which can pay off over "
            "many epochs but makes a run impossible to reproduce exactly."
        ),
    },
    # ── titles, axes and legends of the saved figures (core/plotter.py) ──────
    # Figures are artifacts: they are worded for whoever started the run, and a
    # run with no language bound (the command line, a script) gets English, which
    # is what they always said there. Read through ``tr_artifact``.
    "plot.epoch": {"pt": "Época", "en": "Epoch"},
    "plot.loss": {"pt": "Perda", "en": "Loss"},
    "plot.accuracy": {"pt": "Acurácia", "en": "Accuracy"},
    "plot.train": {"pt": "treino", "en": "train"},
    "plot.val": {"pt": "validação", "en": "val"},
    "plot.title_loss": {
        "pt": "Perda de treino e de validação",
        "en": "Training & Validation Loss",
    },
    "plot.title_accuracy": {
        "pt": "Acurácia de treino e de validação",
        "en": "Training & Validation Accuracy",
    },
    "plot.title_val_map50": {"pt": "mAP@50 de validação", "en": "Validation mAP@50"},
    "plot.title_over_epochs": {"pt": "{label} por época", "en": "{label} over epochs"},
    "plot.perfect": {"pt": "perfeito (y = ŷ)", "en": "perfect (y = ŷ)"},
    "plot.axis_actual": {"pt": "real ({target})", "en": "actual ({target})"},
    "plot.axis_predicted": {"pt": "predito ({target})", "en": "predicted ({target})"},
    "plot.title_predicted_vs_actual": {
        "pt": "Predito vs real",
        "en": "Predicted vs actual",
    },
    "plot.no_error": {"pt": "sem erro", "en": "no error"},
    "plot.mean_bias": {
        "pt": "viés médio = {bias:+.4f}",
        "en": "mean bias = {bias:+.4f}",
    },
    "plot.axis_residual": {
        "pt": "resíduo (predito − real)",
        "en": "residual (predicted − actual)",
    },
    "plot.samples": {"pt": "amostras", "en": "samples"},
    "plot.title_residuals": {
        "pt": "Distribuição dos resíduos",
        "en": "Residual distribution",
    },
    "plot.mean": {"pt": "média = {mean:.4f}", "en": "mean = {mean:.4f}"},
    "plot.title_per_class": {"pt": "{metric} por classe", "en": "{metric} per class"},
    "plot.defect": {"pt": "defeito", "en": "defect"},
    "plot.threshold": {
        "pt": "limiar = {threshold:.4f}",
        "en": "threshold = {threshold:.4f}",
    },
    "plot.axis_anomaly_score": {"pt": "escore de anomalia", "en": "anomaly score"},
    "plot.images": {"pt": "imagens", "en": "images"},
    "plot.title_scores": {
        "pt": "Distribuição dos escores",
        "en": "Score distribution",
    },
    "plot.axis_predicted_class": {"pt": "Predito", "en": "Predicted"},
    "plot.axis_true_class": {"pt": "Real", "en": "True"},
    "plot.title_confusion": {"pt": "Matriz de confusão", "en": "Confusion Matrix"},
    "plot.title_confusion_normalized": {
        "pt": "Matriz de confusão (normalizada pela classe real)",
        "en": "Confusion Matrix (normalized by true class)",
    },
    "plot.class_n": {"pt": "classe {index}", "en": "class {index}"},
    "plot.chance": {"pt": "acaso", "en": "chance"},
    "plot.axis_fpr": {
        "pt": "Taxa de falsos positivos",
        "en": "False Positive Rate",
    },
    "plot.axis_tpr": {
        "pt": "Taxa de verdadeiros positivos",
        "en": "True Positive Rate",
    },
    "plot.title_roc": {"pt": "Curva ROC", "en": "ROC Curve"},
    "plot.axis_recall": {"pt": "Revocação", "en": "Recall"},
    "plot.axis_precision": {"pt": "Precisão", "en": "Precision"},
    "plot.title_precision_recall": {
        "pt": "Curva de precisão e revocação",
        "en": "Precision-Recall Curve",
    },
    # ── LaTeX tables (core/latex_export.py) ──────────────────────────────────
    # Artifacts too, so read through ``tr_artifact``. LaTeX markup that is not
    # language (``\textbf``, ``$n<5$``, ``\%``) stays in the template and its
    # literal braces are doubled, because the template goes through ``format``.
    "latex.experiment": {"pt": "Experimento", "en": "Experiment"},
    "latex.metric": {"pt": "Métrica", "en": "Metric"},
    "latex.mean": {"pt": "Média", "en": "Mean"},
    "latex.sd": {"pt": r"DP ($n-1$)", "en": r"SD ($n-1$)"},
    "latex.ci_t": {"pt": r"IC 95\% (t)", "en": r"95\% CI (t)"},
    "latex.ci_bootstrap": {"pt": r"IC 95\% (bootstrap)", "en": r"95\% CI (bootstrap)"},
    "latex.ci": {"pt": r"IC 95\%", "en": r"95\% CI"},
    "latex.time_s": {"pt": "Tempo (s)", "en": "Time (s)"},
    "latex.comparison": {"pt": "Comparação", "en": "Comparison"},
    "latex.difference": {"pt": "Diferença", "en": "Difference"},
    "latex.test": {"pt": "Teste", "en": "Test"},
    "latex.versus": {"pt": "{a} vs. {b}", "en": "{a} vs {b}"},
    "latex.count_stopped": {"pt": "{n} interrompida(s)", "en": "{n} stopped"},
    "latex.count_failed": {"pt": "{n} com falha", "en": "{n} failed"},
    "latex.count_not_run": {"pt": "{n} não executada(s)", "en": "{n} not run"},
    "latex.replicates_caption": {
        "pt": "{experiment}: réplicas com várias seeds ($n={n}$ seeds).",
        "en": "{experiment}: multi-seed replicates ($n={n}$ seeds).",
    },
    "latex.seeds_requested": {
        "pt": "({finished} de {planned} solicitadas; {detail})",
        "en": "({finished} of {planned} requested; {detail})",
    },
    "latex.replicates_note": {
        "pt": (
            r"$\star$ métrica principal. Seeds: {seeds}. Os intervalos são sobre "
            "a variação entre seeds de uma configuração fixa, não sobre a "
            "distribuição dos dados.{caveat}"
        ),
        "en": (
            r"$\star$ headline metric. Seeds: {seeds}. Intervals are over "
            "seed-to-seed variation of a fixed configuration, not over the data "
            "distribution.{caveat}"
        ),
    },
    "latex.replicates_caution": {
        "pt": (
            r"\textbf{{Atenção:}} com $n<5$ seeds o intervalo $t$ é muito largo e "
            "o bootstrap percentílico, estreito demais; trate ambos como "
            "indicativos e aumente o número de seeds antes de reportar."
        ),
        "en": (
            r"\textbf{{Caution:}} with $n<5$ seeds the $t$ interval is very wide and "
            "the percentile bootstrap too narrow; treat both as indicative and "
            "increase the number of seeds before reporting."
        ),
    },
    "latex.sweep_caption": {
        "pt": (
            "{experiment}: varredura de hiperparâmetros ({mode}), ordenada por "
            "{metric}."
        ),
        "en": "{experiment}: {mode} hyperparameter sweep, ranked by {metric}.",
    },
    "latex.sweep_note": {
        "pt": (
            "{succeeded} de {total} tentativas concluídas com sucesso.{stop_note} "
            "A ordenação com uma única execução por configuração reflete o ruído "
            "de seed e também o hiperparâmetro; confirme a vencedora com réplicas."
        ),
        "en": (
            "{succeeded} of {total} trials succeeded.{stop_note} Ranking over a "
            "single run per configuration reflects seed noise as well as the "
            "hyperparameter; confirm the winner with replicates."
        ),
    },
    "latex.sweep_stopped": {
        "pt": (
            "A varredura foi interrompida após {ran} de {planned} tentativas "
            "planejadas."
        ),
        "en": "The sweep was stopped after {ran} of {planned} planned trials.",
    },
    "latex.sweep_stopped_cut": {
        "pt": (
            "A varredura foi interrompida após {ran} de {planned} tentativas "
            "planejadas ({cut} interrompida(s) antes de terminar)."
        ),
        "en": (
            "The sweep was stopped after {ran} of {planned} planned trials "
            "({cut} stopped before finishing)."
        ),
    },
    "latex.cv_caption": {
        "pt": "{experiment}: validação cruzada com {n_folds} folds.",
        "en": "{experiment}: {n_folds}-fold cross-validation.",
    },
    "latex.cv_caption_partial": {
        "pt": (
            "{experiment}: validação cruzada com {n_folds} folds, {finished} de "
            "{n_folds} folds concluídos ({detail})."
        ),
        "en": (
            "{experiment}: {n_folds}-fold cross-validation, {finished} of "
            "{n_folds} folds finished ({detail})."
        ),
    },
    "latex.cv_mean_sd": {
        "pt": r"\textbf{{Média $\pm$ DP ($n-1$)}}",
        "en": r"\textbf{{Mean $\pm$ SD ($n-1$)}}",
    },
    "latex.comparison_caption": {
        "pt": "{experiment}: comparação pareada nas seeds compartilhadas.",
        "en": "{experiment}: paired comparison over shared seeds.",
    },
    "latex.comparison_note": {
        "pt": (
            r"$^{{*}}$ significativo a $\alpha={alpha}$ após a correção de "
            "Holm-Bonferroni na família de comparações. As diferenças são A menos "
            "B nas seeds em que as duas configurações foram executadas.{power_note}"
        ),
        "en": (
            r"$^{{*}}$ significant at $\alpha={alpha}$ after Holm-Bonferroni "
            "correction across the family. Differences are A minus B on the "
            "seeds both configurations ran.{power_note}"
        ),
    },
    "latex.comparison_underpowered": {
        "pt": (
            r"\textbf{{Poder insuficiente:}} com tão poucas seeds o teste de "
            r"postos não consegue retornar $p<{floor:.4f}$, então nenhuma "
            r"diferença poderia atingir $\alpha={alpha}$, qualquer que fosse o "
            "seu tamanho. Adicione seeds antes de interpretar um resultado não "
            "significativo."
        ),
        "en": (
            r"\textbf{{Underpowered:}} with this many seeds the rank test cannot "
            r"return $p<{floor:.4f}$, so no difference could reach "
            r"$\alpha={alpha}$ regardless of its size. Add seeds before "
            "interpreting a non-significant result."
        ),
    },
    # ── Markdown model card (gui/api/routes.py) ──────────────────────────────
    # The card is an artifact, but the route that serves it always runs under
    # the page language, so a Portuguese page gets a Portuguese card.
    "card.run_id": {"pt": "ID da execução", "en": "Run ID"},
    "card.timestamp": {"pt": "Data e hora", "en": "Timestamp"},
    "card.status": {"pt": "Situação", "en": "Status"},
    "card.status_completed": {"pt": "concluída", "en": "completed"},
    "card.status_failed": {"pt": "falhou", "en": "failed"},
    "card.status_running": {"pt": "em andamento", "en": "running"},
    "card.status_stopped": {"pt": "interrompida", "en": "stopped"},
    "card.stopped": {"pt": "Interrompida", "en": "Stopped"},
    "card.yes": {"pt": "sim", "en": "yes"},
    "card.device_used": {"pt": "Dispositivo usado", "en": "Device used"},
    "card.run_dir": {"pt": "Diretório da execução", "en": "Run directory"},
    "card.checkpoint": {"pt": "Arquivo do checkpoint", "en": "Checkpoint"},
    "card.configuration": {"pt": "Configuração", "en": "Configuration"},
    "card.architecture": {"pt": "Arquitetura", "en": "Architecture"},
    "card.pretrained": {"pt": "Pré-treinado", "en": "Pretrained"},
    "card.task": {"pt": "Tarefa", "en": "Task"},
    "card.optimizer": {"pt": "Otimizador", "en": "Optimizer"},
    "card.learning_rate": {"pt": "Taxa de aprendizado", "en": "Learning rate"},
    "card.batch_size": {"pt": "Tamanho do batch", "en": "Batch size"},
    "card.epochs_max": {"pt": "Épocas (máx.)", "en": "Epochs (max)"},
    "card.mixed_precision": {"pt": "Precisão mista", "en": "Mixed precision"},
    "card.enabled": {"pt": "ativada", "en": "enabled"},
    "card.preprocessing": {
        "pt": "Pipeline de pré-processamento",
        "en": "Preprocessing pipeline",
    },
    "card.preprocessing_intro": {
        "pt": (
            "Aplicadas a cada imagem carregada, antes do data augmentation, nesta "
            "ordem:"
        ),
        "en": "Applied to every loaded image before augmentation, in this order:",
    },
    "card.augmentation": {
        "pt": "Data augmentation / Normalização",
        "en": "Augmentation / Normalization",
    },
    "card.field": {"pt": "Campo", "en": "Field"},
    "card.value": {"pt": "Valor", "en": "Value"},
    "card.final_metrics": {"pt": "Métricas finais", "en": "Final metrics"},
    "card.metric": {"pt": "Métrica", "en": "Metric"},
    "card.ci_bootstrap": {"pt": "IC 95% (bootstrap)", "en": "95% CI (bootstrap)"},
    "card.bootstrap_note": {
        "pt": (
            "Os intervalos são bootstrap percentílico sobre as {n_samples} imagens "
            "de teste ({n_resamples} reamostragens): quantificam o ruído amostral "
            "do conjunto de teste para este modelo fixo, não a variância entre "
            "execuções. Para esta última, treine réplicas com várias seeds."
        ),
        "en": (
            "Intervals are percentile bootstrap over the {n_samples} test images "
            "({n_resamples} resamples): they quantify test-split sampling noise "
            "for this fixed model, not run-to-run variance. For the latter, train "
            "replicates under several seeds."
        ),
    },
    "card.none_recorded": {"pt": "_(nenhuma registrada)_", "en": "_(none recorded)_"},
    "card.history": {
        "pt": "Histórico de treino ({n} época(s))",
        "en": "Training history ({n} epoch(s))",
    },
    "card.epoch": {"pt": "Época", "en": "Epoch"},
    "card.plots": {"pt": "Gráficos", "en": "Plots"},
    "card.tests": {
        "pt": "Testes em conjuntos de dados adicionais ({n})",
        "en": "Test runs on additional datasets ({n})",
    },
    "card.test_base_dir": {"pt": "diretório-base", "en": "base_dir"},
    "card.test_timestamp": {"pt": "data e hora", "en": "timestamp"},
    "card.stopped_folds": {
        "pt": "{ok} de {n} folds concluídos",
        "en": "{ok} of {n} folds finished",
    },
    "card.stopped_bank_built": {
        "pt": "banco de memória construído, pontuação não realizada",
        "en": "memory bank built, scoring skipped",
    },
    "card.stopped_before_bank": {
        "pt": "antes de o banco de memória ser construído",
        "en": "before the memory bank was built",
    },
    "card.stopped_epochs": {
        "pt": "{done} de {planned} épocas",
        "en": "{done} of {planned} epochs",
    },
}


def message(lang: Lang, key: str, /, **params: object) -> str:
    """The catalog text ``key`` in ``lang`` with ``params`` filled in."""
    return CATALOG[key][lang].format(**params)


def tr(key: str, /, **params: object) -> str:
    """The catalog text ``key`` in the language of the current request."""
    return message(current_lang(), key, **params)


def tr_artifact(key: str, /, **params: object) -> str:
    """The catalog text ``key`` for a file a run saves: English unless a language is bound."""
    return message(bound_lang() or ARTIFACT_DEFAULT_LANG, key, **params)


__all__ = [
    "ARTIFACT_DEFAULT_LANG",
    "CATALOG",
    "DEFAULT_LANG",
    "LANG_ENV",
    "LANG_HEADER",
    "Lang",
    "bound_lang",
    "current_lang",
    "lang_from_env",
    "message",
    "parse_lang",
    "resolve_lang",
    "set_lang",
    "tr",
    "tr_artifact",
    "using_lang",
]
