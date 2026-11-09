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
# Texts the Hub, Zenodo and GitHub exporters repeat (ADR-171, ADR-174, ADR-175).
_TOKEN_INVALID_PT = "Token inválido: cole o token inteiro, sem espaços."
_TOKEN_INVALID_EN = "Invalid token: paste the whole token, without spaces."
_RUN_NOT_FOUND_PT = "Execução não encontrada."
_RUN_NOT_FOUND_EN = "Run not found."
_PREVIEW_CHANGED_PT = (
    "Os arquivos mudaram depois da prévia. Abra a prévia de novo e confira."
)
_PREVIEW_CHANGED_EN = (
    "The files changed after the preview. Open the preview again and review it."
)
_RECORD_MISSING_PT = "O registro desta execução não foi encontrado."
_RECORD_MISSING_EN = "The record of this run was not found."
_UPLOAD_REFUSED_PT = "Esta execução não pode ser enviada: {reason}"
_UPLOAD_REFUSED_EN = "This run cannot be uploaded: {reason}"
# Texts the derived-dataset tools (incorporate, merge, re-split, COCO) repeat.
_PREVIEW_NOT_READY_PT = "A prévia ainda não terminou ou não deu certo."
_PREVIEW_NOT_READY_EN = "The preview has not finished or did not succeed."
_BAD_DEST_NAME_PT = "Nome de pasta inválido para o dataset novo."
_BAD_DEST_NAME_EN = "Invalid folder name for the new dataset."
_BASE_MISSING_PT = "A pasta do dataset não existe: {path}"
_BASE_MISSING_EN = "The dataset folder does not exist: {path}"
_DEST_EXISTS_PT = (
    "Já existe uma pasta chamada '{name}' ao lado do dataset. Escolha outro nome."
)
_DEST_EXISTS_EN = (
    "A folder named '{name}' already exists next to the dataset. Pick another name."
)

CATALOG: dict[str, dict[Lang, str]] = {
    # ── class vocabulary (core/data.py) ─────────────────────────────────────
    "data.empty_split": {
        "pt": "O split {split} ({root}) não tem imagens em nenhuma pasta de classe.",
        "en": "The {split} split ({root}) has no images in any class folder.",
    },
    "data.unknown_classes": {
        "pt": (
            "O split {split} tem pasta(s) de classe {unknown} ausentes no split "
            "de treino (classes de treino: {classes}). Mova ou renomeie essas pastas."
        ),
        "en": (
            "The {split} split has class folder(s) {unknown} that the "
            "training split does not have (training classes: "
            "{classes}). Move or rename them."
        ),
    },
    "data.class_count_mismatch": {
        "pt": (
            "O split de treino tem {found} pasta(s) de classe {classes}, mas "
            "a tarefa '{task}' com model.num_classes={num_classes} "
            "exigem {expected}."
        ),
        "en": (
            "Training split has {found} class folder(s) {classes}, but "
            "task '{task}' with model.num_classes={num_classes} "
            "require {expected}."
        ),
    },
    # ── validation selection (core/sweep.py, task runners) ──────────────────
    "sweep.no_metrics": {"pt": "nenhuma", "en": "none"},
    "sweep.missing_validation_value": {
        "pt": (
            "Nenhum valor de {split} para '{metric}' "
            "(métricas de validação: {metrics}); um sweep nunca "
            "classifica tentativas por métricas de teste."
        ),
        "en": (
            "No {split} value for '{metric}' "
            "(validation metrics: {metrics}); a sweep never "
            "ranks trials on test metrics."
        ),
    },
    "sweep.nonfinite_validation_value": {
        "pt": (
            "O valor de {split} '{metric}' não é finito ({value}); "
            "tentativa fora do ranking."
        ),
        "en": "The {split} '{metric}' is not finite ({value}); not ranked.",
    },
    "sweep.anomaly_selection_unavailable": {
        "pt": (
            "A detecção de anomalias não tem split de validação rotulado para "
            "selecionar hiperparâmetros; AUROC/F1 vêm do split de teste, por isso "
            "um sweep é recusado. Compare configurações em execuções separadas "
            "ou réplicas."
        ),
        "en": (
            "Anomaly detection has no labelled validation split to select "
            "hyperparameters on; its AUROC/F1 come from the test split, so a sweep "
            "is refused. Compare settings with separate runs or replicates instead."
        ),
    },
    "sweep.level2_selection_unavailable": {
        "pt": (
            "A tarefa '{task}' executa seu próprio loop (Level 2), então o "
            "VisionForge não consegue identificar quais métricas vêm de um "
            "split de validação; um sweep exige métricas de validação por época "
            "(hooks Level 1)."
        ),
        "en": (
            "Task '{task}' runs its own loop (Level 2), so VisionForge "
            "cannot tell which metrics come from a validation split; a sweep "
            "needs per-epoch validation metrics (Level 1 hooks)."
        ),
    },
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
    "testrun.not_class_root": {
        "pt": (
            "{path} não tem subpastas: parece a pasta de uma classe só. Escolha "
            "a pasta que contém uma subpasta por classe (ex.: a pasta test/)."
        ),
        "en": (
            "{path} has no sub-folders: it looks like a single class folder. "
            "Choose the folder that contains one sub-folder per class (e.g. test/)."
        ),
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
    "hub.request_invalid": {
        "pt": "Pedido inválido para o Hugging Face Hub.",
        "en": "Invalid Hugging Face Hub request.",
    },
    "hub.token_invalid": {
        "pt": _TOKEN_INVALID_PT,
        "en": _TOKEN_INVALID_EN,
    },
    "hub.token_missing": {
        "pt": "Salve um token do Hugging Face neste perfil antes de enviar.",
        "en": "Save a Hugging Face token in this profile before uploading.",
    },
    "hub.run_not_found": {
        "pt": _RUN_NOT_FOUND_PT,
        "en": _RUN_NOT_FOUND_EN,
    },
    "hub.artifact_missing": {
        "pt": "O checkpoint ou o registro desta execução não foi encontrado.",
        "en": "The checkpoint or the record of this run was not found.",
    },
    "hub.repo_invalid": {
        "pt": "Informe o destino no formato usuário/modelo.",
        "en": "Enter the destination as user/model.",
    },
    "hub.preview_changed": {
        "pt": _PREVIEW_CHANGED_PT,
        "en": _PREVIEW_CHANGED_EN,
    },
    "hub.private_required": {
        "pt": "O repositório precisa ser privado. Use um destino novo ou torne-o privado no Hub.",
        "en": "The repository must be private. Use a new destination or make it private on the Hub.",
    },
    "hub.package_invalid": {
        "pt": "Esta execução não pode ser exportada: {reason}",
        "en": "This run cannot be exported: {reason}",
    },
    "hub.export_failed": {
        "pt": "O envio ao Hugging Face Hub falhou. Confira o token, a conexão e o nome do repositório.",
        "en": "The Hugging Face Hub upload failed. Check the token, the connection and the repository name.",
    },
    "github.request_invalid": {
        "pt": "Pedido inválido para o GitHub.",
        "en": "Invalid GitHub request.",
    },
    "github.token_invalid": {
        "pt": _TOKEN_INVALID_PT,
        "en": _TOKEN_INVALID_EN,
    },
    "github.token_missing": {
        "pt": "Salve um token do GitHub neste perfil antes de enviar.",
        "en": "Save a GitHub token in this profile before uploading.",
    },
    "github.token_rejected": {
        "pt": "O GitHub recusou o token. Confira se ele foi copiado inteiro e se não expirou.",
        "en": "GitHub refused the token. Check that it was copied whole and has not expired.",
    },
    "github.token_forbidden": {
        "pt": "O token não pode escrever neste repositório. Dê a ele a permissão Contents: leitura e escrita para este repositório.",
        "en": "The token cannot write to this repository. Give it the Contents: read and write permission for this repository.",
    },
    "github.repo_not_found": {
        "pt": "Repositório não encontrado, ou o token não tem acesso a ele. Crie o repositório privado no GitHub antes.",
        "en": "Repository not found, or the token has no access to it. Create the private repository on GitHub first.",
    },
    "github.repo_invalid": {
        "pt": "Informe o repositório no formato usuário/repositório.",
        "en": "Enter the repository as user/repository.",
    },
    "github.private_required": {
        "pt": "O repositório precisa ser privado. Torne-o privado no GitHub ou use outro.",
        "en": "The repository must be private. Make it private on GitHub or use another one.",
    },
    "github.run_not_found": {
        "pt": _RUN_NOT_FOUND_PT,
        "en": _RUN_NOT_FOUND_EN,
    },
    "github.artifact_missing": {
        "pt": _RECORD_MISSING_PT,
        "en": _RECORD_MISSING_EN,
    },
    "github.package_invalid": {
        "pt": _UPLOAD_REFUSED_PT,
        "en": _UPLOAD_REFUSED_EN,
    },
    "github.preview_changed": {
        "pt": _PREVIEW_CHANGED_PT,
        "en": _PREVIEW_CHANGED_EN,
    },
    "github.export_failed": {
        "pt": "O envio ao GitHub falhou. Confira a conexão e tente de novo.",
        "en": "The GitHub upload failed. Check the connection and try again.",
    },
    "dvc.request_invalid": {
        "pt": "Pedido inválido para o DVC.",
        "en": "Invalid DVC request.",
    },
    "dvc.remote_client": {
        "pt": "O DVC só funciona no computador onde o VisionForge roda: ele grava numa pasta e roda git e DVC nesta máquina.",
        "en": "DVC only works on the computer running VisionForge: it writes to a folder and runs git and DVC on this machine.",
    },
    "dvc.foreign_origin": {
        "pt": "Pedido recusado: ele veio de outra página.",
        "en": "Request refused: it came from another page.",
    },
    "dvc.repo_invalid": {
        "pt": "Informe o caminho completo da pasta do repositório.",
        "en": "Enter the full path of the repository folder.",
    },
    "dvc.repo_not_ready": {
        "pt": "A pasta não é um repositório git com DVC. Use Preparar pasta antes.",
        "en": "The folder is not a git repository with DVC. Use Prepare folder first.",
    },
    "dvc.prepare_refused": {
        "pt": "Não dá para preparar este caminho: ele é um arquivo, ou a pasta de cima não existe.",
        "en": "This path cannot be prepared: it is a file, or its parent folder does not exist.",
    },
    "dvc.git_missing": {
        "pt": "O git não foi encontrado neste computador. Instale o Git e reabra o VisionForge.",
        "en": "git was not found on this computer. Install Git and reopen VisionForge.",
    },
    "dvc.dvc_missing": {
        "pt": "O DVC não foi encontrado. Instale o extra visionforge-studio[dvc], ou o DVC no PATH, e reabra o VisionForge.",
        "en": "DVC was not found. Install the visionforge-studio[dvc] extra, or DVC on the PATH, and reopen VisionForge.",
    },
    "dvc.git_failed": {
        "pt": "O git falhou em {step}. Confira se o git tem nome e e-mail configurados e se um hook do repositório não bloqueou o commit. O log do servidor tem a saída.",
        "en": "git failed at {step}. Check that git has a name and e-mail configured and that no repository hook blocked the commit. The server log has the output.",
    },
    "dvc.dvc_failed": {
        "pt": "O DVC falhou em {step}. O log do servidor tem a saída.",
        "en": "DVC failed at {step}. The server log has the output.",
    },
    "dvc.run_not_found": {
        "pt": _RUN_NOT_FOUND_PT,
        "en": _RUN_NOT_FOUND_EN,
    },
    "dvc.artifact_missing": {
        "pt": _RECORD_MISSING_PT,
        "en": _RECORD_MISSING_EN,
    },
    "dvc.package_invalid": {
        "pt": _UPLOAD_REFUSED_PT,
        "en": _UPLOAD_REFUSED_EN,
    },
    "dvc.preview_changed": {
        "pt": _PREVIEW_CHANGED_PT,
        "en": _PREVIEW_CHANGED_EN,
    },
    "zenodo.request_invalid": {
        "pt": "Pedido inválido para o Zenodo.",
        "en": "Invalid Zenodo request.",
    },
    "zenodo.token_invalid": {
        "pt": _TOKEN_INVALID_PT,
        "en": _TOKEN_INVALID_EN,
    },
    "zenodo.token_missing": {
        "pt": "Salve um token do Zenodo neste perfil antes de enviar.",
        "en": "Save a Zenodo token in this profile before uploading.",
    },
    "zenodo.token_rejected": {
        "pt": "O Zenodo recusou o token. Confira se ele é deste site (Zenodo ou Sandbox) e se tem o escopo deposit:write.",
        "en": "Zenodo refused the token. Check that it belongs to this site (Zenodo or Sandbox) and has the deposit:write scope.",
    },
    "zenodo.run_not_found": {
        "pt": _RUN_NOT_FOUND_PT,
        "en": _RUN_NOT_FOUND_EN,
    },
    "zenodo.artifact_missing": {
        "pt": _RECORD_MISSING_PT,
        "en": _RECORD_MISSING_EN,
    },
    "zenodo.package_invalid": {
        "pt": _UPLOAD_REFUSED_PT,
        "en": _UPLOAD_REFUSED_EN,
    },
    "zenodo.preview_changed": {
        "pt": _PREVIEW_CHANGED_PT,
        "en": _PREVIEW_CHANGED_EN,
    },
    "zenodo.export_failed": {
        "pt": "O envio ao Zenodo falhou. Confira a conexão e tente de novo.",
        "en": "The Zenodo upload failed. Check the connection and try again.",
    },
    "wandb.request_invalid": {
        "pt": "Pedido inválido para o Weights & Biases.",
        "en": "Invalid Weights & Biases request.",
    },
    "wandb.token_invalid": {
        "pt": "Chave inválida: cole a chave de API inteira, sem espaços.",
        "en": "Invalid key: paste the whole API key, without spaces.",
    },
    "wandb.token_missing": {
        "pt": "Salve uma chave de API do W&B neste perfil antes de enviar.",
        "en": "Save a W&B API key in this profile before uploading.",
    },
    "wandb.name_invalid": {
        "pt": "Use só letras, números, ponto, hífen e sublinhado (até 128).",
        "en": "Use only letters, digits, dot, hyphen and underscore (up to 128).",
    },
    "wandb.retry_unavailable": {
        "pt": "O envio ao W&B está indisponível para esta execução ou perfil.",
        "en": "W&B export is unavailable for this run or profile.",
    },
    "code.tour_missing": {
        "pt": "Ainda não há código comentado para esta tarefa.",
        "en": "There is no annotated code for this task yet.",
    },
    "code.data.resize": {
        "pt": "Toda imagem fica do mesmo tamanho (image_size × image_size): a rede recebe um tensor de forma fixa. O Resize ajusta o lado menor e o CenterCrop corta o centro.",
        "en": "Every image gets the same size (image_size × image_size): the network takes a tensor of fixed shape. Resize scales the shorter side and CenterCrop cuts the centre.",
    },
    "code.data.augment": {
        "pt": "Aumento de dados, só no treino: a cada época a mesma foto chega um pouco diferente (espelhada, girada, com outra cor). O modelo aprende a forma, não a foto exata.",
        "en": "Data augmentation, training only: each epoch the same photo arrives slightly different (flipped, rotated, recoloured). The model learns the shape, not the exact photo.",
    },
    "code.data.tensor": {
        "pt": "ToTensor transforma a imagem em números de 0 a 1, no formato [canais, altura, largura]. Normalize centraliza cada canal com a média e o desvio que a rede viu no pré-treino.",
        "en": "ToTensor turns the image into numbers from 0 to 1, shaped [channels, height, width]. Normalize centres each channel with the mean and deviation the network saw in pre-training.",
    },
    "code.model.family": {
        "pt": "Ramo da família {family}: a arquitetura termina numa camada linear com uma saída por classe do ImageNet (1000). Ela é trocada por uma nova, com uma saída por classe do seu dataset (uma só no binário). O resto da rede continua igual; com pesos pré-treinados, ele já sabe ver bordas e texturas.",
        "en": "The {family} family branch: the architecture ends in a linear layer with one output per ImageNet class (1000). It is replaced by a new one with one output per class of your dataset (a single one for binary). The rest of the network stays the same; with pre-trained weights it already sees edges and textures.",
    },
    "code.loss.binary": {
        "pt": "Binário: uma saída só e a BCEWithLogitsLoss, que aplica a sigmoide e mede o erro contra o rótulo 0 ou 1.",
        "en": "Binary: a single output and BCEWithLogitsLoss, which applies the sigmoid and measures the error against the 0 or 1 label.",
    },
    "code.loss.multiclass": {
        "pt": "Multiclasse: uma saída por classe e a CrossEntropyLoss, que aplica o softmax e pune pouca probabilidade na classe certa.",
        "en": "Multiclass: one output per class and CrossEntropyLoss, which applies the softmax and penalises low probability on the right class.",
    },
    "code.train.mode": {
        "pt": "Modo de treino: liga o dropout e faz o BatchNorm atualizar suas médias com cada lote.",
        "en": "Training mode: turns dropout on and lets BatchNorm update its running averages with each batch.",
    },
    "code.train.batch": {
        "pt": "Um lote por vez: as imagens e os rótulos vão para o dispositivo (GPU ou CPU). Uma época é uma volta por todos os lotes do treino.",
        "en": "One batch at a time: the images and labels move to the device (GPU or CPU). An epoch is one pass over every training batch.",
    },
    "code.train.binary_target": {
        "pt": "No binário, a loss quer o rótulo como número real e com a mesma forma da saída, [lote, 1].",
        "en": "For binary, the loss wants the label as a real number with the output's shape, [batch, 1].",
    },
    "code.train.zero_grad": {
        "pt": "Zera os gradientes do lote anterior. Sem isso, eles se somariam de um lote para o outro.",
        "en": "Clears the previous batch's gradients. Without it they would add up from one batch to the next.",
    },
    "code.train.amp": {
        "pt": "Precisão mista: as contas rodam em float16 onde é seguro, o que é mais rápido na GPU. O GradScaler aumenta a loss antes do backward para gradientes pequenos não virarem zero.",
        "en": "Mixed precision: the maths runs in float16 where it is safe, which is faster on a GPU. GradScaler scales the loss up before backward so small gradients do not round to zero.",
    },
    "code.train.step": {
        "pt": "O passo de aprendizado: forward (a rede prevê), loss (o erro em relação ao rótulo), backward (o gradiente de cada peso) e step (o otimizador mexe nos pesos na direção que diminui o erro).",
        "en": "The learning step: forward (the network predicts), loss (the error against the label), backward (each weight's gradient) and step (the optimiser moves the weights in the direction that lowers the error).",
    },
    "code.train.count": {
        "pt": "Conta acertos sem guardar gradientes: serve só para mostrar a acurácia de treino, não muda os pesos.",
        "en": "Counts hits without keeping gradients: it only shows the training accuracy and does not change the weights.",
    },
    "code.predict.binary": {
        "pt": "Binário: a sigmoide transforma a saída em probabilidade da classe positiva; acima de 0,5 a previsão é positiva.",
        "en": "Binary: the sigmoid turns the output into the positive class probability; above 0.5 the prediction is positive.",
    },
    "code.predict.multiclass": {
        "pt": "Multiclasse: a classe prevista é a de maior saída (argmax).",
        "en": "Multiclass: the predicted class is the one with the largest output (argmax).",
    },
    "code.validate.mode": {
        "pt": "Modo de avaliação: desliga o dropout, e o BatchNorm usa as médias aprendidas. Nada nesta função mexe nos pesos.",
        "en": "Evaluation mode: dropout is off and BatchNorm uses the averages it learned. Nothing in this function changes the weights.",
    },
    "code.validate.no_grad": {
        "pt": "Sem gradientes: a validação só mede. Gasta menos memória e roda mais rápido.",
        "en": "No gradients: validation only measures. It uses less memory and runs faster.",
    },
    "code.validate.loss": {
        "pt": "Esta é a val_loss. A época com a menor val_loss é a que fica salva em best_model.pth.",
        "en": "This is the val_loss. The epoch with the lowest val_loss is the one saved in best_model.pth.",
    },
    "code.test.once": {
        "pt": "O teste roda uma vez, no fim, com os pesos do best_model.pth. São imagens que não influenciaram nenhuma escolha do treino.",
        "en": "The test runs once, at the end, with the weights of best_model.pth. These images influenced no choice made during training.",
    },
    "code.test.metrics": {
        "pt": "As métricas comparam o previsto com o verdadeiro. Na multiclasse a média é macro: cada classe pesa igual, mesmo as que têm poucas imagens.",
        "en": "The metrics compare predicted and true labels. For multiclass the average is macro: every class weighs the same, even those with few images.",
    },
    "mlflow.retry_unavailable": {
        "pt": "A exportação para MLflow está indisponível para esta execução ou perfil.",
        "en": "MLflow export is unavailable for this run or profile.",
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
    # ── Host and Origin guard (gui/security.py) ──────────────────────────────
    "guard.foreign_host": {
        "pt": (
            "Pedido recusado: o endereço usado não é um dos nomes deste servidor. "
            "Abra o VisionForge por localhost ou, num servidor de laboratório, "
            "inicie-o com --allowed-host <nome>."
        ),
        "en": (
            "Request refused: the address used is not one of this server's names. "
            "Open VisionForge through localhost or, on a lab server, start it "
            "with --allowed-host <name>."
        ),
    },
    "guard.foreign_origin": {
        "pt": "Pedido recusado: ele veio de uma página que não é do VisionForge.",
        "en": "Request refused: it came from a page that is not VisionForge's.",
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
    # ── dataset explorer (gui/api/explorer_routes.py, ADR-165) ───────────────
    "explorer.scan_busy": {
        "pt": "Já existe uma varredura em andamento. Espere ou interrompa a atual.",
        "en": "A scan is already running. Wait for it or cancel it.",
    },
    "explorer.scan_missing": {
        "pt": "Varredura não encontrada.",
        "en": "Scan not found.",
    },
    "explorer.unreadable_image": {
        "pt": "Não foi possível abrir esta imagem.",
        "en": "This image could not be opened.",
    },
    # ── incorporating images (gui/api/incorporate_*.py, ADR-166) ─────────────
    "incorporate.busy": {
        "pt": "Já existe uma incorporação em andamento. Espere ou interrompa a atual.",
        "en": "An incorporation is already running. Wait for it or cancel it.",
    },
    "incorporate.job_missing": {
        "pt": "Incorporação não encontrada. Refaça a prévia.",
        "en": "Incorporation not found. Run the preview again.",
    },
    "incorporate.preview_not_ready": {
        "pt": _PREVIEW_NOT_READY_PT,
        "en": _PREVIEW_NOT_READY_EN,
    },
    "incorporate.base_not_opened": {
        "pt": "Abra o dataset no explorador antes de incorporar imagens a ele.",
        "en": "Open the dataset in the explorer before adding images to it.",
    },
    "incorporate.base_missing": {
        "pt": _BASE_MISSING_PT,
        "en": _BASE_MISSING_EN,
    },
    "incorporate.source_missing": {
        "pt": "A pasta das imagens novas não existe: {path}",
        "en": "The folder of new images does not exist: {path}",
    },
    "incorporate.source_overlaps_base": {
        "pt": "A pasta das imagens novas não pode ficar dentro do dataset nem conter o dataset.",
        "en": "The folder of new images cannot be inside the dataset or contain it.",
    },
    "incorporate.base_empty": {
        "pt": "O dataset não tem imagens, então não há proporção a seguir.",
        "en": "The dataset has no images, so there is no proportion to follow.",
    },
    "incorporate.bad_ratios": {
        "pt": "Proporção inválida: use números não negativos com soma maior que zero.",
        "en": "Invalid ratios: use non-negative numbers that sum to more than zero.",
    },
    "incorporate.ratio_split_absent": {
        "pt": "O dataset não tem a divisão '{split}'; ela não será criada.",
        "en": "The dataset has no '{split}' split; it will not be created.",
    },
    "incorporate.not_a_new_class": {
        "pt": "'{name}' não é uma classe nova desta prévia.",
        "en": "'{name}' is not a new class of this preview.",
    },
    "incorporate.bad_dest_name": {
        "pt": _BAD_DEST_NAME_PT,
        "en": _BAD_DEST_NAME_EN,
    },
    "incorporate.dest_exists": {
        "pt": _DEST_EXISTS_PT,
        "en": _DEST_EXISTS_EN,
    },
    "incorporate.base_changed": {
        "pt": "O dataset mudou desde a prévia. Refaça a prévia.",
        "en": "The dataset changed since the preview. Run the preview again.",
    },
    "incorporate.source_changed": {
        "pt": "A imagem '{path}' mudou desde a prévia. Refaça a prévia.",
        "en": "The image '{path}' changed since the preview. Run the preview again.",
    },
    # ── merging two datasets (gui/api/merge_*.py, ADR-167) ─────────────────
    "merge.busy": {
        "pt": "Já existe uma junção em andamento. Espere ou interrompa a atual.",
        "en": "A merge is already running. Wait for it or cancel it.",
    },
    "merge.job_missing": {
        "pt": "Junção não encontrada. Refaça a prévia.",
        "en": "Merge not found. Run the preview again.",
    },
    "merge.preview_not_ready": {
        "pt": _PREVIEW_NOT_READY_PT,
        "en": _PREVIEW_NOT_READY_EN,
    },
    "merge.base_not_opened": {
        "pt": "Abra o dataset no explorador antes de juntá-lo a outro.",
        "en": "Open the dataset in the explorer before merging it with another.",
    },
    "merge.a_missing": {
        "pt": "A pasta do dataset aberto não existe: {path}",
        "en": "The folder of the opened dataset does not exist: {path}",
    },
    "merge.b_missing": {
        "pt": "A pasta do outro dataset não existe: {path}",
        "en": "The folder of the other dataset does not exist: {path}",
    },
    "merge.datasets_overlap": {
        "pt": "Os dois datasets precisam ser pastas diferentes, e um não pode ficar dentro do outro.",
        "en": "The two datasets must be different folders, and neither can be inside the other.",
    },
    "merge.empty": {
        "pt": "O dataset {side} não tem imagens.",
        "en": "Dataset {side} has no images.",
    },
    "merge.unreadable": {
        "pt": "Não foi possível ler a imagem '{path}'.",
        "en": "The image '{path}' could not be read.",
    },
    "merge.bad_mapping": {
        "pt": "Mapeamento inválido: '{name}' não é um dos dois datasets.",
        "en": "Invalid mapping: '{name}' is not one of the two datasets.",
    },
    "merge.unknown_class": {
        "pt": "A classe '{name}' não existe no dataset {side}.",
        "en": "The class '{name}' does not exist in dataset {side}.",
    },
    "merge.empty_target": {
        "pt": "A classe '{name}' precisa de um nome de destino, ou ser descartada.",
        "en": "The class '{name}' needs a target name, or to be discarded.",
    },
    "merge.bad_target": {
        "pt": "'{name}' não serve como nome de classe (pasta inválida).",
        "en": "'{name}' is not usable as a class name (invalid folder).",
    },
    "merge.targets_differ_by_case": {
        "pt": "'{first}' e '{second}' só diferem por maiúsculas e minúsculas; no Windows seriam a mesma pasta. Use o mesmo nome nas duas classes para juntá-las.",
        "en": "'{first}' and '{second}' differ only by letter case; on Windows they would be the same folder. Give both classes the same name to merge them.",
    },
    "merge.nothing_left": {
        "pt": "Com este mapeamento não sobra nenhuma imagem.",
        "en": "With this mapping no image is left.",
    },
    "merge.bad_dest_name": {
        "pt": _BAD_DEST_NAME_PT,
        "en": _BAD_DEST_NAME_EN,
    },
    "merge.dest_exists": {
        "pt": _DEST_EXISTS_PT,
        "en": _DEST_EXISTS_EN,
    },
    "merge.a_changed": {
        "pt": "O dataset aberto mudou desde a prévia. Refaça a prévia.",
        "en": "The opened dataset changed since the preview. Run the preview again.",
    },
    "merge.b_changed": {
        "pt": "O outro dataset mudou desde a prévia. Refaça a prévia.",
        "en": "The other dataset changed since the preview. Run the preview again.",
    },
    "merge.image_changed": {
        "pt": "A imagem '{path}' mudou desde a prévia. Refaça a prévia.",
        "en": "The image '{path}' changed since the preview. Run the preview again.",
    },
    # ── re-splitting a dataset (gui/api/resplit_*.py, ADR-168) ───────────────
    "resplit.busy": {
        "pt": "Já existe uma nova divisão em andamento. Espere ou interrompa a atual.",
        "en": "A re-split is already running. Wait for it or cancel it.",
    },
    "resplit.job_missing": {
        "pt": "Divisão não encontrada. Refaça a prévia.",
        "en": "Re-split not found. Run the preview again.",
    },
    "resplit.preview_not_ready": {
        "pt": _PREVIEW_NOT_READY_PT,
        "en": _PREVIEW_NOT_READY_EN,
    },
    "resplit.base_not_opened": {
        "pt": "Abra o dataset no explorador antes de refazer a divisão dele.",
        "en": "Open the dataset in the explorer before re-splitting it.",
    },
    "resplit.base_missing": {
        "pt": _BASE_MISSING_PT,
        "en": _BASE_MISSING_EN,
    },
    "resplit.base_empty": {
        "pt": "O dataset não tem imagens para dividir.",
        "en": "The dataset has no images to split.",
    },
    "resplit.unreadable": {
        "pt": "Não foi possível ler a imagem '{path}'.",
        "en": "The image '{path}' could not be read.",
    },
    "resplit.group_csv_missing": {
        "pt": "Não foi possível ler o arquivo de grupos: {path}",
        "en": "The group file could not be read: {path}",
    },
    "resplit.bad_group_csv": {
        "pt": "Arquivo de grupos inválido ({detail}). Use um CSV UTF-8 com as colunas path,group.",
        "en": "Invalid group file ({detail}). Use a UTF-8 CSV with the columns path,group.",
    },
    "resplit.no_group_csv": {
        "pt": "Para dividir por grupo, escolha um arquivo de grupos na prévia.",
        "en": "To split by group, pick a group file in the preview.",
    },
    "resplit.bad_ratios": {
        "pt": "Proporção inválida: use números não negativos com soma maior que zero.",
        "en": "Invalid ratios: use non-negative numbers that sum to more than zero.",
    },
    "resplit.ratio_test_locked": {
        "pt": "O teste está mantido como está; desmarque a opção para dar uma proporção a ele.",
        "en": "The test split is kept as it is; untick the option to give it a ratio.",
    },
    "resplit.nothing_to_split": {
        "pt": "Não há imagens para redistribuir com estas opções.",
        "en": "There are no images to redistribute with these options.",
    },
    "resplit.bad_dest_name": {
        "pt": _BAD_DEST_NAME_PT,
        "en": _BAD_DEST_NAME_EN,
    },
    "resplit.dest_exists": {
        "pt": _DEST_EXISTS_PT,
        "en": _DEST_EXISTS_EN,
    },
    "resplit.base_changed": {
        "pt": "O dataset mudou desde a prévia. Refaça a prévia.",
        "en": "The dataset changed since the preview. Run the preview again.",
    },
    "resplit.group_csv_changed": {
        "pt": "O arquivo de grupos mudou desde a prévia. Refaça a prévia.",
        "en": "The group file changed since the preview. Run the preview again.",
    },
    # ── converting between YOLO and COCO (gui/api/coco_*.py, ADR-169) ────────
    "coco.busy": {
        "pt": "Já existe uma conversão em andamento. Espere ou interrompa a atual.",
        "en": "A conversion is already running. Wait for it or cancel it.",
    },
    "coco.job_missing": {
        "pt": "Conversão não encontrada. Refaça a prévia.",
        "en": "Conversion not found. Run the preview again.",
    },
    "coco.preview_not_ready": {
        "pt": _PREVIEW_NOT_READY_PT,
        "en": _PREVIEW_NOT_READY_EN,
    },
    "coco.base_not_opened": {
        "pt": "Abra o dataset no explorador antes de exportá-lo para COCO.",
        "en": "Open the dataset in the explorer before exporting it to COCO.",
    },
    "coco.base_missing": {
        "pt": _BASE_MISSING_PT,
        "en": _BASE_MISSING_EN,
    },
    "coco.source_missing": {
        "pt": "A pasta do COCO não existe: {path}",
        "en": "The COCO folder does not exist: {path}",
    },
    "coco.base_empty": {
        "pt": "O dataset não tem imagens para exportar.",
        "en": "The dataset has no images to export.",
    },
    "coco.nothing_to_export": {
        "pt": "Nenhuma imagem pode ser exportada: todas falharam ao abrir ou têm rótulo malformado.",
        "en": "No image can be exported: all of them failed to open or have a malformed label.",
    },
    "coco.no_annotations": {
        "pt": "Não achei nenhum arquivo de anotações COCO (.json com images, annotations e categories) na pasta nem em annotations/.",
        "en": "No COCO annotation file (a .json with images, annotations and categories) was found in the folder or in annotations/.",
    },
    "coco.bad_dest_name": {
        "pt": _BAD_DEST_NAME_PT,
        "en": _BAD_DEST_NAME_EN,
    },
    "coco.dest_exists": {
        "pt": _DEST_EXISTS_PT,
        "en": _DEST_EXISTS_EN,
    },
    "coco.base_changed": {
        "pt": "A pasta mudou desde a prévia. Refaça a prévia.",
        "en": "The folder changed since the preview. Run the preview again.",
    },
    "coco.unassigned_split": {
        "pt": "O arquivo '{name}' ainda não tem divisão. Escolha train, val ou test, ou ignore o arquivo.",
        "en": "The file '{name}' has no split yet. Pick train, val or test, or skip the file.",
    },
    "coco.nothing_selected": {
        "pt": "Todos os arquivos foram ignorados: não há nada para importar.",
        "en": "Every file was skipped: there is nothing to import.",
    },
    "coco.unknown_json": {
        "pt": "O arquivo '{name}' não está nesta prévia.",
        "en": "The file '{name}' is not part of this preview.",
    },
    "coco.bad_split": {
        "pt": "Divisão inválida: '{split}'. Use train, val, test ou skip.",
        "en": "Invalid split: '{split}'. Use train, val, test or skip.",
    },
    "coco.split_taken": {
        "pt": "Os arquivos '{first}' e '{second}' foram atribuídos à mesma divisão ({split}). Mude um deles ou ignore-o.",
        "en": "The files '{first}' and '{second}' are assigned to the same split ({split}). Change one or skip it.",
    },
    "coco.category_conflict": {
        "pt": "A categoria {id} se chama '{first}' em um arquivo e '{second}' em outro. Ignore um dos arquivos.",
        "en": "Category {id} is called '{first}' in one file and '{second}' in another. Skip one of the files.",
    },
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
    "credentials.invalid": {
        "pt": "Pedido de credencial inválido.",
        "en": "Invalid credential request.",
    },
    "credentials.empty": {
        "pt": "O valor da credencial não pode ficar vazio.",
        "en": "Credential value cannot be empty.",
    },
    # ── the Colab connection (gui/api/colab_connection.py, ADR-163) ──────────
    "colab.install_missing": {
        "pt": "A instalação terminou, mas a google-colab-cli {version} não foi encontrada.",
        "en": "The install finished, but google-colab-cli {version} was not found.",
    },
    "colab.install_failed": {
        "pt": "Não foi possível instalar a google-colab-cli: {detail}",
        "en": "Failed to install google-colab-cli: {detail}",
    },
    "colab.not_installed": {
        "pt": "A google-colab-cli ainda não está instalada.",
        "en": "google-colab-cli is not installed yet.",
    },
    "colab.not_logged_in": {
        "pt": "Este perfil ainda não entrou numa conta Google.",
        "en": "This profile has not signed in to a Google account yet.",
    },
    "colab.cli_failed": {
        "pt": "O comando colab {command} falhou: {detail}",
        "en": "colab {command} failed: {detail}",
    },
    "colab.cli_timeout": {
        "pt": "O comando colab {command} não respondeu a tempo.",
        "en": "colab {command} did not answer in time.",
    },
    "colab.login_no_url": {
        "pt": "A CLI do Colab não mostrou o endereço de autorização.",
        "en": "The Colab CLI did not print the authorization URL.",
    },
    "colab.login_bad_code": {
        "pt": "O código de autorização está vazio, tem espaços ou é longo demais.",
        "en": "The authorization code is empty, has spaces or is too long.",
    },
    "colab.login_not_started": {
        "pt": "Nenhum login está esperando um código. Comece o login de novo.",
        "en": "No login is waiting for a code. Start the login again.",
    },
    "colab.login_stuck": {
        "pt": "A CLI do Colab não terminou o login.",
        "en": "The Colab CLI did not finish the login.",
    },
    "colab.no_accelerator": {
        "pt": "O Colab não concedeu uma {accelerator} para esta conta: {detail}",
        "en": "Colab did not grant a {accelerator} to this account: {detail}",
    },
    "colab.run_stopped": {
        "pt": "Treino parado: a VM do Colab foi liberada e a época em andamento se perdeu.",
        "en": "Training stopped: the Colab VM was released and the epoch in progress was lost.",
    },
    "colab.kernel_not_ready": {
        "pt": "O kernel do Colab não respondeu depois de {attempts} tentativas; a VM foi liberada.",
        "en": "The Colab kernel did not answer after {attempts} attempts; the VM was released.",
    },
    "colab.run_no_start": {
        "pt": "O treino não começou na VM: nenhum evento em {seconds} s, em {attempts} tentativas. A VM foi liberada.",
        "en": "The training never started on the VM: no event in {seconds} s, over {attempts} attempts. The VM was released.",
    },
    "colab.drive_consent_failed": {
        "pt": "O acesso ao Google Drive não foi permitido em {attempts} tentativas; a VM foi liberada.",
        "en": "Access to Google Drive was not allowed in {attempts} attempts; the VM was released.",
    },
    "colab.drive_mount_failed": {
        "pt": "O Google Drive não montou na VM: {detail}. A VM foi liberada.",
        "en": "Google Drive did not mount on the VM: {detail}. The VM was released.",
    },
    "colab.drive_not_in_drive": {
        "pt": 'A pasta do dataset ({path}) não está dentro do Google Drive para desktop. Informe a pasta "Meu Drive" deste computador ou escolha outra origem.',
        "en": 'The dataset folder ({path}) is not inside Google Drive for desktop. Give this computer\'s "My Drive" folder or choose another source.',
    },
    "colab.drive_empty": {
        "pt": "A pasta do dataset ({path}) não tem arquivos neste computador.",
        "en": "The dataset folder ({path}) has no files on this computer.",
    },
    "colab.drive_not_waiting": {
        "pt": "Esse job não está esperando a permissão do Google Drive.",
        "en": "That job is not waiting for the Google Drive permission.",
    },
    "colab.download_failed": {
        "pt": "O treino terminou, mas o run não veio da VM: {detail}. A VM foi liberada e nada foi gravado nos runs.",
        "en": "The training finished, but the run did not come back from the VM: {detail}. The VM was released and nothing was written to the runs.",
    },
    "colab.bundle_too_large": {
        "pt": "O run na VM tem {size_mb} MB, acima do limite de {limit_mb} MB para importar. A VM foi liberada.",
        "en": "The run on the VM is {size_mb} MB, over the {limit_mb} MB import limit. The VM was released.",
    },
    "colab.import_failed": {
        "pt": "O run veio da VM, mas não pôde ser importado: {detail}. Nada foi gravado nos runs.",
        "en": "The run came back from the VM but could not be imported: {detail}. Nothing was written to the runs.",
    },
    "colab.run_no_end": {
        "pt": "O treino no Colab terminou sem o evento final. Última saída: {detail}",
        "en": "The Colab training ended without its final event. Last output: {detail}",
    },
    "colab.login_rejected": {
        "pt": "O Google não aceitou o código. Comece o login de novo e cole o código novo.",
        "en": "Google did not accept the code. Start the login again and paste the new code.",
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
    "config.lr_multiplier_axes": {
        "pt": (
            "grid_search.lr_mode=suggested_multiplier exige os eixos model.name "
            "e training.learning_rate no grid."
        ),
        "en": (
            "grid_search.lr_mode=suggested_multiplier requires both the model.name "
            "and the training.learning_rate axes."
        ),
    },
    "config.lr_multiplier_value": {
        "pt": (
            "Com lr_mode=suggested_multiplier, os valores de training.learning_rate "
            "são multiplicadores e precisam ser números positivos; recebido {value}."
        ),
        "en": (
            "With lr_mode=suggested_multiplier, training.learning_rate values are "
            "multipliers and must be positive numbers; got {value}."
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
