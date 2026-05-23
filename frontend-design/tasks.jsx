// Per-task definitions. Each task has:
//  - key, label, accent (used to drive the body data attribute and tab color)
//  - models (for the "modelos" selector in the sketch)
//  - params (rendered as a grid of fields)
//  - defaults
//
// The Tasks export is consumed by app.jsx to render the active task pane and
// the history mock.

const TASKS = [
  {
    key: 'classification',
    label: 'Classificação',
    short: 'class',
    description: 'Categorize imagens em rótulos discretos',
    accent: '#f16363',
    models: [
      { value: 'resnet50',     label: 'ResNet-50',         sub: 'imagenet · 25.6M params' },
      { value: 'resnet18',     label: 'ResNet-18',         sub: 'leve · 11.7M params' },
      { value: 'efficientnet', label: 'EfficientNet-B3',   sub: 'eficiente · 12M params' },
      { value: 'convnext',     label: 'ConvNeXt-Tiny',     sub: 'moderno · 28M params' },
      { value: 'vit_b16',      label: 'ViT-B/16',          sub: 'transformer · 86M params' },
      { value: 'mobilenetv3',  label: 'MobileNet-v3',      sub: 'edge · 5.4M params' },
    ],
    defaults: {
      modelo: 'resnet50',
      epocas: 50,
      lr: 0.001,
      batch: 32,
      optimizer: 'adam',
      loss: 'cross_entropy',
      val_split: 0.2,
      augment: true,
      pretrained: true,
    },
    params: [
      { type: 'number',    key: 'epocas',    label: 'Épocas',           hint: 'iterações', min: 1, max: 1000, step: 1 },
      { type: 'number',    key: 'lr',        label: 'Learning Rate',    hint: 'lr',        min: 0.000001, max: 1, step: 0.0001 },
      { type: 'number',    key: 'batch',     label: 'Batch size',       hint: 'amostras',  min: 1, max: 1024, step: 1 },
      { type: 'segmented', key: 'optimizer', label: 'Otimizador',
        options: [{value:'adam',label:'Adam'},{value:'sgd',label:'SGD'},{value:'adamw',label:'AdamW'}] },
      { type: 'segmented', key: 'loss', label: 'Função de perda',
        options: [{value:'cross_entropy',label:'Cross-Ent'},{value:'focal',label:'Focal'},{value:'label_smooth',label:'Smooth'}] },
      { type: 'number',    key: 'val_split', label: 'Validation split', hint: '0–1',       min: 0, max: 0.5, step: 0.05 },
      { type: 'toggle',    key: 'augment',   label: 'Data augmentation' },
      { type: 'toggle',    key: 'pretrained',label: 'Pesos pré-treinados' },
    ],
  },
  {
    key: 'detection',
    label: 'Detecção de Objeto',
    short: 'detect',
    description: 'Localize e classifique objetos com bounding boxes',
    accent: '#48cf8e',
    models: [
      { value: 'yolov8n',  label: 'YOLOv8-n',    sub: 'nano · 3.2M · 80 fps' },
      { value: 'yolov8s',  label: 'YOLOv8-s',    sub: 'small · 11.2M · 50 fps' },
      { value: 'yolov8m',  label: 'YOLOv8-m',    sub: 'medium · 25.9M · 28 fps' },
      { value: 'detr',     label: 'DETR-R50',    sub: 'end-to-end · 41M' },
      { value: 'rtdetr',   label: 'RT-DETR',     sub: 'real-time · 32M' },
      { value: 'faster',   label: 'Faster R-CNN',sub: 'two-stage · 41.7M' },
    ],
    defaults: {
      modelo: 'yolov8s',
      epocas: 100,
      lr: 0.01,
      batch: 16,
      img_size: 640,
      conf_threshold: 0.25,
      iou_threshold: 0.45,
      anchors: 'auto',
      mosaic: true,
    },
    params: [
      { type: 'number',    key: 'epocas',         label: 'Épocas',           hint: 'iterações', min: 1, max: 1000, step: 1 },
      { type: 'number',    key: 'lr',             label: 'Learning Rate',    hint: 'lr',        min: 0.000001, max: 1, step: 0.0001 },
      { type: 'number',    key: 'batch',          label: 'Batch size',       hint: 'amostras',  min: 1, max: 256, step: 1 },
      { type: 'number',    key: 'img_size',       label: 'Image size',       hint: 'px',        min: 64,  max: 2048, step: 32 },
      { type: 'number',    key: 'conf_threshold', label: 'Confidence',       hint: '0–1',       min: 0, max: 1, step: 0.05 },
      { type: 'number',    key: 'iou_threshold',  label: 'IoU threshold',    hint: '0–1',       min: 0, max: 1, step: 0.05 },
      { type: 'segmented', key: 'anchors',        label: 'Anchors',
        options: [{value:'auto',label:'Auto'},{value:'fixed',label:'Fixo'},{value:'k_means',label:'K-means'}] },
      { type: 'toggle',    key: 'mosaic',         label: 'Mosaic augment' },
    ],
  },
  {
    key: 'regression',
    label: 'Regressão',
    short: 'reg',
    description: 'Estime valores contínuos a partir das entradas',
    accent: '#5b9fff',
    models: [
      { value: 'mlp',          label: 'MLP (3 layers)',     sub: 'baseline · ~250k params' },
      { value: 'mlp_deep',     label: 'MLP-Deep (6 layers)',sub: 'profundo · 1.2M params' },
      { value: 'resnet_reg',   label: 'ResNet-Reg',         sub: 'cnn para reg de imagem' },
      { value: 'xgboost',      label: 'XGBoost',            sub: 'tabular · gradient boosting' },
      { value: 'tabnet',       label: 'TabNet',             sub: 'atenção tabular' },
      { value: 'random_forest',label: 'Random Forest',      sub: 'ensemble · 100 árvores' },
    ],
    defaults: {
      modelo: 'mlp',
      epocas: 200,
      lr: 0.0005,
      batch: 64,
      loss: 'mse',
      val_split: 0.2,
      early_stop: 20,
      scaler: 'standard',
      shuffle: true,
    },
    params: [
      { type: 'number',    key: 'epocas',     label: 'Épocas',           hint: 'iterações', min: 1, max: 5000, step: 1 },
      { type: 'number',    key: 'lr',         label: 'Learning Rate',    hint: 'lr',        min: 0.000001, max: 1, step: 0.0001 },
      { type: 'number',    key: 'batch',      label: 'Batch size',       hint: 'amostras',  min: 1, max: 1024, step: 1 },
      { type: 'segmented', key: 'loss',       label: 'Loss function',
        options: [{value:'mse',label:'MSE'},{value:'mae',label:'MAE'},{value:'huber',label:'Huber'}] },
      { type: 'number',    key: 'val_split',  label: 'Validation split', hint: '0–1',       min: 0, max: 0.5, step: 0.05 },
      { type: 'number',    key: 'early_stop', label: 'Early stop',       hint: 'paciência (épocas)', min: 0, max: 200, step: 1 },
      { type: 'segmented', key: 'scaler',     label: 'Scaler',
        options: [{value:'standard',label:'Standard'},{value:'minmax',label:'MinMax'},{value:'none',label:'Nenhum'}] },
      { type: 'toggle',    key: 'shuffle',    label: 'Shuffle dataset' },
    ],
  },
  {
    key: 'segmentation',
    label: 'Segmentação',
    short: 'seg',
    description: 'Máscaras a nível de pixel para cada classe',
    accent: '#b079ff',
    models: [
      { value: 'unet',     label: 'U-Net',          sub: 'clássico · 31M params' },
      { value: 'deeplab',  label: 'DeepLabV3+',     sub: 'estado-da-arte · 41M' },
      { value: 'mask2',    label: 'Mask2Former',    sub: 'transformer · 47M' },
      { value: 'sam',      label: 'SAM-B (fine-tune)', sub: 'segment-anything base' },
    ],
    defaults: {
      modelo: 'unet',
      epocas: 80,
      lr: 0.0003,
      batch: 8,
      img_size: 512,
      loss: 'dice',
      classes: 4,
      tta: false,
    },
    params: [
      { type: 'number',    key: 'epocas',   label: 'Épocas',         hint: 'iterações', min: 1, max: 1000, step: 1 },
      { type: 'number',    key: 'lr',       label: 'Learning Rate',  hint: 'lr', min: 0.000001, max: 1, step: 0.0001 },
      { type: 'number',    key: 'batch',    label: 'Batch size',     hint: 'amostras', min: 1, max: 64, step: 1 },
      { type: 'number',    key: 'img_size', label: 'Image size',     hint: 'px', min: 64, max: 2048, step: 32 },
      { type: 'segmented', key: 'loss',     label: 'Loss',
        options: [{value:'dice',label:'Dice'},{value:'tversky',label:'Tversky'},{value:'bce',label:'BCE+Dice'}] },
      { type: 'number',    key: 'classes',  label: 'Nº de classes',  hint: 'inteiro', min: 1, max: 100, step: 1 },
      { type: 'toggle',    key: 'tta',      label: 'Test-time augment' },
    ],
  },
];

// Mock recent training runs for history modal
const HISTORY = [
  {
    id: 'rn-7c41',
    task: 'classification',
    name: 'cls_resnet50_fundus_v3',
    model: 'ResNet-50',
    finishedAt: '14 Mai · 11:32',
    epochs: 50,
    finalLoss: 0.082,
    finalMetric: { label: 'Top-1 acc', value: '94.7%' },
    duration: '38m 12s',
    device: 'CUDA · RTX 4090',
    status: 'success',
    curve: makeCurve(50, 1.8, 0.082, 12),
  },
  {
    id: 'rn-7c39',
    task: 'detection',
    name: 'det_yolov8s_pcb_defects',
    model: 'YOLOv8-s',
    finishedAt: '13 Mai · 22:04',
    epochs: 120,
    finalLoss: 0.0413,
    finalMetric: { label: 'mAP@50', value: '0.871' },
    duration: '2h 14m',
    device: 'CUDA · RTX 4090',
    status: 'success',
    curve: makeCurve(120, 2.4, 0.041, 18),
  },
  {
    id: 'rn-7c2b',
    task: 'regression',
    name: 'reg_mlp_house_prices',
    model: 'MLP-Deep',
    finishedAt: '13 Mai · 09:51',
    epochs: 200,
    finalLoss: 0.0114,
    finalMetric: { label: 'R²', value: '0.912' },
    duration: '6m 41s',
    device: 'CPU · 16 cores',
    status: 'success',
    curve: makeCurve(200, 0.6, 0.0114, 22),
  },
  {
    id: 'rn-7c1d',
    task: 'classification',
    name: 'cls_vit_b16_satellite',
    model: 'ViT-B/16',
    finishedAt: '12 Mai · 17:18',
    epochs: 80,
    finalLoss: 0.123,
    finalMetric: { label: 'Top-1 acc', value: '89.2%' },
    duration: '1h 47m',
    device: 'CUDA · RTX 4090',
    status: 'success',
    curve: makeCurve(80, 1.6, 0.123, 10),
  },
  {
    id: 'rn-7c0a',
    task: 'detection',
    name: 'det_rtdetr_traffic',
    model: 'RT-DETR',
    finishedAt: '12 Mai · 02:28',
    epochs: 60,
    finalLoss: 0.092,
    finalMetric: { label: 'mAP@50', value: '0.762' },
    duration: '52m 33s',
    device: 'CUDA · RTX 4090',
    status: 'failed',
    curve: makeCurve(60, 3.0, 0.84, 24),
  },
  {
    id: 'rn-7bf8',
    task: 'segmentation',
    name: 'seg_unet_microscopy',
    model: 'U-Net',
    finishedAt: '10 Mai · 14:02',
    epochs: 80,
    finalLoss: 0.0317,
    finalMetric: { label: 'Dice', value: '0.926' },
    duration: '1h 12m',
    device: 'CUDA · RTX 4090',
    status: 'success',
    curve: makeCurve(80, 1.2, 0.031, 16),
  },
];

function makeCurve(n, start, end, noise) {
  const arr = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    // exponential decay
    const base = end + (start - end) * Math.exp(-3.2 * t);
    const wiggle = (Math.sin(i * 0.6) + Math.cos(i * 0.31)) * (start / noise) * (1 - t);
    arr.push(Math.max(0.001, base + wiggle));
  }
  return arr;
}

window.TASKS = TASKS;
window.HISTORY = HISTORY;
