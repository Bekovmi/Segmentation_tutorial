tutorial.md
# Как учить модели в скиптах с помощью с Сonfig API Catalyst
Привет, для воспроизводимости и версионирования пайплайна опытные DS сохраняют все параметры в YAML или JSON файлы. Catalyst предоставляет такой функционал прямо из коробки,  в этом туториале я покажу на примере [Segmentation tutorial](https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb), как обернуть его в Config API.
В качестве основы возьмем Segmentation tutorial и последовательно шаг за шагом перепишем. Сначала разобьем jupyter-notebook на составные части: 
1) Настройка виртуального окружения
2) Подготовка данных
3) Загрузка данных в модель 
4) Обучение модели 
5) Анализ результатов обучения
6) Предсказание результатов

Перед тем, как начать писать код, настроим и активируем виртуальное окружение. Вот здесь есть небольшой [туториал](https://python-scripts.com/virtualenv). 
```
mkdir segmentation_tutorial
cd segmentation_tutorial
virtualenv venv
source venv/bin/activate

```
Установим необходимые библиотеки для работы с данным туториалом
```
pip install albumentations # Библиотека для работы с аугментациями
pip install -U catalyst # Библиотека для постороения автоматизированных пайплайнов
pip install segmentation-models-pytorch # Библиотека в которой содержатся архитектуры и веса моделей нейронных сетей для решения задач сегментации
pip install ttach библиотека с оберткой для test-time аргументаций
```
Для того чтобы обучать модель в режиме fp16 необходимо установить библиотеку Apex.
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

После того, как нужные библиотеки установлены необходимо подготовить данные для запуска пайплайна. Стоит отметить, что при установке Catalyst устанавливается bash функция download-gdrive для загрузки данных с гугл диск, которая принимает ключ. Данные для данного туториала можно скачать данные по следующему ключу 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj.
Напишем bash-скрипт download_dataset.sh для скачивания данных:
```
download-gdrive 1iYaNijLmzsrMlAdMoUEhhJuo-5bkeAuj segmentation_data.zip
mkdir data
unzip segmentation_data.zip &>/dev/null 
mv segmentation_data/* data
rm segmentation_data.zip
```
и запустим
```
bash download_dataset.sh
```
Для дальнейшей работы нужно сформировать два csv файла - один для тренировочной выборки, а другой для теста. В csv файле будет два столбца - один для хранения пути к картинке, а другой для пути к маске. 
```
prepare_dataset.py
import pathlib
import pandas as pd
from sklearn import model_selection
def main(
    datapath: str = "./data", valid_size: float = 0.2, random_state: int = 42
):
    datapath = pathlib.Path(datapath)
    dataframe = pd.DataFrame({
        "image": sorted((datapath / "train").glob("*.jpg")),
        "mask": sorted((datapath / "train_masks").glob("*.gif"))
    })
    df_train, df_valid = model_selection.train_test_split(
        dataframe,
        test_size=valid_size,
        random_state=random_state,
        shuffle=False
    )
    for source, mode in zip((df_train, df_valid), ("train", "valid")):
        source.to_csv(datapath / f"dataset_{mode}.csv", index=False)
if __name__ == "__main__":
```
  
Подготовим данные:
```
python prepare_dataset.py --datapath=./data
```

Далее создадим папку src в которой находятся файлы с кодом частей пайплайна.
В папке должно находится 4 файла: __init__.py, dataset.py, experiment.py, transforms.py.

1) __init__.py - служит для указания интерпретатору того, что папка в которой он лежит является питон-пакетом и в ней можно искать модули (файлы). В каталист мире этот файл так же используется для добавления кастомных модулей в регистри. В нем нужно обязательно указать какой эксперимент и раннер ты используешь.

```
from .experiment import Experiment
from catalyst.dl import SupervisedRunner as Runner
```

2) dataset.py - файл в котором прописывается класс dataset по аналогии с классом dataset в оригинальном туториале
```
from typing import Callable, Dict, List
import numpy as np
import imageio
from torch.utils.data import Dataset
class SegmentationDataset(Dataset):
    def __init__(self, list_data: List[Dict], dict_transform: Callable = None):
        self.data = list_data
        self.dict_transform = dict_transform
    def __getitem__(self, index: int) -> Dict:
        dict_ = self.data[index]
        dict_ = {
            "image": np.asarray(imageio.imread(dict_["image"], pilmode="RGB")),
            "mask": np.asarray(imageio.imread(dict_["mask"]))}
        if self.dict_transform is not None:
            dict_ = self.dict_transform(**dict_)
        return dict_
    def __len__(self) -> int:
        return len(self.data)
```
3) В experiment.py - содержится класс Experiment, он наследуется от класса ConfigExperiment. Данный класс содержит два класса:
  get_transforms - метод выполняет аугментации изображений, которые описаны в transform.py на различных этапах (Тренировка, Валидация, Инференс)
  get_datasets - метод, который возвращает словарь {"image": ..., "mask":...} из предварительно подготовленных csv файлов.


```
iimport collections
import pandas as pd
from catalyst.dl import ConfigExperiment
from .dataset import SegmentationDataset
from .transforms import (
    pre_transforms, post_transforms, hard_transform, Compose
)
class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(
        stage: str = None, mode: str = None, image_size: int = 256,
    ):
        pre_transform_fn = pre_transforms(image_size=image_size)
        if mode == "train":
            post_transform_fn = Compose([
                hard_transform(image_size=image_size),
                post_transforms()
            ])
        elif mode in ["valid", "infer"]:
            post_transform_fn = post_transforms()
        else:
            raise NotImplementedError()
        
        transform_fn = Compose([pre_transform_fn, post_transform_fn])
        return transform_fn
    def get_datasets(
        self,
        stage: str,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        image_size: int = 256,
    ):
        datasets = collections.OrderedDict()
        for source, mode in zip(
            (in_csv_train, in_csv_valid), ("train", "valid")
        ):
            dataframe = pd.read_csv(source)
            datasets[mode] = SegmentationDataset(
                dataframe.to_dict('records'),
                dict_transform=self.get_transforms(
                    stage=stage, mode=mode, image_size=image_size
                ),
            )
        return datasets
```
4) В файле transforms.py  содержатся функции, которые определяют препроцессинг, аугментации и постпроцессинг изображениий.
```
import cv2
from albumentations import (
    Compose, LongestMaxSize, PadIfNeeded, Normalize, ShiftScaleRotate,
    RandomGamma
)
from albumentations.pytorch import ToTensor
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def pre_transforms(image_size=256):
    return Compose([
        LongestMaxSize(max_size=image_size),
        PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
    ])

def post_transforms():
    return Compose([Normalize(), ToTensor()])

def hard_transform(image_size=256):
    return Compose([
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
    ])
```
Стоит отдельно упомянуть про Registry. В Catalyst уже реализовано несколько моделей, колбэков, лоссов, оптимайзеров. Если пользователь хочет использовать свои модели, колбэки, лоссы, оптимайзеры он может самостоятельно добавить их в Registry при помощи следующего API - `Callback(SomeCallback)`, `Model(MyModel)`, `Criterion(GreatLoss)`, `Optimizer(NewAwesomeOptimizer)`, `registry.Scheduler(EvenScheduler)`. Чтобы потом это название использовать в Config.yml. В качестве примера создадим модель классификации SimpleNet в файле model.py 
model.py
```
import torch.nn as nn
import torch.nn.functional as F
from catalyst.dl import reqistry
@registry.Model # Этот декоратор региструет модель 
class SimpleNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2D(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
  return x
```
После этого в поле model config файла можно использовать имя SimpleModel.
Полное описание полей в config.yml можно найти здесь https://github.com/catalyst-team/catalyst/tree/master/examples/configs. Обращаю внимание, что в поле runner_params в качестве параметров для задачи сегментации указываем - input_key - "image", output_key - "logits". 

По данной ссылке доступно полное описание полей [config](https://github.com/catalyst-team/catalyst/tree/master/examples/configs)
На его примере напишем config.yml, в котором укажем необходимые параметры для запуска моделей:
```
model_params:                      
  model: ResnetFPNUnet             
  num_classes: 1                   
  arch: resnet18                   
  pretrained: True                 
args:
  expdir: "src"                    
  logdir: "logs" 
runner_params:  # OPTIONAL KEYWORD, params for Runner's init
  input_key: "image"
  output_key: "logits"                
stages:                                    
  state_params:                             
    main_metric: &reduce_metric dice        
    minimize_metric: False
  data_params:                                  
    num_workers: 0                              
    batch_size: 64                              
    in_csv_train: "./data/dataset_train.csv"   
    in_csv_valid: "./data/dataset_valid.csv"    
    image_size : 256                            
  criterion_params:
    criterion: DiceLoss                                            
  stage1:                                       
    state_params:                               
      num_epochs: 10                           
    optimizer_params:                           
      optimizer: Adam                          
      lr: 0.001                                
      weight_decay: 0.0003                      
    scheduler_params:                           
      scheduler: MultiStepLR                    
      milestones: [10]                          
      gamma: 0.3                                
    callbacks_params:                           
      loss_dice:                               
        callback: CriterionCallback             
        input_key: mask                         
        output_key: logits
        prefix: &loss loss_dice                  
      
      accuracy:                                 
        callback: DiceCallback                  
        input_key: mask
        output_key: logits
      optimizer:                           
        callback: OptimizerCallback
        loss_key: *loss
      scheduler:                              
        callback: SchedulerCallback
        reduce_metric: *reduce_metric
      saver:                                 
        callback: CheckpointCallback
```
После чего можно запустить процесс обучения:
```
catalyst-dl run --config=./config.yml --verbose
```
Результаты обучения модели можно посмотреть с использованием tensorboard при помощи следующей комманды:
```
tensorboard --logdir ./logs
```
После обучения веса моделей можно найти в ./logs/checkpoints и с помощью их делать инференс согласно этому [туториалу](https://pytorch.org/tutorials/beginner/saving_loading_models.html).
Но для продакшена, я рекомендую другой способ, а именно сделать jit trace, чтобы получить бинарник. Это позволяет ускорить инференс и загружать модели на С++. Подробнее об этом [здесь](https://github.com/catalyst-team/catalyst-info#catalyst-info-2-tracing-with-torchjit).
Для того, чтобы сделать trace, нужно запустить следующую комманду:
```
catalyst-dl trace ./logs
```
В папке logs/trace/ и будут находиться веса модели - traced-best-forward.pth.
Для инференса напишем следующий класс для инициализации модели и метода предикт и сохраним его в inference.py:
```
import numpy as np
import cv2
import torch.nn as nn
import torch 
import os
from catalyst.dl import utils
from transforms import (pre_transforms, post_transforms, Compose)
class Predictor:
    def __init__(self, model_path,image_size):
        self.augmentation =Compose([pre_transforms(image_size=image_size),post_transforms()])
        self.m = nn.Sigmoid()
        self.model = utils.load_traced_model(model_path).cuda()
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.image_size = image_size
    def predict(self, image,threshold):
        augmented = self.augmentation(image=image)
        inputs = augmented['image']
        inputs = inputs.unsqueeze(0).to(self.device) 
        output = self.m(self.model(inputs)[0][0]).cpu().numpy()
        probability = (output > threshold).astype(np.uint8)
        return probability
```
Для предсказания масок картинок запишем следующий скрипт и сохраним его в CarSegmentation.py:
```
from inferens import Predictor
import argparse
import pandas as pd
import os
import cv2
import numpy as np
def Car_Segmenatation(path_in, path_out, path_model,threshold,image_size):
    mask_predictor = RemoveBackground(path_model,image_size)
    data = pd.DataFrame({"image":os.listdir(path_in)})
    for image_file in data.iloc[:,0]:
        image = cv2.resize(cv2.imread(os.path.join(path_in,image_file)), (image_size,image_size))
        probability = mask_predictor.predict(image, threshold=0.5)
        image[probability==0]=[255,255,255]
        cv2.imwrite(os.path.join(path_out,image_file),image)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'folders to files')
    parser.add_argument('path_in', type=str, help='dir with image')
    parser.add_argument('path_out', type=str, help='output dir')
    parser.add_argument('path_model', type = str, help = 'path_model')
    parser.add_argument('threshold', type = float, help = 'threshold')
    parser.add_argument('image_size', type = int, help = 'image_size')
    args = parser.parse_args()
    Car_Segmenatation(args.path_in,args.path_out,args.path_model,args.threshold,args.image_size)


```
Запустим скрипт 
```
python CarSegmentation.py path_in,path_out,path_model,treshold, image_size
```
Таким образом, получилось создать полноценный пайплайн тренировки и инференса без использованию juputer notebook, позволяющий запускать обучение меняя лишь параметры config файла. Полный код тренировки и инференса доступен в этом [репозитории](https://github.com/Bekovmi/Segmentation_tutorial).


Выражаю благодарность [Евгению Качану](https://github.com/bagxi), [Роману Тезикову](https://github.com/TezRomacH), [Сергею Колесникову](https://github.com/Scitator), [Алексею Газиеву](https://github.com/gazay) за помощь при написании статьи. 












