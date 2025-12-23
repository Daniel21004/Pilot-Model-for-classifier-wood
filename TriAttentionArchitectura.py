import numpy as np
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_large
import torch.nn as nn
import torch
import torchvision
import logging

weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
ENABLE_LOGS = True

logger = logging.getLogger(__name__)  # Logger específico del módulo
logger.setLevel(logging.INFO)  # INFO o DEBUG

# Solo si no hay handlers configurados aún (evita duplicados)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PositionalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Capas de convolució. Representación de B,C y D (según el paper)
        self.query_conv_B = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv_C = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv_D = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Alfa (Ecuación 2 del paper)
        self.alpha = nn.Parameter(torch.zeros(1)) # Configura un parámetro entrenable. Tensor de dimensión 1 de ceros (valor escalar)

        # Configuración de Softmax
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # Descompresión. Batch_size, Channels, Height and Width
        B,C,H,W = x.size()
        # Para el reshape (C x N, según el paper)
        N = H*W

        
        # Generación de tensores
        # Tensor B.
        # Conv a mapa de características (según paper)
        query_B = self.query_conv_B(x) # Retonar un tensor con la forma [B, C//8, H, W]
        # Reshape -> [B,C,N]
        query_B = query_B.view(B,-1,N) # -1 Índica una inferencia para el calculo automático de canales (hecho por Pytorch)

        # Tensor C
        # Conv a mapa de características (según paper)
        key_C = self.key_conv_C(x)
        # Reshape -> [B,C,N]
        key_C = key_C.view(B,-1,N) # -1 Índica una inferencia para el calculo automático de canales (hecho por Pytorch)

        # Tensor C
        # Conv a mapa de características (según paper)
        value_D = self.value_conv_D(x)
        # Reshape -> [B,C,N]
        value_D = value_D.view(B,-1,N) # -1 Índica una inferencia para el calculo automático de canales (hecho por Pytorch)

        # Operaciones B con C
        # Transpuesta de B y multiplicación con C
        # bmm= Batch Matrix Multiplication
        attention = torch.bmm(query_B.permute(0,2,1),key_C) # La permutación cambia de [B,C,N] a [B,N,C], para que al ser multiplicada por C[B,C,N] dé como resultado [B,N,N]
        # Aplica Softmax -> matriz S
        matrix_S = self.softmax(attention)

        # Operación con D y S
        out = torch.bmm(value_D, matrix_S.permute(0, 2, 1))
        # Reshape de [B,N,N] -> [B,C,H,W]
        out = out.view(B,C,H,W) # Vector global de características

        # Paso final
        out = self.alpha * out + x # Fusión con alfa y el x (mapa de características de entrada, Residual Connection)

        return out

class ChanneAttention(nn.Module):
    def __init__(self, in_channels, reduction_radio=16):
        super().__init__()

        # Seteo de la configuración del pooling. Salida 1x1 [B,C,1,1]
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # Multi-layer Perceptron (MLP) con reducción de dimensionalidad
        # Primera capa que representa a W_0 ∈ R^(C/r × C)
        self.fc1 = nn.Linear(in_channels, in_channels//reduction_radio, bias=True)
        # Segunda capa que representa a W_1 ∈ R^(C × C/r)
        self.fc2 = nn.Linear(in_channels//reduction_radio, in_channels, bias=True)

        # ReLU: Función de activación entre capas
        self.relu = nn.ReLU(inplace=True)


        # Batch normalization. 
        self.bn = nn.BatchNorm2d(in_channels) # in_channels siendo el número de características de la capa anterior (fc2, salida de MLP)

        # Sigmoid, para la normalización de pesos de atención
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extracción de las dimensiones
        B,C,_,_ = x.size() # Extraemos Batch y Channel

        # Operaciones
        # Avg pooling para compresión de la información. Conseguimos F_c ∈ R^(C×1×1)
        y = self.avg_pooling(x)
        # Aplanamiento para pasar por el MLP
        y = y.view(B,C) # -> [B,C]

        # Paso por MLP
        # Primera capa, aplicamos. W0 ∈ R^(C/r×C)
        y = self.fc1(y) # [B,C/r]
        y = self.relu(y)
        # Segunda capa, aplicamos W1 ∈ R^(C×C/r)
        y = self.fc2(y) # [B,C]

        # Reshape para aplicar Batch Normalization
        y = y.view(B,C,1,1) # [B,C,1,1]

        # Aplicamos Batch Normalization
        y = self.bn(y)

        # Por último, obtenemos pesos de atención
        y = self.sigmoid(y)

        # Aplicamos la atención (elemento a elemento)
        return x * y.expand_as(x) # y intenta ser de las mismas dimensiones que el tensor x. Solo si es posible.

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_radio=16):
        super().__init__()

        # Estructura Bottleneck 
        # f0 1x1
        self.conv_1 = nn.Conv2d(in_channels,
                                in_channels//reduction_radio, 
                                kernel_size=1, 
                                bias=False)
        # f1 3x3 y f2 3x3
        self.conv_3 = nn.Conv2d(in_channels//reduction_radio, 
                                in_channels//reduction_radio, 
                                kernel_size=3, 
                                padding=2, 
                                dilation=2, 
                                bias=False)
        # f3 1x1
        self.conv_last_1 = nn.Conv2d(in_channels//reduction_radio,
                                    1, 
                                     kernel_size=1,
                                     bias=False)
        
        # Batch Normalización para la estabilización
        self.bn_reduction_radio =nn.BatchNorm2d(in_channels//reduction_radio)
        self.bn1 =nn.BatchNorm2d(1)

        # Función de activación ReLU
        self.relu = nn.ReLU(inplace=True)

        # Función Sigmoid para obtener los pesos ya estabilizados
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Obtención del tamaño del tensor
        B,C,H,W = x.size()

        # Primera convolución. Reducción de dimensionalidad. (f0 1x1)
        # De C canales a C/r
        y = self.conv_1(x) # [B,C/r,H,W]
        y = self.bn_reduction_radio(y)
        y = self.relu(y)

        # Segunda convolución. (f1 3x3)
        # Dilatación 
        y = self.conv_3(y)
        y = self.bn_reduction_radio(y)
        y = self.relu(y)

        # Tercera Convolución . (f1 3x3)
        # Dilatación (Más información contextual)
        y = self.conv_3(y)
        y = self.bn_reduction_radio(y)
        y = self.relu(y)

        # Ultima convolución,
        # Genera M_s(F) ∈ R^(H×W)
        y = self.conv_last_1(y) # se obtiene [B,1,H,W] = [H,W]
        y = self.bn1(y)

        # Estandarizamos los pesos con Sigmoid
        out = self.sigmoid(y) # [B,1,H,W]

        # Aplicamos la atención espacial con la salida y la entrada
        return x * out

class TriAttention(nn.Module):
    """
    Clase que implementa las tres clases anteriores [PositionalAttention, ChannelAttention, y SpatialAttention]
    """
    def __init__(self, in_channels, reduction_radio=16):
        super().__init__()

        # INICIALIZACION - Ramas de anteción
        # Inicialización de la Attention posicional
        self.positional_attention = PositionalAttention(in_channels=in_channels)
        # Inicialización de la Attention por canal
        self.channel_attention = ChanneAttention(in_channels=in_channels,reduction_radio=reduction_radio)
        # Inicialización de la Attention spacial
        self.spatial_attention = SpatialAttention(in_channels=in_channels,reduction_radio=reduction_radio)

        
        # FUSIÓN DE LOS TRES MECANISMOS
        # Capa de convolución para juntar los tres canales y que recupere los canales originales
        self.fusion_conv = nn.Conv2d(in_channels*3, in_channels, kernel_size=1)
        # Capa para Batch Normalizarion
        self.bn = nn.BatchNorm2d(in_channels)
        # Función de activación ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x : [B,C,H,W]

        # Aplicamos al mapa de características los ramas de atención de forma independiente
        # Salida de la rama de atención Posicional
        positional_out = self.positional_attention(x)
        # Salida de la rama de atención Espacial
        spatial_out = self.spatial_attention(x)
        # Salida de la rama de atención de Canal
        channel_out = self.channel_attention(x)

        # Concatenamos las tres salidas
        out_concat = torch.cat([positional_out, spatial_out, channel_out], dim=1) # [B,3C,H,W]

        # Fusionamos en un solo vector
        fused = self.fusion_conv(out_concat) # [B,3C,H,W] -> [B,C,H,W]
        fused = self.bn(fused) # Norlaizamos las salidas
        fused = self.relu(fused) # Aplicamos la activación con ReLU

        # Aplicamos una conexión residual para estabilizar el entrenamiento
        return fused + x
        
        

class WoodClassifierWithTriAttention(nn.Module):
    """
        Clase que implementa el mecanismo de atención en la arquitectura (backbone) MobileNetV3.
        Por defecto, use_tri_attention es igual a True, por lo que se implementa el mecanismo al Backbone.
        En caso de ser Falso, simplemente se modifica la última capa. Sin Tri Attention
    """

    def __init__(self, num_classes=4, use_tri_attention=True):
        super().__init__()

        # Se extrae el backbone (completo, con clasificador) de MobileNetV3
        self.backbone = mobilenet_v3_large(weights=weights)
    
        # Se extraen las características (sin clasificador)
        self.features = self.backbone.features

        # Banderas para aplicar mecanismo de atention
        self.use_tri_attention = use_tri_attention

        # 3 módulos de Tri Attention
        if self.use_tri_attention:
            self.tri_1 = TriAttention(24, reduction_radio=8) # Aplicación en la capa 3
            self.tri_2 = TriAttention(80, reduction_radio=8) # Aplicación en la capa 10
            self.tri_3 = TriAttention(960, reduction_radio=16) # Aplicación en la capa 17 (Ultima antes del clasificador)

        # MODIFICACIÓN CLASIFICADOR FINAL
        # Obtenemos las características de entrada de esa capa Linear
        in_features = self.backbone.classifier[3].in_features
        # Modificamos solo la última capa (la que clasifica) a las característiacs de clasificación = número de clases
        self.backbone.classifier[3] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # Reducción a [B,C,1,1] (para Tri Attention)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Inicializamos los pesos de TODO la red
        self._initialize_weights()

    def forward(self, x):
        # x -> [B,3,224,224] (Tamaño de entrada para MobileNetV3)

        # En caso 'use_tri_attention' == false. Se hace la clasificación con el modelo original
        if not self.use_tri_attention:
            if ENABLE_LOGS: logger.info(f"SIN mecanismo de atención. Clasificador original") # <- LOG
            x = self.backbone(x)
            return x

        
        # Solo se ejecuta siempre que 'use_tri_attention' == true
        if ENABLE_LOGS: logger.info(f"Tri Attention está activado") # <- LOG
        for i, layer in enumerate(self.features):
            x = layer(x)

            if i == 3:
                x = self.tri_1(x)
            elif i == 10:
                x = self.tri_2(x)
            elif i == 17:
                x = self.tri_3(x)

        # Global Avarage Pooling
        if ENABLE_LOGS: logger.info(f"AVG Pooling y Aplanación") # <- LOG
        x = self.avgpool(x) # [B,C,1,1]
        x = torch.flatten(x, 1) # [B, 960]

        # Clasificación con el clasificador de la arquitectura, puesto que se modificó el original
        x = self.backbone.classifier(x) # [B, [num_clases]]
        if ENABLE_LOGS: logger.info(f"Clasificación finalizada") # <- LOG
        return x
    
    
    def _initialize_weights(self):
        """
        Inicializa los pesos del clasificador y los módulos de a
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming para capas convolucionales 
                # Es óptima para funciones de activación ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Para BatchNorm2D -> weight=1 y bias=0 Es el estándar
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Para las capas lineales, una configuración pequeña
                nn.init.normal_(m.weight,0 ,0.01)
                nn.init.constant_(m.bias, 0)
        if ENABLE_LOGS: logger.info(f"Inicialización de pesos (Arquitectura MobileNetV3)")


# Forma de llamarlo
# model =     WoodClassifierWithTriAttention(num_classes=4,use_tri_attention=False)
# print("todobien")

