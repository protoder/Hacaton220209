import NetFromHromosom
'''
    Изначально все инициализируется значениями случайного генератора от 0 до 252
    Формат зависит от типа сети.
    Всегда заголовок сети начинается с:

        0 Кол-во слоев
        1 Оптимизатор
        2 - (Batch Size + 2) * 4? (0..251)
        3 Тип входа - может быть пропущен. Интерпретируется произвольно
        3 или 4 SoftMax/Sigmoid если задан признак Классификация или БинарнаяКлассификация
        3 или 4 или 5 Тип сети, если Тип сети Auto
        3 или 4 или 5 или 6 выходной слой
        
        Наш вариант:
        0 Кол-во слоев
        1 Оптимизатор
        2 (Batch Size + 2) * 4? (0..251) - не используется
        3 SoftMax/Sigmoid - не используется (вместо него - активация выходного слоя)
        4 Размер входных данных (473 + 86*х)
        5-8 - запас на будущее
        9-13 - выходной слой
        
    Выходной сверточный слой:
        0 Activation (расширенный вариант, с softmax)    
        1 kernel_regularizer (0..2)  (0..83 - None, 84 - 167 - L1, 168 - 252 - L2. Число указывает коэффициент. 0.005 + (N+1)/500. ТО есть максимум 0.088)
        2 Activity-regularizer
        3, 4 WinX, WinY - (0..7)+1
        
    Слой - задает Conv слой, Dence или Conv2DTranspose
        0 Neyrons - размер, ((0 - 251) + 1) * 4. Максимум зависит от слоя
        1 Activation
        2 ActivationAfterNorm
        3 BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
        4 kernel_regularizer (0..2)  (0..83 - None, 84 - 167 - L1, 168 - 252 - L2. Число указывает коэффициент. 0.005 + (N+1)/500. ТО есть максимум 0.088)
        5 Activity-regularizer
        6, 7 Для Conv еще WinX, WinY - (0..7)+1

    Блок -
        Заголовок блока
            0 Размер (0..2) = 1
            1 Вид Pooling - Max, Avg или Strides на последнем слое блока
            2 Проброс не прямой, а через слой
            
        Conv слой1
        Conv слой2
        Conv слой3
        TransposeБлок
        1 Conv слой для пробросов

'''

Hromosom = (  #Шапка сети
                  2, #Кол-во слоев
                  4, #Оптимизатор Adam
                  0, BatchSz н.и.
                  1, #SoftMax/Sigmoid н.и.
                  252, # размер входных данных

                  # Выходной блок
                  1, # активация softmax
                  0, #kernel_regularizer (0..2)  (0..83 - None, 84 - 167 - L1, 168 - 252 - L2. Число указывает коэффициент. 0.005 + (N+1)/500. ТО есть максимум 0.088)
                  0, # Activity-regularizer
                  3, 3, # WinX, WinY

                  # Блок 0
                  2, # Размер (0..2) = 1
                  0, #Max Pooling
                  1, # Проброс через слой

                  # Слой 0
                  15, # (1+Нейронов) * 4
                  0, # Активаця relu
                  1, # Активация после BatchNorm
                  84, # Batch Norm
                  0,  # kernel_regularizer (0..2)  (0..83 - None, 84 - 167 - L1, 168 - 252 - L2. Число указывает коэффициент. 0.005 + (N+1)/500. ТО есть максимум 0.088)
                  0,  # Activity-regularizer
                  3, 3,  # WinX, WinY

                  # Слой 1
                  15,  # (1+Нейронов) * 4
                  0,  # Активация relu
                  1,  # Активация после BatchNorm
                  84,  # Batch Norm
                  0,  # kernel_regularizer (0..2)  (0..83 - None, 84 - 167 - L1, 168 - 252 - L2. Число указывает коэффициент. 0.005 + (N+1)/500. ТО есть максимум 0.088)
                  0,  # Activity-regularizer
                  3, 3,  # WinX, WinY

# Слой 1
                  15,  # (1+Нейронов) * 4
                  0,  # Активаця
                  1,  # Активация после BatchNorm
                  84,  # Batch Norm
                  0,  # kernel_regularizer (0..2)  (0..83 - None, 84 - 167 - L1, 168 - 252 - L2. Число указывает коэффициент. 0.005 + (N+1)/500. ТО есть максимум 0.088)
                  0,  # Activity-regularizer
                  3, 3,  # WinX, WinY

1
Activation
2
ActivationAfterNorm
              0, #MaxPooling
              0, # проброс на этот же слой
              0, #  Два проброса - обычный и через слой.

              # Слой 1
              10, # Neyrons - размер, 0 - 251
              1, # Activation
              1, # ActivationAfterNorm
              240, # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              1, # kernel_regularizer (0..2)  (None, L1, L2)
              2, #Activity-regularizer
              3, # WinX
              3,  # WinY - (0..7)+1

              # Слой 2
              25,  # Neyrons - размер, 0 - 251
              4,  # Activation
              0,  # ActivationAfterNorm
              0,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              0,  # kernel_regularizer (0..2)  (None, L1, L2)
              0,  # Activity-regularizer
              3,  # WinX
              5,  # WinY - (0..7)+1

              # Слой 3
              12,  # Neyrons - размер, 0 - 251
              0,  # Activation
              0,  # ActivationAfterNorm
              100,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              2,  # kernel_regularizer (0..2)  (None, L1, L2)
              1,  # Activity-regularizer
              3,  # WinX
              5,  # WinY - (0..7)+1

              # Слой Transpose
              20,  # Neyrons - размер, 0 - 251
              0,  # Activation
              0,  # ActivationAfterNorm
              100,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              2,  # kernel_regularizer (0..2)  (None, L1, L2)
              1,  # Activity-regularizer

              # Блок 1
              3,  # Размер блока
              2,  # Pooling strides
              4,  # Проброс на конечный слой
              0,  # MaxPooling
              0,  # проброс на этот же слой
              0,  # Два проброса - обычный и через слой.

              # Слой 1
              10,  # Neyrons - размер, 0 - 251
              1,  # Activation
              1,  # ActivationAfterNorm
              240,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              1,  # kernel_regularizer (0..2)  (None, L1, L2)
              2,  # Activity-regularizer
              3,  # WinX
              3,  # WinY - (0..7)+1

              # Слой 2
              25,  # Neyrons - размер, 0 - 251
              4,  # Activation
              0,  # ActivationAfterNorm
              0,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              0,  # kernel_regularizer (0..2)  (None, L1, L2)
              0,  # Activity-regularizer
              3,  # WinX
              5,  # WinY - (0..7)+1

              # Слой 3
              12,  # Neyrons - размер, 0 - 251
              0,  # Activation
              0,  # ActivationAfterNorm
              100,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              2,  # kernel_regularizer (0..2)  (None, L1, L2)
              1,  # Activity-regularizer
              3,  # WinX
              5,  # WinY - (0..7)+1

              # Слой Transpose
              20,  # Neyrons - размер, 0 - 251
              0,  # Activation
              0,  # ActivationAfterNorm
              100,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              2,  # kernel_regularizer (0..2)  (None, L1, L2)
              1,  # Activity-regularizer

# Блок 1
              3, #Размер блока
              2, # Pooling strides
              0, # Проброс на конечный слой
              0, #MaxPooling
              0, # проброс на этот же слой
              0, #  Два проброса - обычный и через слой.

              # Слой 1
              10, # Neyrons - размер, 0 - 251
              1, # Activation
              1, # ActivationAfterNorm
              240, # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              1, # kernel_regularizer (0..2)  (None, L1, L2)
              2, #Activity-regularizer
              3, # WinX
              3,  # WinY - (0..7)+1

              # Слой 2
              25,  # Neyrons - размер, 0 - 251
              4,  # Activation
              0,  # ActivationAfterNorm
              0,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              0,  # kernel_regularizer (0..2)  (None, L1, L2)
              0,  # Activity-regularizer
              3,  # WinX
              5,  # WinY - (0..7)+1

              # Слой 3
              12,  # Neyrons - размер, 0 - 251
              0,  # Activation
              0,  # ActivationAfterNorm
              100,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              2,  # kernel_regularizer (0..2)  (None, L1, L2)
              1,  # Activity-regularizer
              3,  # WinX
              5,  # WinY - (0..7)+1

              # Слой Transpose
              20,  # Neyrons - размер, 0 - 251
              0,  # Activation
              0,  # ActivationAfterNorm
              100,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              2,  # kernel_regularizer (0..2)  (None, L1, L2)
              1,  # Activity-regularizer

              # Блок 2
              1,  # Размер блока
              1,  # Pooling strides
              0,  # Проброс на конечный слой
              0,  # MaxPooling
              0,  # проброс на этот же слой
              0,  # Два проброса - обычный и через слой.

              # Слой 1
              10,  # Neyrons - размер, 0 - 251
              1,  # Activation
              1,  # ActivationAfterNorm
              240,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              1,  # kernel_regularizer (0..2)  (None, L1, L2)
              2,  # Activity-regularizer
              3,  # WinX
              3,  # WinY - (0..7)+1

              # Слой 2
              25,  # Neyrons - размер, 0 - 251
              4,  # Activation
              0,  # ActivationAfterNorm
              0,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              0,  # kernel_regularizer (0..2)  (None, L1, L2)
              0,  # Activity-regularizer
              3,  # WinX
              5,  # WinY - (0..7)+1

              # Слой 3
              12,  # Neyrons - размер, 0 - 251
              0,  # Activation
              0,  # ActivationAfterNorm
              100,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              2,  # kernel_regularizer (0..2)  (None, L1, L2)
              1,  # Activity-regularizer
              3,  # WinX
              5,  # WinY - (0..7)+1

              # Слой Transpose
              20,  # Neyrons - размер, 0 - 251
              0,  # Activation
              0,  # ActivationAfterNorm
              100,  # BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
              2,  # kernel_regularizer (0..2)  (None, L1, L2)
              1,  # Activity-regularizer

)

Hr = NetFromHromosom.TNNHromosom(ResType = NetFromHromosom.BinClassification)
Hr.ProcessNet(Hromosom, (120, 120, 1))